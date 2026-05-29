#!/usr/bin/env python3
"""
Clean inference API for a trained HGNN checkpoint.

Usage:
    from scripts.infer_api import HGNNInference
    api = HGNNInference.from_run_dir("runs/c1_hgnn_baseline/20260528_230106")
    result = api.infer(image_path="photo.jpg")
    print(result["leaf_label"])        # predicted leaf class name
    print(result["leaf_probs"])        # softmax probs over 37 leaves
    print(result["path_nodes"])        # full hierarchical path as list of names
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# --------------------------------------------------------------------------- #
# Imports with repo-root insertion
# --------------------------------------------------------------------------- #
import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_repo_root / "gnn_classifier") not in sys.path:
    sys.path.insert(0, str(_repo_root / "gnn_classifier"))

from gnn_classifier.hgnn import HGNN
from taxonomy.tree import get_hierarchy_levels, get_taxonomy

# --------------------------------------------------------------------------- #
# Helpers (duplicated here to keep the file self-contained)
# --------------------------------------------------------------------------- #


def _resolve_device(device_str: str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _canonicalize_graph(graph: nx.DiGraph, node_order: List[str]) -> nx.DiGraph:
    new_graph = nx.DiGraph()
    for node in node_order:
        new_graph.add_node(node, **graph.nodes[node])
    for u, v in graph.edges:
        new_graph.add_edge(u, v, **graph.edges[u, v])
    return new_graph


def _compute_hierarchy_levels(graph: nx.DiGraph, node_to_idx: Dict[str, int]) -> List[np.ndarray]:
    levels_dict = get_hierarchy_levels(graph, root_name="root")
    hierarchy_levels = []
    for level in sorted(levels_dict.keys()):
        indices = [node_to_idx[node] for node in levels_dict[level]]
        hierarchy_levels.append(np.array(indices, dtype=np.int64))
    return hierarchy_levels


def _get_leaf_indices(graph: nx.DiGraph, node_to_idx: Dict[str, int]) -> torch.Tensor:
    leaves = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    leaf_indices = sorted([node_to_idx[node] for node in leaves])
    return torch.tensor(leaf_indices, dtype=torch.long)


def _patch_bidirectional_global_edges(model: HGNN):
    """Add taxonomy->global reverse edges."""
    old = model.edge_index_global_context
    num_nodes = model.num_nodes
    dev = old.device
    reverse = torch.stack([
        torch.arange(1, num_nodes + 1, dtype=torch.long, device=dev),
        torch.zeros(num_nodes, dtype=torch.long, device=dev),
    ], dim=0)
    model.edge_index_global_context = torch.cat([old, reverse], dim=1)


# --------------------------------------------------------------------------- #
# Inference API class
# --------------------------------------------------------------------------- #


class HGNNInference:
    """
    Load a trained HGNN checkpoint and run inference on single images.

    Example:
        api = HGNNInference.from_run_dir("runs/c1_hgnn_baseline/20260528_230106")
        result = api.infer(image="path/to/image.jpg", return_probs=True)
        print(result["leaf_label"])   # e.g. "concrete"
        print(result["path_nodes"])   # ["root", "solid", "abiotic", ...]
    """

    def __init__(
        self,
        model: HGNN,
        device: torch.device,
        transform: T.Compose,
        idx_to_node: List[str],
        node_to_idx: Dict[str, int],
        leaf_indices: torch.Tensor,
        hierarchy_levels: List[np.ndarray],
    ):
        self.model = model
        self.device = device
        self.transform = transform
        self.idx_to_node = idx_to_node
        self.node_to_idx = node_to_idx
        self.leaf_indices = leaf_indices.to(device)
        self.hierarchy_levels = hierarchy_levels
        self.num_nodes = len(idx_to_node)

        # Build mapping from leaf index -> leaf name
        self.leaf_names = [self.idx_to_node[i] for i in self.leaf_indices.cpu().tolist()]

    # ----------------------------------------------------------------------- #
    # Factory
    # ----------------------------------------------------------------------- #

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Union[str, Path],
        device: Union[str, torch.device] = "auto",
    ) -> "HGNNInference":
        """
        Build an inference wrapper from a training run directory.

        Args:
            run_dir: path containing config.yaml, checkpoint_best.pt, node_index.json
            device: "auto", "cpu", "cuda", or torch.device
        """
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Load config and metadata
        with open(run_dir / "config.yaml") as f:
            import yaml
            config = yaml.safe_load(f)
        with open(run_dir / "node_index.json") as f:
            node_index = json.load(f)

        node_to_idx = node_index["node_to_idx"]
        idx_to_node = node_index["idx_to_node"]

        device = _resolve_device(device if isinstance(device, str) else str(device))
        print(f"Device: {device}")

        # Rebuild graph
        graph = get_taxonomy(config["data"]["taxonomy_json"])
        graph = _canonicalize_graph(graph, idx_to_node)
        hierarchy_levels = _compute_hierarchy_levels(graph, node_to_idx)
        leaf_indices = _get_leaf_indices(graph, node_to_idx)

        # Build model exactly as during training
        model = HGNN(
            graph=graph,
            path_predict=False,
            dropout_prob=config["model"]["dropout"],
            cnn_kwargs={
                "backbone": config["model"]["cnn_backbone"],
                "pretrained": False,
                "output_dim": config["model"]["cnn_output_dim"],
            },
            gnn_kwargs={
                "input_dim": config["model"]["gnn_input_dim"],
                "hidden_dim": config["model"]["gnn_hidden_dim"],
                "output_dim": config["model"]["gnn_output_dim"],
                "num_layers": config["model"]["gnn_layers"],
                "num_heads": config["model"]["gnn_heads"],
                "skip_connection": config["model"].get("skip_connection", True),
            },
        )

        if config.get("graph", {}).get("bidirectional_global_edges", False):
            _patch_bidirectional_global_edges(model)

        # Load weights
        ckpt_path = run_dir / "checkpoint_best.pt"
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        print(f"Loaded checkpoint from {ckpt_path}")

        # Build transform
        img_size = config["training"]["image_size"]
        mean = config.get("data", {}).get("mean", [0.485, 0.456, 0.406])
        std = config.get("data", {}).get("std", [0.229, 0.224, 0.225])
        transform = T.Compose([
            T.Resize(img_size, antialias=True),
            T.CenterCrop(img_size),
            T.Normalize(mean=mean, std=std),
        ])

        return cls(
            model=model,
            device=device,
            transform=transform,
            idx_to_node=idx_to_node,
            node_to_idx=node_to_idx,
            leaf_indices=leaf_indices,
            hierarchy_levels=hierarchy_levels,
        )

    # ----------------------------------------------------------------------- #
    # Inference helpers
    # ----------------------------------------------------------------------- #

    def _load_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Convert arbitrary input to CHW float32 tensor [0,1]."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            # Assume HWC uint8 or float
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
            tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        elif isinstance(image, Image.Image):
            arr = np.array(image).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return tensor

    @torch.inference_mode()
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run model and ensure 2D logits output."""
        logits = self.model(images)
        # GraphBackbone.squeeze() removes batch dim when B==1
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits

    @torch.inference_mode()
    def predict_path(self, logits: torch.Tensor) -> List[Tuple[int, str]]:
        """
        Greedy path decode from raw logits.
        Returns list of (node_idx, node_name) along the predicted path.
        """
        probs = torch.sigmoid(logits).squeeze()  # (num_nodes,)
        adj = self.model.adjacency_matrix  # (num_nodes, num_nodes)

        path = []
        current = 0  # root is index 0? No — root index depends on node ordering.
        # Map node name to index
        root_idx = self.node_to_idx["root"]
        current = root_idx
        active = True
        path.append((current, self.idx_to_node[current]))

        visited = {current}
        while active:
            neighbor_mask = adj[current] > 0
            neighbor_probs = neighbor_mask * probs
            if neighbor_probs.sum() == 0:
                break
            next_node = neighbor_probs.argmax().item()
            if next_node in visited:
                break
            visited.add(next_node)
            path.append((next_node, self.idx_to_node[next_node]))
            current = next_node

        return path

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #

    def infer(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_probs: bool = True,
        decode_path: bool = True,
    ) -> Dict:
        """
        Run inference on a single image.

        Args:
            image: file path, PIL Image, or HWC numpy array
            return_probs: include leaf and full-node probability vectors
            decode_path: run greedy path decoding

        Returns:
            dict with keys:
                - leaf_label (str): predicted leaf class name
                - leaf_idx (int): leaf node index in full graph
                - path_nodes (List[str]): hierarchical path root->leaf (if decode_path)
                - leaf_probs (np.ndarray): softmax over leaves, shape (37,)
                - node_probs (np.ndarray): sigmoid over all 58 nodes, shape (58,)
                - logits (np.ndarray): raw logits, shape (58,)
        """
        tensor = self._load_image(image)
        tensor = self.transform(tensor).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        logits = self._forward(tensor).squeeze(0)  # (num_nodes,)
        probs = torch.sigmoid(logits)
        leaf_logits = logits[self.leaf_indices]
        leaf_probs = torch.softmax(leaf_logits, dim=0)
        pred_leaf_pos = leaf_probs.argmax().item()
        pred_leaf_idx = self.leaf_indices[pred_leaf_pos].item()
        pred_leaf_name = self.leaf_names[pred_leaf_pos]

        result = {
            "leaf_label": pred_leaf_name,
            "leaf_idx": pred_leaf_idx,
        }

        if return_probs:
            result["leaf_probs"] = leaf_probs.cpu().numpy()
            result["node_probs"] = probs.cpu().numpy()
            result["logits"] = logits.cpu().numpy()

        if decode_path:
            path = self.predict_path(logits)
            result["path_nodes"] = [name for _, name in path]
            result["path_indices"] = [idx for idx, _ in path]

        return result

    def infer_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        return_probs: bool = True,
        decode_path: bool = False,
    ) -> List[Dict]:
        """
        Run inference on a batch of images.

        Returns a list of result dicts, one per image.
        """
        tensors = [self.transform(self._load_image(im)) for im in images]
        batch = torch.stack(tensors).to(self.device)  # [B, 3, H, W]

        logits = self._forward(batch)  # [B, num_nodes]
        probs = torch.sigmoid(logits)
        leaf_logits = logits[:, self.leaf_indices]  # [B, num_leaves]
        leaf_probs = torch.softmax(leaf_logits, dim=1)
        pred_leaf_pos = leaf_probs.argmax(dim=1)  # [B]
        pred_leaf_indices = self.leaf_indices[pred_leaf_pos.cpu()]  # [B]

        results = []
        for i in range(len(images)):
            res = {
                "leaf_label": self.leaf_names[pred_leaf_pos[i].item()],
                "leaf_idx": pred_leaf_indices[i].item(),
            }
            if return_probs:
                res["leaf_probs"] = leaf_probs[i].cpu().numpy()
                res["node_probs"] = probs[i].cpu().numpy()
                res["logits"] = logits[i].cpu().numpy()
            if decode_path:
                path = self.predict_path(logits[i])
                res["path_nodes"] = [name for _, name in path]
                res["path_indices"] = [idx for idx, _ in path]
            results.append(res)

        return results


# --------------------------------------------------------------------------- #
# CLI sanity check
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HGNN inference API sanity check.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    api = HGNNInference.from_run_dir(args.run_dir, device=args.device)
    result = api.infer(args.image)
    print(f"Leaf: {result['leaf_label']}")
    if "path_nodes" in result:
        print(f"Path: {' / '.join(result['path_nodes'])}")
