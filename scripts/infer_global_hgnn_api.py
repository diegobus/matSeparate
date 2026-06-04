#!/usr/bin/env python3
"""Inference API for the global-context HGNN C1 classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Union

import networkx as nx
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "gnn_classifier"))

from gnn_classifier.hgnn import GlobalContextHGNN  # noqa: E402
from taxonomy.tree import get_hierarchy_levels, get_taxonomy  # noqa: E402


def _resolve_device(device_str: str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
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


def _get_leaf_indices(graph: nx.DiGraph, node_to_idx: dict[str, int]) -> torch.Tensor:
    leaves = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    return torch.tensor(sorted(node_to_idx[node] for node in leaves), dtype=torch.long)


def _patch_bidirectional_global_edges(model):
    old = model.edge_index_global_context
    num_nodes = model.num_nodes
    dev = old.device
    reverse = torch.stack(
        [
            torch.arange(1, num_nodes + 1, dtype=torch.long, device=dev),
            torch.zeros(num_nodes, dtype=torch.long, device=dev),
        ],
        dim=0,
    )
    model.edge_index_global_context = torch.cat([old, reverse], dim=1)


class GlobalHGNNInference:
    def __init__(
        self,
        model,
        device,
        local_transform,
        context_transform,
        idx_to_node,
        node_to_idx,
        leaf_indices,
    ):
        self.model = model
        self.device = device
        self.local_transform = local_transform
        self.context_transform = context_transform
        self.idx_to_node = list(idx_to_node)
        self.node_to_idx = dict(node_to_idx)
        self.leaf_indices = leaf_indices.to(device)
        self.leaf_names = [self.idx_to_node[i] for i in self.leaf_indices.cpu().tolist()]

    @classmethod
    def from_run_dir(cls, run_dir: Union[str, Path], device: str = "auto"):
        import yaml

        run_dir = Path(run_dir)
        with open(run_dir / "config.yaml") as handle:
            config = yaml.safe_load(handle)
        with open(run_dir / "node_index.json") as handle:
            node_index = json.load(handle)

        node_to_idx = node_index["node_to_idx"]
        idx_to_node = node_index["idx_to_node"]
        graph = get_taxonomy(config["data"]["taxonomy_json"])
        graph = _canonicalize_graph(graph, idx_to_node)
        leaf_indices = _get_leaf_indices(graph, node_to_idx)

        device_obj = _resolve_device(device)
        model = GlobalContextHGNN(
            graph=graph,
            path_predict=False,
            dropout_prob=config["model"]["dropout"],
            head_type=config["model"].get("head_type", "fixed_global_pool"),
            cnn_kwargs={
                "backbone": config["model"]["cnn_backbone"],
                "pretrained": False,
                "output_dim": config["model"]["cnn_output_dim"],
            },
            context_cnn_kwargs={
                "backbone": config["model"].get("context_cnn_backbone", config["model"]["cnn_backbone"]),
                "pretrained": False,
                "output_dim": config["model"].get("context_cnn_output_dim", config["model"]["cnn_output_dim"]),
            },
            gnn_kwargs={
                "input_dim": config["model"]["gnn_input_dim"],
                "hidden_dim": config["model"]["gnn_hidden_dim"],
                "output_dim": config["model"]["gnn_output_dim"],
                "num_layers": config["model"]["gnn_layers"],
                "num_heads": config["model"]["gnn_heads"],
                "skip_connection": config["model"].get("skip_connection", True),
                "dropout": config["model"].get("dropout", 0.0),
            },
        )
        if config.get("graph", {}).get("bidirectional_global_edges", False):
            _patch_bidirectional_global_edges(model)

        checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location=device_obj)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device_obj).eval()
        print(
            f"Loaded global HGNN checkpoint epoch={checkpoint.get('epoch')} "
            f"val_acc={checkpoint.get('val_acc')}"
        )

        mean = config.get("data", {}).get("mean", [0.485, 0.456, 0.406])
        std = config.get("data", {}).get("std", [0.229, 0.224, 0.225])
        local_size = config["training"].get("local_image_size", config["training"].get("image_size", 224))
        context_size = config["training"].get("context_image_size", local_size)
        local_transform = cls._build_transform(local_size, mean, std)
        context_transform = cls._build_transform(context_size, mean, std)
        return cls(
            model,
            device_obj,
            local_transform,
            context_transform,
            idx_to_node,
            node_to_idx,
            leaf_indices,
        )

    @staticmethod
    def _build_transform(image_size: int, mean, std):
        return T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.Normalize(mean=mean, std=std),
        ])

    @staticmethod
    def _to_tensor(image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr /= 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return torch.from_numpy(arr).permute(2, 0, 1)

    @torch.inference_mode()
    def infer_batch(
        self,
        local_images: List[Union[str, Path, Image.Image, np.ndarray]],
        context_images: List[Union[str, Path, Image.Image, np.ndarray]],
    ):
        if len(local_images) != len(context_images):
            raise ValueError("local_images and context_images must have the same length")
        local = torch.stack([
            self.local_transform(self._to_tensor(image)) for image in local_images
        ]).to(self.device)
        context = torch.stack([
            self.context_transform(self._to_tensor(image)) for image in context_images
        ]).to(self.device)
        logits = self.model(local, context)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        leaf_logits = logits[:, self.leaf_indices]
        leaf_probs = torch.softmax(leaf_logits, dim=1)
        pred = leaf_probs.argmax(dim=1)
        results = []
        for i in range(len(local_images)):
            results.append({
                "leaf_label": self.leaf_names[int(pred[i].item())],
                "leaf_probs": leaf_probs[i].detach().cpu().numpy(),
                "logits": logits[i].detach().cpu().numpy(),
            })
        return results
