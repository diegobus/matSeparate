#!/usr/bin/env python3
"""
Evaluate a trained HGNN checkpoint on a Matador-C1 split.

Usage:
    python scripts/eval_c1_hgnn.py \
        --run-dir runs/c1_hgnn_baseline/20260528_123000 \
        --split test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "gnn_classifier"))

from datasets.matador import MatadorC1Dataset
from gnn_classifier.hgnn import HGNN
from gnn_classifier.loss import greedy_loss

# Monkey-patch global_mean_pool for MPS compatibility (patch module-level binding)
import gnn_classifier.hgnn as _hgnn_mod
_orig_global_mean_pool = _hgnn_mod.global_mean_pool

def _mps_safe_global_mean_pool(x, batch, size=None):
    orig_device = x.device
    if orig_device.type == "mps":
        x = x.cpu()
        batch = batch.cpu()
    result = _orig_global_mean_pool(x, batch, size=size)
    if orig_device.type == "mps":
        result = result.to(orig_device)
    return result

_hgnn_mod.global_mean_pool = _mps_safe_global_mean_pool
from taxonomy.tree import get_hierarchy_levels, get_taxonomy


def _resolve_device(device_str: str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _canonicalize_graph(graph, node_order):
    import networkx as nx
    new_graph = nx.DiGraph()
    for node in node_order:
        new_graph.add_node(node, **graph.nodes[node])
    for u, v in graph.edges:
        new_graph.add_edge(u, v, **graph.edges[u, v])
    return new_graph


def _compute_hierarchy_levels(graph, node_to_idx):
    levels_dict = get_hierarchy_levels(graph, root_name="root")
    hierarchy_levels = []
    for level in sorted(levels_dict.keys()):
        indices = [node_to_idx[node] for node in levels_dict[level]]
        hierarchy_levels.append(np.array(indices, dtype=np.int64))
    return hierarchy_levels


def _build_transform(image_size, mean, std):
    return T.Compose([
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
        T.Normalize(mean=mean, std=std),
    ])


def _collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    target_multihot = torch.stack([b["target_multihot"] for b in batch])
    leaf_idx = torch.tensor([b["leaf_idx"] for b in batch], dtype=torch.long)
    return {
        "image": images,
        "target_multihot": target_multihot,
        "leaf_idx": leaf_idx,
    }


class _HGNNSubset:
    def __init__(self, base_ds, split_rows, transform):
        self.base_ds = base_ds
        self.rows = split_rows
        self.transform = transform
        self.sample_id_to_idx = {
            base_ds.samples[i]["sample_id"]: i for i in range(len(base_ds))
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        base_idx = self.sample_id_to_idx[row["sample_id"]]
        sample = self.base_ds[base_idx]
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return {
            "image": sample["image"],
            "target_multihot": sample["target_multihot"],
            "leaf_idx": sample["target_indices"][-1].item(),
        }


def _load_split_rows(split_csv: Path):
    import csv
    with open(split_csv, newline="") as f:
        return list(csv.DictReader(f))


def _get_leaf_indices(graph, node_to_idx) -> torch.Tensor:
    """Return sorted tensor of leaf node indices (out_degree == 0)."""
    leaves = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    assert len(leaves) == 37, f"Expected 37 leaf nodes, got {len(leaves)}"
    leaf_indices = sorted([node_to_idx[node] for node in leaves])
    return torch.tensor(leaf_indices, dtype=torch.long)


def _get_true_leaf_indices(target_multihot: torch.Tensor, leaf_indices: torch.Tensor) -> torch.Tensor:
    """For each sample, find the single active leaf among leaf_indices."""
    leaf_mask = target_multihot[:, leaf_indices]
    active_counts = leaf_mask.sum(dim=1)
    if not torch.all(active_counts > 0.5):
        bad = (active_counts <= 0.5).nonzero(as_tuple=True)[0]
        raise ValueError(f"Samples {bad.tolist()} have no active leaf in target_multihot")
    leaf_positions = leaf_mask.argmax(dim=1)
    return leaf_indices[leaf_positions]


def leaf_top1_accuracy(logits: torch.Tensor, target_multihot: torch.Tensor, leaf_indices: torch.Tensor) -> float:
    true_leaf = _get_true_leaf_indices(target_multihot, leaf_indices)
    leaf_logits = logits[:, leaf_indices]
    pred_leaf_pos = leaf_logits.argmax(dim=1)
    pred_leaf = leaf_indices[pred_leaf_pos]
    return (pred_leaf == true_leaf).float().mean().item()


def node_bce_accuracy(logits: torch.Tensor, target_multihot: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == target_multihot).float().mean().item()


def exact_path_match(logits: torch.Tensor, target_multihot: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    matches = (preds == target_multihot).all(dim=1).float()
    return matches.mean().item()


def path_f1(logits: torch.Tensor, target_multihot: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    tp = (preds * target_multihot).sum(dim=1)
    fp = (preds * (1 - target_multihot)).sum(dim=1)
    fn = ((1 - preds) * target_multihot).sum(dim=1)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean().item()


def hierarchy_level_accuracy(logits: torch.Tensor, target_multihot: torch.Tensor, hierarchy_levels: list) -> float:
    correct = 0.0
    total = 0
    for level in hierarchy_levels:
        level = torch.tensor(level, device=logits.device, dtype=torch.long)
        level_logits = logits[:, level]
        level_targets = target_multihot[:, level]
        participation = level_targets.sum(dim=1) > 0
        if participation.sum() == 0:
            continue
        pred_pos = level_logits[participation].argmax(dim=1)
        pred_nodes = level[pred_pos]
        true_pos = level_targets[participation].argmax(dim=1)
        true_nodes = level[true_pos]
        correct += (pred_nodes == true_nodes).float().sum().item()
        total += participation.sum().item()
    return correct / total if total > 0 else 0.0


def _patch_bidirectional_global_edges(model):
    """Add taxonomy→global reverse edges to edge_index_global_context buffer."""
    old = model.edge_index_global_context
    num_nodes = model.num_nodes
    dev = old.device
    reverse = torch.stack([
        torch.arange(1, num_nodes + 1, dtype=torch.long, device=dev),
        torch.zeros(num_nodes, dtype=torch.long, device=dev),
    ], dim=0)
    new = torch.cat([old, reverse], dim=1)
    model.edge_index_global_context = new
    print(f"Graph: patched bidirectional global edges ({old.shape[1]} → {new.shape[1]} edges)")


def _ensure_2d_logits(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    if logits.dim() == 1:
        return logits.unsqueeze(0)
    return logits


def main():
    parser = argparse.ArgumentParser(description="Evaluate HGNN C1 checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    with open(args.run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(args.run_dir / "node_index.json") as f:
        node_index = json.load(f)

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers

    device_str = args.device
    device = _resolve_device(device_str)
    print(f"Device: {device}")

    node_to_idx = node_index["node_to_idx"]
    idx_to_node = node_index["idx_to_node"]

    # Rebuild canonicalized graph and hierarchy levels
    graph = get_taxonomy(config["data"]["taxonomy_json"])
    graph = _canonicalize_graph(graph, idx_to_node)
    hierarchy_levels = _compute_hierarchy_levels(graph, node_to_idx)
    leaf_indices = _get_leaf_indices(graph, node_to_idx).to(device)

    loss_mode = config["training"].get("loss_mode", "combined")
    parent_child_pairs = torch.tensor(
        [[node_to_idx[u], node_to_idx[v]] for u, v in graph.edges()],
        dtype=torch.long,
    )

    # Build model
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
    checkpoint = torch.load(args.run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Dataset
    extracted_root = config["data"].get("extracted_root")
    appearance_tar = config["data"].get("appearance_tar")
    if extracted_root:
        appearance_tar = None
    else:
        extracted_root = None

    base_ds = MatadorC1Dataset(
        manifest_csv=config["data"]["manifest_csv"],
        taxonomy_json=config["data"]["taxonomy_json"],
        appearance_tar=appearance_tar or None,
        extracted_root=extracted_root or None,
        node_index_json=config["data"]["node_index"],
        transform=None,
    )

    split_csv = Path(config["data"][f"{args.split}_split"])
    rows = _load_split_rows(split_csv)
    transform = _build_transform(
        config["training"]["image_size"],
        config.get("data", {}).get("mean", [0.485, 0.456, 0.406]),
        config.get("data", {}).get("std", [0.229, 0.224, 0.225]),
    )
    ds = _HGNNSubset(base_ds, rows, transform)
    num_workers = config["training"].get("num_workers", 0)
    loader = DataLoader(
        ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )

    total_loss = 0.0
    total_leaf_acc = 0.0
    total_node_acc = 0.0
    total_exact = 0.0
    total_f1 = 0.0
    total_hier_acc = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target_multihot"].to(device)

            logits = _ensure_2d_logits(model(images), images.size(0))
            loss = greedy_loss(logits, targets, hierarchy_levels, mode=loss_mode, parent_child_pairs=parent_child_pairs)

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_leaf_acc += leaf_top1_accuracy(logits, targets, leaf_indices) * bs
            total_node_acc += node_bce_accuracy(logits, targets) * bs
            total_exact += exact_path_match(logits, targets) * bs
            total_f1 += path_f1(logits, targets) * bs
            total_hier_acc += hierarchy_level_accuracy(logits, targets, hierarchy_levels) * bs
            count += bs

    print(f"{args.split.upper()} results:")
    print(f"  Loss:       {total_loss / count:.4f}")
    print(f"  Leaf acc:   {total_leaf_acc / count:.4f}")
    print(f"  Node acc:   {total_node_acc / count:.4f}")
    print(f"  Exact match:{total_exact / count:.4f}")
    print(f"  Path F1:    {total_f1 / count:.4f}")
    print(f"  Hier acc:   {total_hier_acc / count:.4f}")


if __name__ == "__main__":
    main()
