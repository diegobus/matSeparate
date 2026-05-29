#!/usr/bin/env python3
"""
Train HGNN on Matador-C1 using existing gnn_classifier/hgnn.py and loss.py.

Usage:
    python scripts/train_c1_hgnn.py \
        --config configs/experiments/c1_hgnn_baseline.yaml

    python scripts/train_c1_hgnn.py --dry-run
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_str: str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _canonicalize_graph(graph, node_order):
    """Create a new DiGraph with nodes in the specified deterministic order."""
    import networkx as nx
    new_graph = nx.DiGraph()
    for node in node_order:
        new_graph.add_node(node, **graph.nodes[node])
    for u, v in graph.edges:
        new_graph.add_edge(u, v, **graph.edges[u, v])
    return new_graph


def _compute_hierarchy_levels(graph, node_to_idx):
    """Return list of np.ndarray, one per level, containing node indices."""
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
    """Picklable wrapper that keeps target_multihot and leaf_idx."""

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
    with open(split_csv, newline="") as f:
        return list(csv.DictReader(f))


def _init_prototypes(model, idx_to_node, config):
    mode = config["prototypes"]["init"]
    if mode == "model_default":
        print("Prototype init: model default (unchanged)")
        return

    if mode == "random":
        print("Prototype init: random")
        nn.init.normal_(model.prototypes.weight, mean=0, std=config["prototypes"].get("random_std", 0.02))
        return

    # cnn_average
    artifact_path = Path(config["prototypes"]["path"])
    if not artifact_path.exists():
        raise FileNotFoundError(f"Prototype artifact not found: {artifact_path}")

    artifact = torch.load(artifact_path, map_location="cpu")
    artifact_nodes = artifact["idx_to_node"]
    artifact_prototypes = artifact["prototypes"]

    # Validate ordering
    if artifact_nodes != idx_to_node:
        raise ValueError(
            f"Artifact node ordering differs from node_index.json.\n"
            f"Artifact: {artifact_nodes}\n"
            f"Expected: {idx_to_node}"
        )

    # Validate shape
    expected_shape = (model.num_nodes, model.prototypes.weight.shape[1])
    if artifact_prototypes.shape != expected_shape:
        raise ValueError(
            f"Artifact prototypes shape {artifact_prototypes.shape} != {expected_shape}"
        )

    with torch.no_grad():
        model.prototypes.weight.copy_(artifact_prototypes)

    # Projection synchronization: ensure query image and prototypes live in same space
    if config["prototypes"].get("sync_projection", True):
        proj_sd = artifact.get("projection_state_dict")
        if proj_sd is None:
            if not config["prototypes"].get("allow_projection_mismatch", False):
                raise ValueError(
                    "Artifact missing projection_state_dict. "
                    "Set prototypes.allow_projection_mismatch=true to skip, "
                    "or regenerate the artifact with init_hgnn_prototypes.py."
                )
            print("WARNING: projection_state_dict missing; prototypes and image features may be in mismatched spaces.")
        else:
            # Load artifact projection into ImageEncoder.classifier
            # This is the same Linear(2048,1024) that produced the prototypes
            model.cnn.classifier.load_state_dict(proj_sd)
            print("  Sync: loaded artifact projection into ImageEncoder.classifier")

            # Set HGNN.projection to identity so global node is in the same 1024-d space
            gnn_input_dim = model.projection.weight.shape[1]
            with torch.no_grad():
                model.projection.weight.copy_(torch.eye(gnn_input_dim))
                if model.projection.bias is not None:
                    model.projection.bias.zero_()
            print("  Sync: set HGNN.projection to identity")

    print(f"Prototype init: loaded cnn_average from {artifact_path}")


def _get_leaf_indices(graph, node_to_idx, expected_leaves=None) -> torch.Tensor:
    """Return sorted tensor of leaf node indices (out_degree == 0)."""
    leaves = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    if expected_leaves is not None:
        assert len(leaves) == expected_leaves, (
            f"Expected {expected_leaves} leaf nodes, got {len(leaves)}: {leaves}"
        )
    leaf_indices = sorted([node_to_idx[node] for node in leaves])
    return torch.tensor(leaf_indices, dtype=torch.long)


def _get_true_leaf_indices(target_multihot: torch.Tensor, leaf_indices: torch.Tensor) -> torch.Tensor:
    """
    For each sample, find the single active leaf among leaf_indices.
    target_multihot: [B, num_nodes] float
    Returns: [B] long tensor of leaf node indices.
    """
    # Mask to only leaf columns
    leaf_mask = target_multihot[:, leaf_indices]  # [B, 37]
    # Each sample should have exactly one active leaf
    active_counts = leaf_mask.sum(dim=1)
    # Allow tiny floating-point tolerance
    if not torch.all(active_counts > 0.5):
        bad = (active_counts <= 0.5).nonzero(as_tuple=True)[0]
        raise ValueError(f"Samples {bad.tolist()} have no active leaf in target_multihot")
    # Argmax within leaf columns to get position in leaf_indices
    leaf_positions = leaf_mask.argmax(dim=1)  # [B]
    true_leaf_idx = leaf_indices[leaf_positions]  # [B]
    return true_leaf_idx


def leaf_top1_accuracy(logits: torch.Tensor, target_multihot: torch.Tensor, leaf_indices: torch.Tensor) -> float:
    """
    Restrict argmax to leaf nodes only.
    logits: [B, num_nodes]
    target_multihot: [B, num_nodes]
    """
    true_leaf = _get_true_leaf_indices(target_multihot, leaf_indices)
    leaf_logits = logits[:, leaf_indices]  # [B, 37]
    pred_leaf_pos = leaf_logits.argmax(dim=1)  # position in leaf_indices
    pred_leaf = leaf_indices[pred_leaf_pos]
    return (pred_leaf == true_leaf).float().mean().item()


def node_bce_accuracy(logits: torch.Tensor, target_multihot: torch.Tensor) -> float:
    """Thresholded node-level accuracy (sigmoid > 0.5 vs target)."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == target_multihot).float().mean().item()


def exact_path_match(logits: torch.Tensor, target_multihot: torch.Tensor) -> float:
    """Fraction of samples where every node in the thresholded path matches target."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    matches = (preds == target_multihot).all(dim=1).float()
    return matches.mean().item()


def path_f1(logits: torch.Tensor, target_multihot: torch.Tensor) -> float:
    """Macro-averaged F1 score for thresholded multi-label path prediction."""
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
    """Per-sample: for each level, is the argmax node correct? Average over levels and samples."""
    correct = 0.0
    total = 0
    for level in hierarchy_levels:
        level = torch.tensor(level, device=logits.device, dtype=torch.long)
        level_logits = logits[:, level]
        level_targets = target_multihot[:, level]
        # Only evaluate samples that participate in this level
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


def _load_cnn_checkpoint(model, checkpoint_path: Path):
    """Load compatible backbone weights from a flat ResNet checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    source_state = ckpt.get("model_state_dict", ckpt)

    # Flat checkpoint keys are like "layer1.0.conv1.weight" etc.
    # HGNN ImageEncoder stores backbone under cnn.cnn.*
    target_state = model.state_dict()
    matched, missing, unexpected = {}, [], []

    for key, val in source_state.items():
        # Try mapping into cnn.cnn.* first
        target_key = f"cnn.cnn.{key}"
        if target_key in target_state:
            if target_state[target_key].shape == val.shape:
                matched[target_key] = val
            else:
                unexpected.append(f"{target_key} (shape mismatch)")
        else:
            # Also try direct match (e.g. if checkpoint already has cnn.cnn prefix)
            if key in target_state and target_state[key].shape == val.shape:
                matched[key] = val
            else:
                missing.append(key)

    for key in target_state:
        if key not in matched:
            unexpected.append(key)

    model.load_state_dict(matched, strict=False)
    print(f"CNN checkpoint loaded from {checkpoint_path}")
    print(f"  Loaded keys:     {len(matched)}")
    print(f"  Missing keys:    {len(missing)}  (first 5: {missing[:5]})")
    print(f"  Unexpected keys: {len([u for u in unexpected if 'shape mismatch' not in u])}")


def _patch_bidirectional_global_edges(model):
    """Add taxonomy→global reverse edges to edge_index_global_context buffer."""
    old = model.edge_index_global_context
    num_nodes = model.num_nodes
    dev = old.device
    # Build reverse: taxonomy → global on same device
    reverse = torch.stack([
        torch.arange(1, num_nodes + 1, dtype=torch.long, device=dev),
        torch.zeros(num_nodes, dtype=torch.long, device=dev),
    ], dim=0)
    new = torch.cat([old, reverse], dim=1)
    model.edge_index_global_context = new
    print(f"Graph: patched bidirectional global edges ({old.shape[1]} → {new.shape[1]} edges)")


def _ensure_2d_logits(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    """GraphBackbone.squeeze() removes batch dim when batch_size==1."""
    if logits.dim() == 1:
        return logits.unsqueeze(0)
    return logits


# --------------------------------------------------------------------------- #
# Training / validation
# --------------------------------------------------------------------------- #

def train_epoch(model, loader, optimizer, leaf_indices, hierarchy_levels, device):
    model.train()
    total_loss = 0.0
    total_leaf_acc = 0.0
    total_node_acc = 0.0
    total_exact = 0.0
    total_f1 = 0.0
    total_hier_acc = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target_multihot"].to(device)

        optimizer.zero_grad()
        logits = _ensure_2d_logits(model(images), images.size(0))
        loss = greedy_loss(logits, targets, hierarchy_levels)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_leaf_acc += leaf_top1_accuracy(logits, targets, leaf_indices) * bs
        total_node_acc += node_bce_accuracy(logits, targets) * bs
        total_exact += exact_path_match(logits, targets) * bs
        total_f1 += path_f1(logits, targets) * bs
        total_hier_acc += hierarchy_level_accuracy(logits, targets, hierarchy_levels) * bs
        count += bs

    return {
        "loss": total_loss / count,
        "leaf_acc": total_leaf_acc / count,
        "node_acc": total_node_acc / count,
        "exact_match": total_exact / count,
        "path_f1": total_f1 / count,
        "hier_acc": total_hier_acc / count,
    }


@torch.no_grad()
def validate(model, loader, leaf_indices, hierarchy_levels, device):
    model.eval()
    total_loss = 0.0
    total_leaf_acc = 0.0
    total_node_acc = 0.0
    total_exact = 0.0
    total_f1 = 0.0
    total_hier_acc = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target_multihot"].to(device)

        logits = _ensure_2d_logits(model(images), images.size(0))
        loss = greedy_loss(logits, targets, hierarchy_levels)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_leaf_acc += leaf_top1_accuracy(logits, targets, leaf_indices) * bs
        total_node_acc += node_bce_accuracy(logits, targets) * bs
        total_exact += exact_path_match(logits, targets) * bs
        total_f1 += path_f1(logits, targets) * bs
        total_hier_acc += hierarchy_level_accuracy(logits, targets, hierarchy_levels) * bs
        count += bs

    return {
        "loss": total_loss / count,
        "leaf_acc": total_leaf_acc / count,
        "node_acc": total_node_acc / count,
        "exact_match": total_exact / count,
        "path_f1": total_f1 / count,
        "hier_acc": total_hier_acc / count,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train HGNN on Matador-C1.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/c1_hgnn_baseline.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Run one fwd/bwd and one val batch, then exit.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prototype-init", type=str, default=None, choices=["cnn_average", "random", "model_default"])
    parser.add_argument("--prototype-path", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.prototype_init is not None:
        config["prototypes"]["init"] = args.prototype_init
    if args.prototype_path is not None:
        config["prototypes"]["path"] = args.prototype_path

    _set_seed(config["training"]["seed"])

    device_str = args.device or config["training"].get("device", "auto")
    device = _resolve_device(device_str)
    print(f"Device: {device}")

    # ----------------------------------------------------------------------- #
    # Load taxonomy graph and canonicalize node order
    # ----------------------------------------------------------------------- #
    with open(config["data"]["node_index"]) as f:
        node_index = json.load(f)
    node_to_idx = node_index["node_to_idx"]
    idx_to_node = node_index["idx_to_node"]
    num_nodes = len(idx_to_node)
    print(f"Taxonomy nodes: {num_nodes}")

    graph = get_taxonomy(config["data"]["taxonomy_json"])
    graph = _canonicalize_graph(graph, idx_to_node)
    hierarchy_levels = _compute_hierarchy_levels(graph, node_to_idx)
    expected_leaves = config.get("model", {}).get("expected_leaves", None)
    leaf_indices = _get_leaf_indices(graph, node_to_idx, expected_leaves).to(device)
    print(f"Hierarchy levels: {len(hierarchy_levels)}")
    for i, level in enumerate(hierarchy_levels):
        print(f"  Level {i}: {len(level)} nodes")
    print(f"Leaf nodes: {len(leaf_indices)}")

    # ----------------------------------------------------------------------- #
    # Build HGNN
    # ----------------------------------------------------------------------- #
    model = HGNN(
        graph=graph,
        path_predict=False,
        dropout_prob=config["model"]["dropout"],
        cnn_kwargs={
            "backbone": config["model"]["cnn_backbone"],
            "pretrained": config["model"]["cnn_pretrained"],
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
    model = model.to(device)

    # Validate num_nodes
    assert model.num_nodes == num_nodes, f"model.num_nodes ({model.num_nodes}) != {num_nodes}"

    # CNN checkpoint warm-start
    cnn_ckpt = config["model"].get("cnn_checkpoint")
    if cnn_ckpt:
        _load_cnn_checkpoint(model, Path(cnn_ckpt))

    # Bidirectional global edges
    if config.get("graph", {}).get("bidirectional_global_edges", False):
        _patch_bidirectional_global_edges(model)

    # Initialize prototypes
    _init_prototypes(model, idx_to_node, config)

    # ----------------------------------------------------------------------- #
    # Datasets
    # ----------------------------------------------------------------------- #
    dataset_class_name = config["data"].get("dataset_class", "MatadorC1Dataset")
    if dataset_class_name == "MINCDataset":
        from datasets.minc import MINCDataset
        base_ds = MINCDataset(
            manifest_csv=config["data"]["manifest_csv"],
            taxonomy_json=config["data"]["taxonomy_json"],
            images_root=config["data"]["images_root"],
            node_index_json=config["data"]["node_index"],
        )
    else:
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

    train_rows = _load_split_rows(Path(config["data"]["train_split"]))
    val_rows = _load_split_rows(Path(config["data"]["val_split"]))

    img_cfg = config["training"]
    transform = _build_transform(
        img_cfg["image_size"],
        config.get("data", {}).get("mean", [0.485, 0.456, 0.406]),
        config.get("data", {}).get("std", [0.229, 0.224, 0.225]),
    )

    train_ds = _HGNNSubset(base_ds, train_rows, transform)
    val_ds = _HGNNSubset(base_ds, val_rows, transform)

    num_workers = config["training"].get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )

    # ----------------------------------------------------------------------- #
    # Optimizer
    # ----------------------------------------------------------------------- #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # ----------------------------------------------------------------------- #
    # Dry run
    # ----------------------------------------------------------------------- #
    if args.dry_run:
        print("\n--- DRY RUN ---")
        print("Loading two train batches...")
        train_iter = iter(train_loader)
        batch1 = next(train_iter)
        batch2 = next(train_iter)
        print(f"Batch1 image shape: {batch1['image'].shape}")
        print(f"Batch1 target_multihot shape: {batch1['target_multihot'].shape}")
        print(f"Batch1 leaf_idx shape: {batch1['leaf_idx'].shape}")

        print("Forward + backward on batch1...")
        model.train()
        optimizer.zero_grad()
        logits = _ensure_2d_logits(model(batch1["image"].to(device)), batch1["image"].size(0))
        print(f"  Logits shape: {logits.shape}")
        loss = greedy_loss(logits, batch1["target_multihot"].to(device), hierarchy_levels)
        loss.backward()
        optimizer.step()
        print(f"  Loss: {loss.item():.4f}")

        print("Evaluating one val batch...")
        model.eval()
        with torch.no_grad():
            vbatch = next(iter(val_loader))
            vtargets = vbatch["target_multihot"].to(device)
            vlogits = _ensure_2d_logits(model(vbatch["image"].to(device)), vbatch["image"].size(0))
            vloss = greedy_loss(vlogits, vtargets, hierarchy_levels)
            print(f"  Val loss: {vloss.item():.4f}")
            print(f"  leaf_acc={leaf_top1_accuracy(vlogits, vtargets, leaf_indices):.4f} "
                  f"node_acc={node_bce_accuracy(vlogits, vtargets):.4f} "
                  f"exact={exact_path_match(vlogits, vtargets):.4f} "
                  f"f1={path_f1(vlogits, vtargets):.4f} "
                  f"hier_acc={hierarchy_level_accuracy(vlogits, vtargets, hierarchy_levels):.4f}")

        print("\nDry run complete. Exiting.")
        return

    # ----------------------------------------------------------------------- #
    # Full training
    # ----------------------------------------------------------------------- #
    runs_dir = Path(config["logging"]["runs_dir"]) / config["experiment_name"]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(run_dir / "node_index.json", "w") as f:
        json.dump({"node_to_idx": node_to_idx, "idx_to_node": idx_to_node}, f, indent=2)

    best_val_loss = float("inf")
    metrics_log = []
    num_epochs = config["training"]["num_epochs"]
    log_interval = config["logging"]["log_interval"]

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_m = train_epoch(model, train_loader, optimizer, leaf_indices, hierarchy_levels, device)
        val_m = validate(model, val_loader, leaf_indices, hierarchy_levels, device)
        elapsed = time.time() - t0

        metrics_log.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_leaf_acc": train_m["leaf_acc"],
            "train_node_acc": train_m["node_acc"],
            "train_exact_match": train_m["exact_match"],
            "train_path_f1": train_m["path_f1"],
            "train_hier_acc": train_m["hier_acc"],
            "val_loss": val_m["loss"],
            "val_leaf_acc": val_m["leaf_acc"],
            "val_node_acc": val_m["node_acc"],
            "val_exact_match": val_m["exact_match"],
            "val_path_f1": val_m["path_f1"],
            "val_hier_acc": val_m["hier_acc"],
            "time_sec": elapsed,
        })

        if epoch % log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{num_epochs}  "
                  f"train_loss={train_m['loss']:.4f} "
                  f"leaf_acc={train_m['leaf_acc']:.4f} "
                  f"node_acc={train_m['node_acc']:.4f} "
                  f"exact={train_m['exact_match']:.4f} "
                  f"f1={train_m['path_f1']:.4f} "
                  f"hier={train_m['hier_acc']:.4f}  |  "
                  f"val_loss={val_m['loss']:.4f} "
                  f"leaf_acc={val_m['leaf_acc']:.4f} "
                  f"node_acc={val_m['node_acc']:.4f} "
                  f"exact={val_m['exact_match']:.4f} "
                  f"f1={val_m['path_f1']:.4f} "
                  f"hier={val_m['hier_acc']:.4f}  "
                  f"({elapsed:.1f}s)")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_m["loss"],
                "val_acc": val_m["leaf_acc"],
                "node_to_idx": node_to_idx,
                "idx_to_node": idx_to_node,
                "hierarchy_levels": [level.tolist() for level in hierarchy_levels],
            }, run_dir / "checkpoint_best.pt")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
