#!/usr/bin/env python3
"""
Initialize HGNN taxonomy node prototypes in two modes:
  random      – normal distribution
  cnn_average – average ResNet features over training images per node

Usage:
    python scripts/init_hgnn_prototypes.py \
        --config configs/experiments/hgnn_prototype_init.yaml

    python scripts/init_hgnn_prototypes.py --config ... --mode random
    python scripts/init_hgnn_prototypes.py --config ... --mode cnn_average --max-samples 128
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader, Dataset

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.matador import MatadorC1Dataset


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device_str: str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _load_node_index(path: Path):
    with open(path) as f:
        data = json.load(f)
    return data["node_to_idx"], data["idx_to_node"]


class _TrainSubset(Dataset):
    """Wrap MatadorC1Dataset to serve only train-split rows."""

    def __init__(self, base_ds: MatadorC1Dataset, train_rows, transform):
        self.base_ds = base_ds
        self.rows = train_rows
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
        }


def _collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for key in keys:
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            shapes = {tuple(v.shape) for v in values}
            if len(shapes) == 1:
                out[key] = torch.stack(values)
            else:
                out[key] = values
        else:
            out[key] = values
    return out


# --------------------------------------------------------------------------- #
# Random mode
# --------------------------------------------------------------------------- #

def _init_random(node_to_idx, idx_to_node, output_dim, seed, std):
    _set_seed(seed)
    num_nodes = len(idx_to_node)
    prototypes = torch.randn(num_nodes, output_dim) * std
    counts = torch.zeros(num_nodes)
    zero_count_nodes = sorted(idx_to_node)
    return prototypes, counts, zero_count_nodes


# --------------------------------------------------------------------------- #
# CNN-average mode
# --------------------------------------------------------------------------- #

def _init_cnn_average(
    node_to_idx,
    idx_to_node,
    train_rows,
    config,
    output_dim,
    device,
    max_samples=None,
):
    encoder_cfg = config["encoder"]
    backbone_name = encoder_cfg["backbone"]
    checkpoint = encoder_cfg.get("checkpoint")

    # Build transform
    img_size = encoder_cfg["image_size"]
    mean = encoder_cfg["mean"]
    std = encoder_cfg["std"]
    transform = T.Compose([
        T.Resize(img_size, antialias=True),
        T.CenterCrop(img_size),
        T.Normalize(mean=mean, std=std),
    ])

    # Build dataset
    extracted_root = config["data"].get("extracted_root")
    appearance_tar = config["data"].get("appearance_tar")
    if extracted_root:
        appearance_tar = None
    else:
        extracted_root = None

    base_ds = MatadorC1Dataset(
        manifest_csv=config["data"]["manifest_csv"]
        if "manifest_csv" in config["data"]
        else config["data"]["train_split"].replace("splits/train.csv", "manifest.csv"),
        taxonomy_json=config["data"]["taxonomy_json"],
        appearance_tar=appearance_tar or None,
        extracted_root=extracted_root or None,
        node_index_json=config["data"]["node_index"],
        transform=None,
    )

    # Validate node ordering
    if base_ds.nodes != idx_to_node:
        raise ValueError(
            f"Dataset node ordering differs from node_index.json.\n"
            f"Dataset: {base_ds.nodes}\nnode_index: {idx_to_node}"
        )

    if max_samples is not None:
        train_rows = train_rows[:max_samples]

    ds = _TrainSubset(base_ds, train_rows, transform)
    loader = DataLoader(
        ds,
        batch_size=encoder_cfg["batch_size"],
        shuffle=False,
        num_workers=encoder_cfg.get("num_workers", 0),
        collate_fn=_collate_fn,
    )

    # Feature extractor (backbone only, no classification head)
    import timm
    feature_extractor = timm.create_model(
        backbone_name,
        pretrained=encoder_cfg["pretrained"],
        num_classes=0,
        global_pool="avg",
    )
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # Optionally load checkpoint backbone weights
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        feature_extractor.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint backbone from {checkpoint}")

    # Determine raw feature dimension
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        raw_feat = feature_extractor(dummy)
    feature_dim_raw = raw_feat.shape[-1]
    print(f"Raw feature dim: {feature_dim_raw}")

    # Projection if needed
    projection = None
    if feature_dim_raw != output_dim:
        projection = nn.Linear(feature_dim_raw, output_dim, bias=True)
        projection = projection.to(device)
        projection.eval()
        print(f"Added projection: {feature_dim_raw} -> {output_dim}")

    # Extract features for all training images
    all_features = []
    all_multihots = []
    num_nodes = len(idx_to_node)

    print("Extracting features...")
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            multihot = batch["target_multihot"]  # [B, num_nodes]

            feats = feature_extractor(images)  # [B, feature_dim_raw]
            if projection is not None:
                feats = projection(feats)

            all_features.append(feats.cpu())
            all_multihots.append(multihot)

    features = torch.cat(all_features, dim=0)      # [N, output_dim]
    multihots = torch.cat(all_multihots, dim=0)    # [N, num_nodes]
    print(f"Extracted {features.shape[0]} feature vectors")

    # Average per node
    prototypes = torch.zeros(num_nodes, output_dim)
    counts = torch.zeros(num_nodes)
    zero_count_nodes = []

    for node_idx in range(num_nodes):
        mask = multihots[:, node_idx] > 0
        node_feats = features[mask]
        count = node_feats.shape[0]
        counts[node_idx] = count
        if count > 0:
            prototypes[node_idx] = node_feats.mean(dim=0)
        else:
            # Random fallback for nodes with no assigned samples
            _set_seed(config["prototypes"]["random_seed"] + node_idx)
            prototypes[node_idx] = torch.randn(output_dim) * config["prototypes"]["random_std"]
            zero_count_nodes.append(idx_to_node[node_idx])

    return prototypes, counts, zero_count_nodes, feature_dim_raw, projection


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _validate(prototypes, idx_to_node, node_counts, output_dim, zero_count_nodes, mode):
    num_nodes = len(idx_to_node)
    assert prototypes.shape == (num_nodes, output_dim), (
        f"prototypes.shape {prototypes.shape} != ({num_nodes}, {output_dim})"
    )
    assert len(idx_to_node) == num_nodes
    if mode == "cnn_average":
        assert node_counts.shape == (num_nodes,), (
            f"node_counts.shape {node_counts.shape} != ({num_nodes},)"
        )
        # Check leaf nodes have nonzero counts
        # Leaves are the 37 C1 labels (last 37 nodes? no, they're scattered)
        # Actually we can't easily check without taxonomy structure
        # But we can print stats

    print(f"\nPrototype tensor stats:")
    print(f"  shape: {tuple(prototypes.shape)}")
    print(f"  mean:  {prototypes.mean():.6f}")
    print(f"  std:   {prototypes.std():.6f}")
    print(f"  min:   {prototypes.min():.6f}")
    print(f"  max:   {prototypes.max():.6f}")

    if mode == "cnn_average":
        sorted_counts = sorted(enumerate(node_counts.tolist()), key=lambda x: x[1])
        print(f"\n10 smallest node counts:")
        for idx, cnt in sorted_counts[:10]:
            print(f"  {idx_to_node[idx]:20s} {cnt:6.0f}")
        print(f"\n10 largest node counts:")
        for idx, cnt in sorted_counts[-10:]:
            print(f"  {idx_to_node[idx]:20s} {cnt:6.0f}")

        if zero_count_nodes:
            print(f"\nWARNING: {len(zero_count_nodes)} nodes with zero count (random fallback):")
            for name in zero_count_nodes:
                print(f"  {name}")
        else:
            print("\nAll nodes have nonzero counts.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Initialize HGNN taxonomy node prototypes.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/hgnn_prototype_init.yaml"))
    parser.add_argument("--mode", choices=["random", "cnn_average"], default=None, help="Override prototype mode.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/mps/auto).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--output-dim", type=int, default=None, help="Override prototype output dim.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit train samples (debug only).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    mode = args.mode or config["prototypes"]["mode"]
    checkpoint = args.checkpoint or config["encoder"].get("checkpoint")
    device_str = args.device or config["encoder"].get("device", "auto")
    device = _resolve_device(device_str)
    print(f"Device: {device}")

    if args.batch_size is not None:
        config["encoder"]["batch_size"] = args.batch_size
    if args.output_dim is not None:
        config["prototypes"]["output_dim"] = args.output_dim

    output_dim = config["prototypes"]["output_dim"]

    # Load node index
    node_index_path = Path(config["data"]["node_index"])
    node_to_idx, idx_to_node = _load_node_index(node_index_path)
    num_nodes = len(idx_to_node)
    print(f"Nodes: {num_nodes}")
    print(f"Output dim: {output_dim}")
    print(f"Mode: {mode}")

    # Load train rows for cnn_average
    train_rows = []
    if mode == "cnn_average":
        train_split = Path(config["data"]["train_split"])
        with open(train_split, newline="") as f:
            train_rows = list(csv.DictReader(f))
        print(f"Train samples: {len(train_rows)}")

    # Initialize prototypes
    if mode == "random":
        prototypes, counts, zero_count_nodes = _init_random(
            node_to_idx,
            idx_to_node,
            output_dim,
            config["prototypes"]["random_seed"],
            config["prototypes"]["random_std"],
        )
        feature_dim_raw = None
        projection = None
    else:
        prototypes, counts, zero_count_nodes, feature_dim_raw, projection = _init_cnn_average(
            node_to_idx,
            idx_to_node,
            train_rows,
            config,
            output_dim,
            device,
            max_samples=args.max_samples,
        )

    # Validation
    _validate(prototypes, idx_to_node, counts, output_dim, zero_count_nodes, mode)

    # Save artifact
    out_dir = Path(config["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config["output"]["filename"]

    artifact = {
        "prototypes": prototypes,
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "output_dim": output_dim,
        "mode": mode,
        "backbone": config["encoder"]["backbone"],
        "checkpoint": checkpoint,
        "feature_dim_raw": feature_dim_raw,
        "projection_state_dict": projection.state_dict() if projection is not None else None,
        "node_counts": counts,
        "zero_count_nodes": zero_count_nodes,
        "config": config,
    }
    torch.save(artifact, out_path)
    print(f"\nArtifact saved to {out_path}")


if __name__ == "__main__":
    main()
