#!/usr/bin/env python3
"""
Visualize HGNN predictions on validation images.

Creates a figure with ~20 images showing the true taxonomy path vs the
predicted path from a trained checkpoint.

Usage:
    python scripts/visualize_predictions.py \
        --run-dir runs/c1_hgnn_baseline/20260528_141412 \
        --split val \
        --num-images 20 \
        --out predictions_grid.png
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# MPS-safe global_mean_pool monkey-patch (must happen before HGNN import)
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

from datasets.matador import MatadorC1Dataset
from scripts.infer_api import HGNNInference


def _load_split_rows(split_csv: Path):
    with open(split_csv, newline="") as f:
        return list(csv.DictReader(f))


def _get_val_indices(base_ds, split_rows):
    """Map val split sample_ids to base_ds indices."""
    sample_id_to_idx = {base_ds.samples[i]["sample_id"]: i for i in range(len(base_ds))}
    indices = []
    for row in split_rows:
        sid = row["sample_id"]
        if sid in sample_id_to_idx:
            indices.append(sample_id_to_idx[sid])
    return indices


def _tensor_to_display(tensor):
    """Convert CHW float [0,1] tensor to HWC uint8 numpy for matplotlib."""
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def _tensor_to_pil(tensor):
    """Convert CHW float [0,1] tensor to PIL RGB Image."""
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _path_to_string(path_nodes):
    """Join path nodes with arrows."""
    return " → ".join(path_nodes)


def _leaf_correct(true_path, pred_path):
    """Check if the leaf (last element) matches."""
    return true_path[-1] == pred_path[-1]


def _format_path_line(label, path_nodes, true_path, is_agreed):
    """Format a path line with correctness and agreement indicators."""
    path_str = " → ".join(path_nodes)
    leaf_ok = _leaf_correct(true_path, path_nodes)
    ok_mark = "✓" if leaf_ok else "✗"
    agree_mark = "" if is_agreed else " ⚠ DIFF"
    return f"{label}: {ok_mark} {path_str}{agree_mark}"


def main():
    parser = argparse.ArgumentParser(description="Visualize HGNN predictions.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("predictions_grid.png"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # ----------------------------------------------------------------------- #
    # Load inference API
    # ----------------------------------------------------------------------- #
    print(f"Loading checkpoint from {args.run_dir} ...")
    api = HGNNInference.from_run_dir(args.run_dir, device=args.device)

    # ----------------------------------------------------------------------- #
    # Load config for data paths
    # ----------------------------------------------------------------------- #
    with open(args.run_dir / "config.yaml") as f:
        import yaml
        config = yaml.safe_load(f)

    extracted_root = config["data"].get("extracted_root")
    appearance_tar = config["data"].get("appearance_tar")
    if extracted_root:
        appearance_tar = None
    else:
        extracted_root = None

    # ----------------------------------------------------------------------- #
    # Build dataset (no transform -> raw [0,1] tensors)
    # ----------------------------------------------------------------------- #
    base_ds = MatadorC1Dataset(
        manifest_csv=config["data"]["manifest_csv"],
        taxonomy_json=config["data"]["taxonomy_json"],
        appearance_tar=appearance_tar or None,
        extracted_root=extracted_root or None,
        node_index_json=config["data"]["node_index"],
        transform=None,  # raw float32 [0,1] CHW
    )

    # ----------------------------------------------------------------------- #
    # Select random val samples
    # ----------------------------------------------------------------------- #
    split_csv = Path(config["data"][f"{args.split}_split"])
    split_rows = _load_split_rows(split_csv)
    val_indices = _get_val_indices(base_ds, split_rows)
    print(f"{args.split} samples available: {len(val_indices)}")

    random.seed(args.seed)
    selected = random.sample(val_indices, min(args.num_images, len(val_indices)))
    print(f"Selected {len(selected)} images for visualization.")

    # ----------------------------------------------------------------------- #
    # Run inference and collect results
    # ----------------------------------------------------------------------- #
    results = []
    for idx in selected:
        sample = base_ds[idx]
        image_tensor = sample["image"]  # CHW float [0,1]
        true_path = sample["taxa"]

        # Inference: pass PIL Image so it works with tar or extracted
        pil_img = _tensor_to_pil(image_tensor)
        result = api.infer(image=pil_img, decode_path=True)
        pred_path = result["path_nodes"]
        pred_path_from_leaf = result["path_nodes_from_leaf"]
        paths_agree = result["paths_agree"]

        results.append({
            "image": image_tensor,
            "true_path": true_path,
            "pred_path": pred_path,
            "pred_path_from_leaf": pred_path_from_leaf,
            "paths_agree": paths_agree,
            "leaf_correct": _leaf_correct(true_path, pred_path),
            "leaf_correct_from_leaf": _leaf_correct(true_path, pred_path_from_leaf),
            "sample_id": sample["sample_id"],
        })

    # ----------------------------------------------------------------------- #
    # Build figure
    # ----------------------------------------------------------------------- #
    n = len(results)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 4.2))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    # Count disagreements once
    diff_count = sum(1 for r in results if not r["paths_agree"])

    for i, res in enumerate(results):
        ax = axes[i]
        img_disp = _tensor_to_display(res["image"])
        ax.imshow(img_disp)
        ax.axis("off")

        true_str = _path_to_string(res["true_path"])
        pred_str = _path_to_string(res["pred_path"])
        pred_leaf_str = _path_to_string(res["pred_path_from_leaf"])

        # Primary color: green if greedy sigmoid path is correct, else red
        color = "#228B22" if res["leaf_correct"] else "#B22222"
        marker = "✓" if res["leaf_correct"] else "✗"

        title = (
            f"{marker} {res['sample_id']}\n"
            f"True:  {true_str}\n"
            f"Greedy: {pred_str}\n"
            f"Leaf:   {pred_leaf_str}"
        )
        if not res["paths_agree"]:
            title += "\n⚠ PATHS DIFFER"
        ax.set_title(title, fontsize=6.5, color=color, loc="left", wrap=True)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"HGNN Predictions — {args.run_dir.name}\n"
        f"Split: {args.split}  |  Green = correct leaf  |  Red = wrong leaf  |  "
        f"{diff_count}/{n} samples have disagreeing paths",
        fontsize=12,
        y=1.00,
    )
    fig.tight_layout()

    out_path = args.out
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")

    # Print summary
    correct_greedy = sum(1 for r in results if r["leaf_correct"])
    correct_leaf = sum(1 for r in results if r["leaf_correct_from_leaf"])
    disagree = sum(1 for r in results if not r["paths_agree"])
    print(f"Greedy leaf accuracy: {correct_greedy}/{n} ({100*correct_greedy/n:.1f}%)")
    print(f"Leaf-anchored accuracy: {correct_leaf}/{n} ({100*correct_leaf/n:.1f}%)")
    print(f"Paths disagree: {disagree}/{n} ({100*disagree/n:.1f}%)")


if __name__ == "__main__":
    main()
