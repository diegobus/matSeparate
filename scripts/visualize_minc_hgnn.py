#!/usr/bin/env python3
"""
Visualize MINC HGNN validation predictions.

Produces a grid of sample images annotated with:
  - Ground-truth label
  - Predicted leaf label + confidence
  - Whether the prediction is correct

Usage:
    python scripts/visualize_minc_hgnn.py \
        --run-dir runs/minc_hgnn_baseline/20260529_192559 \
        --out viz_minc_hgnn.png
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "gnn_classifier"))

import gnn_classifier.hgnn as _hgnn_mod
from torch_geometric.nn import global_mean_pool as _orig_gmp
_hgnn_mod.global_mean_pool = lambda x, batch, size=None: _orig_gmp(x, batch, size=size)

from gnn_classifier.hgnn import HGNN
from taxonomy.tree import get_taxonomy
from datasets.minc import MINCDataset
import networkx as nx


CATEGORIES = [
    "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass",
    "hair", "leather", "metal", "mirror", "other", "painted", "paper",
    "plastic", "polishedstone", "skin", "sky", "stone", "tile",
    "wallpaper", "water", "wood",
]

# Color per coarse group
GROUP_COLORS = {
    "biological":    "#4CAF50",
    "environmental": "#2196F3",
    "textile":       "#FF9800",
    "construction":  "#9C27B0",
    "industrial":    "#F44336",
    "organic_derived": "#795548",
}

LEAF_TO_GROUP = {
    "foliage": "biological", "food": "biological", "hair": "biological",
    "skin": "biological", "wood": "biological",
    "sky": "environmental", "stone": "environmental",
    "polishedstone": "environmental", "water": "environmental",
    "carpet": "textile", "fabric": "textile", "wallpaper": "textile",
    "brick": "construction", "ceramic": "construction",
    "painted": "construction", "tile": "construction",
    "glass": "industrial", "metal": "industrial", "mirror": "industrial",
    "paper": "industrial", "plastic": "industrial",
    "leather": "organic_derived", "other": "organic_derived",
}


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def canonicalize_graph(graph, node_order):
    import networkx as nx
    g = nx.DiGraph()
    for n in node_order:
        g.add_node(n, **graph.nodes[n])
    for u, v in graph.edges:
        g.add_edge(u, v, **graph.edges[u, v])
    return g


def patch_bidirectional(model):
    old = model.edge_index_global_context
    num_nodes = model.num_nodes
    dev = old.device
    reverse = torch.stack([
        torch.arange(1, num_nodes + 1, dtype=torch.long, device=dev),
        torch.zeros(num_nodes, dtype=torch.long, device=dev),
    ], dim=0)
    model.edge_index_global_context = torch.cat([old, reverse], dim=1)


def load_model(run_dir, device):
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(run_dir / "node_index.json") as f:
        node_index = json.load(f)

    node_to_idx = node_index["node_to_idx"]
    idx_to_node = node_index["idx_to_node"]

    graph = get_taxonomy(config["data"]["taxonomy_json"])
    graph = canonicalize_graph(graph, idx_to_node)

    leaf_nodes = sorted([n for n in graph.nodes() if graph.out_degree(n) == 0])
    leaf_indices = torch.tensor([node_to_idx[n] for n in leaf_nodes], dtype=torch.long)

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
        patch_bidirectional(model)

    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    return model, leaf_nodes, leaf_indices.to(device), config


def make_grid(samples, model, leaf_nodes, leaf_indices, device, transform,
              n_rows=6, n_per_row=4, seed=42):
    random.seed(seed)

    # Pick n_per_row * n_rows samples, stratified by category (a few correct + incorrect)
    by_cat = {}
    for s in samples:
        by_cat.setdefault(s["c1_label"], []).append(s)

    selected = []
    per_cat = max(1, (n_rows * n_per_row) // len(by_cat))
    for cat in sorted(by_cat):
        selected.extend(random.sample(by_cat[cat], min(per_cat, len(by_cat[cat]))))
    random.shuffle(selected)
    selected = selected[: n_rows * n_per_row]

    fig, axes = plt.subplots(n_rows, n_per_row, figsize=(n_per_row * 3.2, n_rows * 3.6))
    axes = axes.flatten()

    for ax, sample in zip(axes, selected):
        # Load and preprocess image
        from PIL import Image as PILImage
        img_path = Path("data/external/minc/minc-2500") / sample["image_path"]
        pil_img = PILImage.open(img_path).convert("RGB")
        tensor = transform(torch.from_numpy(
            np.array(pil_img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            leaf_logits = logits[0, leaf_indices]
            probs = F.softmax(leaf_logits, dim=0)
            pred_pos = probs.argmax().item()
            pred_label = leaf_nodes[pred_pos]
            confidence = probs[pred_pos].item()

        true_label = sample["c1_label"]
        correct = pred_label == true_label

        # Show raw image (not normalized)
        display_img = np.array(pil_img)
        # Center crop to 224x224 for display
        h, w = display_img.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        display_img = display_img[y0:y0+s, x0:x0+s]

        ax.imshow(display_img)
        ax.axis("off")

        border_color = "#2ecc71" if correct else "#e74c3c"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)

        group_color = GROUP_COLORS.get(LEAF_TO_GROUP.get(pred_label, ""), "#888")
        label_text = f"✓ {pred_label}" if correct else f"✗ {pred_label}"
        ax.set_title(
            f"GT: {true_label}\nPred: {label_text} ({confidence:.0%})",
            fontsize=8.5,
            color=group_color if correct else "#e74c3c",
            fontweight="bold" if correct else "normal",
            pad=4,
        )

    # Hide unused axes
    for ax in axes[len(selected):]:
        ax.axis("off")

    # Legend for groups
    legend_patches = [
        mpatches.Patch(color=c, label=g.replace("_", " "))
        for g, c in GROUP_COLORS.items()
    ]
    fig.legend(
        handles=legend_patches, loc="lower center", ncol=3,
        fontsize=8, title="Predicted group", title_fontsize=8,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle("MINC HGNN — Validation Predictions", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/minc_hgnn_baseline/20260529_192559"))
    parser.add_argument("--out", default="viz_minc_hgnn.png")
    parser.add_argument("--n-rows", type=int, default=6)
    parser.add_argument("--n-per-row", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = resolve_device()
    model, leaf_nodes, leaf_indices, config = load_model(args.run_dir, device)

    transform = T.Compose([
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    import csv
    with open("data/processed/minc/splits/val.csv") as f:
        val_samples = list(csv.DictReader(f))

    print(f"Loaded {len(val_samples)} val samples")
    fig = make_grid(val_samples, model, leaf_nodes, leaf_indices, device, transform,
                    n_rows=args.n_rows, n_per_row=args.n_per_row, seed=args.seed)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
