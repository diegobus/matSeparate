#!/usr/bin/env python3
"""
Evaluate the trained HGNN (Matador C1) on MINC-2500.

Because the Matador taxonomy doesn't cover all 23 MINC categories, we:
  1. Map each MINC category to the closest Matador leaf node(s).
  2. Evaluate only on MINC categories that have a plausible mapping.
  3. Report per-category and overall accuracy on that subset.

Prediction: argmax over the 37 leaf nodes, then check if it falls in
the mapped leaf set for that MINC category.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "gnn_classifier"))

import yaml
import networkx as nx
from gnn_classifier.hgnn import HGNN
import gnn_classifier.hgnn as _hgnn_mod
from torch_geometric.nn import global_mean_pool as _orig_gmp

def _safe_gmp(x, batch, size=None):
    return _orig_gmp(x, batch, size=size)
_hgnn_mod.global_mean_pool = _safe_gmp

from taxonomy.tree import get_taxonomy


# ---------------------------------------------------------------------------
# MINC → Matador leaf mapping
# ---------------------------------------------------------------------------

# Maps each MINC category to the set of Matador leaf nodes that best
# represent it visually. Categories with no plausible mapping are omitted.
MINC_TO_MATADOR_LEAVES = {
    "brick":        {"brick"},
    "carpet":       {"carpet"},
    "ceramic":      {"pottery"},          # MINC ceramic = dishes/tiles → pottery
    "fabric":       {"natural_fiber", "wool", "satin", "nylon", "suede"},
    "foliage":      {"foliage", "grass", "ivy", "moss"},
    "food":         {"bread", "fruit", "vegetable"},
    "leather":      {"leather"},
    "metal":        {"generic_metal"},
    "paper":        {"paper"},
    "plastic":      {"foam", "wax"},
    "polishedstone":{"marble", "granite", "limestone"},
    "stone":        {"shale", "granite", "limestone", "marble", "gravel", "sand"},
    "tile":         {"pottery"},          # ceramic tile → pottery
    "wallpaper":    {"paper"},            # paper-based; weak mapping
    "wood":         {"timber", "tree_bark"},
}

SKIPPED_MINC = ["glass", "hair", "mirror", "other", "painted", "skin", "sky", "water"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MINCSubset(Dataset):
    """MINC-2500 images for categories that have a Matador mapping."""

    def __init__(self, root: str, split_file: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        all_cats = [l.strip() for l in open(self.root / "categories.txt")]
        self.cat2idx = {c: i for i, c in enumerate(all_cats)}

        self.samples = []
        with open(split_file) as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                cat = path.split("/")[1]
                if cat in MINC_TO_MATADOR_LEAVES:
                    self.samples.append((path, cat))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, cat = self.samples[idx]
        img = Image.open(self.root / rel_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def canonicalize_graph(graph, node_order):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/c1_hgnn_baseline/avg_init_20260529_004313"))
    parser.add_argument("--minc-root", type=str,
                        default="data/external/minc/minc-2500")
    parser.add_argument("--split", type=str, default="test1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    # Load config & node index from run dir
    with open(args.run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(args.run_dir / "node_index.json") as f:
        node_index = json.load(f)

    node_to_idx = node_index["node_to_idx"]
    idx_to_node = node_index["idx_to_node"]

    graph = get_taxonomy(config["data"]["taxonomy_json"])
    graph = canonicalize_graph(graph, idx_to_node)

    # Leaf node indices
    leaf_nodes = sorted([n for n in graph.nodes() if graph.out_degree(n) == 0])
    leaf_indices = torch.tensor([node_to_idx[n] for n in leaf_nodes], dtype=torch.long)
    leaf_name_to_pos = {n: i for i, n in enumerate(leaf_nodes)}

    # Build per-MINC-category target leaf positions (indices into leaf_nodes list)
    minc_target_positions = {}
    for minc_cat, matador_leaves in MINC_TO_MATADOR_LEAVES.items():
        positions = [leaf_name_to_pos[l] for l in matador_leaves if l in leaf_name_to_pos]
        if positions:
            minc_target_positions[minc_cat] = set(positions)

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
        patch_bidirectional(model)

    ckpt = torch.load(args.run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Dataset
    transform = T.Compose([
        T.Resize(224, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    split_file = Path(args.minc_root) / "labels" / f"{args.split}.txt"
    ds = MINCSubset(args.minc_root, split_file, transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True)

    print(f"Evaluating on {len(ds)} MINC samples "
          f"({len(minc_target_positions)} categories with mappings)\n")
    print("Skipped MINC categories (no Matador equivalent):", SKIPPED_MINC, "\n")

    # Evaluate
    per_cat_correct = {c: 0 for c in minc_target_positions}
    per_cat_total   = {c: 0 for c in minc_target_positions}
    leaf_indices = leaf_indices.to(device)

    with torch.no_grad():
        for imgs, cats in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            # Argmax over leaf nodes only
            leaf_logits = logits[:, leaf_indices]
            pred_leaf_pos = leaf_logits.argmax(dim=1).cpu().tolist()

            for pred_pos, cat in zip(pred_leaf_pos, cats):
                per_cat_total[cat] += 1
                if pred_pos in minc_target_positions[cat]:
                    per_cat_correct[cat] += 1

    total_correct = sum(per_cat_correct.values())
    total_samples = sum(per_cat_total.values())
    overall_acc = total_correct / total_samples

    print(f"Overall accuracy (on {len(minc_target_positions)} mapped categories): "
          f"{overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"Total samples: {total_samples}\n")
    print("Per-category accuracy:")
    print(f"  {'category':15s}  {'acc':>6}  {'correct':>8}  mapping")
    print(f"  {'-'*60}")
    for cat in sorted(minc_target_positions):
        c = per_cat_correct[cat]
        t = per_cat_total[cat]
        acc = c / t if t else 0
        leaves_str = ", ".join(sorted(MINC_TO_MATADOR_LEAVES[cat]))
        print(f"  {cat:15s}  {acc:6.4f}  {c:3d}/{t:<3d}   → {leaves_str}")


if __name__ == "__main__":
    main()
