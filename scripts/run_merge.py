#!/usr/bin/env python3
"""
Hierarchy-guided SAM mask merging.

Hypothesis: SAM sometimes splits a single material region into multiple adjacent
proposals. If two neighbouring masks share the same taxonomy parent-node prediction,
they likely belong to the same material and should be merged before final labelling.

Method:
  1. Classify all SAM masks with HGNN → get parent-node prediction for each
  2. Find adjacent mask pairs (bounding-box proximity within --proximity-px pixels)
  3. If two adjacent masks share the same predicted parent → merge (union) and
     re-classify the merged crop
  4. Proceed with the merged mask set for reconciliation

This is a post-processing step applied before run_pipeline.py's reconciliation.
Run this script to produce merged mask files, then pass --sam-dir to that output.

Usage:
    python scripts/run_merge.py \\
        --sam-dir out/sam_eval_top200_matador_auto_vit_b \\
        --hgnn-run runs/minc_hgnn \\
        --out-dir out/sam_merged
"""

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.minc import MINC2500Dataset
from gnn_classifier.hgnn import HGNN
from taxonomy.tree import get_taxonomy

CATEGORIES    = MINC2500Dataset.CATEGORIES
OTHER_IDX     = CATEGORIES.index("other")
IMAGENET_MEAN = (123, 116, 103)

TRANSFORM = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model loading (shared with run_pipeline.py) ───────────────────────────────

def load_hgnn(run_dir: Path, device: torch.device):
    ckpts = sorted(run_dir.glob("*/checkpoint_best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    ckpt_path   = ckpts[-1]
    cfg         = json.load(open(ckpt_path.parent / "config.json"))
    ckpt        = torch.load(ckpt_path, map_location=device)
    node_to_idx = ckpt["node_to_idx"]

    tax_path = repo_root / "taxonomy/assets/minc-taxonomy.json"
    g_raw    = get_taxonomy(str(tax_path))
    node_order = list(nx.topological_sort(g_raw))
    graph = nx.DiGraph()
    for n in node_order:
        graph.add_node(n, **g_raw.nodes[n])
    for u, v in g_raw.edges:
        graph.add_edge(u, v, **g_raw.edges[u, v])

    model = HGNN(
        graph=graph,
        cnn_kwargs=dict(backbone=cfg["backbone"], pretrained=False,
                        output_dim=cfg["cnn_output_dim"], finetune=False),
        gnn_kwargs=dict(input_dim=cfg["cnn_output_dim"], hidden_dim=cfg["gnn_hidden_dim"],
                        output_dim=cfg["gnn_output_dim"], num_layers=cfg["gnn_layers"],
                        num_heads=1, skip_connection=True, dropout=cfg["dropout"]),
        dropout_prob=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    leaves = [n for n in graph.nodes if graph.out_degree(n) == 0]
    leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                dtype=torch.long, device=device)
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    leaf_to_cat = torch.zeros(len(leaf_indices), dtype=torch.long, device=device)
    for pos, idx in enumerate(leaf_indices.tolist()):
        leaf_to_cat[pos] = CATEGORIES.index(idx_to_node[idx])

    # Also build leaf → parent mapping using taxonomy
    leaf_to_parent = {}
    for n in graph.nodes:
        if graph.out_degree(n) == 0:  # leaf
            parents = list(graph.predecessors(n))
            leaf_to_parent[n] = parents[0] if parents else "root"
    cat_to_parent = {}
    for pos, idx in enumerate(leaf_indices.tolist()):
        cat_idx = int(leaf_to_cat[pos].item())
        node_name = idx_to_node[idx]
        cat_to_parent[cat_idx] = leaf_to_parent.get(node_name, "root")

    return model, leaf_indices, leaf_to_cat, cat_to_parent


# ── Crop + classify ────────────────────────────────────────────────────────────

def masked_crop(photo: np.ndarray, mask: np.ndarray) -> Image.Image | None:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    crop = photo[r0:r1, c0:c1].copy()
    m    = mask[r0:r1, c0:c1]
    for ch, mv in enumerate(IMAGENET_MEAN):
        crop[:, :, ch][~m] = mv
    return Image.fromarray(crop)


@torch.no_grad()
def classify_one(model, leaf_indices, leaf_to_cat,
                 crop: Image.Image, device: torch.device) -> tuple:
    """Returns (label_idx, max_softmax_conf)."""
    img    = TRANSFORM(crop).unsqueeze(0).to(device)
    logits = model(img)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    leaf_logits = logits[:, leaf_indices]
    probs = F.softmax(leaf_logits, dim=1)[0]
    max_conf, pos = probs.max(0)
    return int(leaf_to_cat[pos].item()), float(max_conf.item())


# ── Adjacency check ────────────────────────────────────────────────────────────

def bbox_of(mask: np.ndarray) -> tuple:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None
    return rows[0], rows[-1], cols[0], cols[-1]


def boxes_adjacent(b1: tuple, b2: tuple, proximity_px: int) -> bool:
    r0a, r1a, c0a, c1a = b1
    r0b, r1b, c0b, c1b = b2
    row_gap = max(0, max(r0a, r0b) - min(r1a, r1b))
    col_gap = max(0, max(c0a, c0b) - min(c1a, c1b))
    return row_gap <= proximity_px and col_gap <= proximity_px


# ── Per-photo merging ──────────────────────────────────────────────────────────

def merge_photo(photo_np: np.ndarray, masks_data: list,
                model, leaf_indices, leaf_to_cat, cat_to_parent,
                device: torch.device, proximity_px: int) -> tuple:
    """
    Classify all masks and merge adjacent ones that share a predicted parent.
    Returns (merged_masks_data, n_merges).
    """
    # Step 1: classify all masks
    classifications = []
    bboxes = []
    for md in masks_data:
        mask = md["segmentation"].astype(bool)
        crop = masked_crop(photo_np, mask)
        if crop is None:
            classifications.append((OTHER_IDX, 0.0))
        else:
            classifications.append(classify_one(model, leaf_indices, leaf_to_cat, crop, device))
        bboxes.append(bbox_of(mask))

    n_merges = 0
    merged_flags = [False] * len(masks_data)
    output_masks = []

    for i in range(len(masks_data)):
        if merged_flags[i]:
            continue
        label_i, _ = classifications[i]
        parent_i = cat_to_parent.get(label_i, "root")

        merge_set = [i]
        for j in range(i + 1, len(masks_data)):
            if merged_flags[j]:
                continue
            label_j, _ = classifications[j]
            if cat_to_parent.get(label_j, "root") != parent_i:
                continue
            if bboxes[i] and bboxes[j] and boxes_adjacent(bboxes[i], bboxes[j], proximity_px):
                merge_set.append(j)
                merged_flags[j] = True

        if len(merge_set) > 1:
            # Union the masks and re-classify
            merged_mask = np.zeros_like(masks_data[0]["segmentation"], dtype=bool)
            for k in merge_set:
                merged_mask |= masks_data[k]["segmentation"].astype(bool)
            new_md = dict(masks_data[i])
            new_md["segmentation"] = merged_mask
            crop = masked_crop(photo_np, merged_mask)
            if crop is not None:
                label, conf = classify_one(model, leaf_indices, leaf_to_cat, crop, device)
            else:
                label, conf = OTHER_IDX, 0.0
            output_masks.append((new_md, label, conf))
            n_merges += 1
        else:
            output_masks.append((masks_data[i], *classifications[i]))

    return output_masks, n_merges


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sam-dir",      default="out/sam_eval_top200_matador_auto_vit_b")
    p.add_argument("--hgnn-run",     default="runs/minc_hgnn")
    p.add_argument("--photos-dir",   default="data/external/minc/minc-s/photos")
    p.add_argument("--out-dir",      default="out/sam_merged")
    p.add_argument("--proximity-px", type=int, default=15,
                   help="Max bounding-box gap (pixels) to consider two masks adjacent")
    p.add_argument("--max-photos",   type=int, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hgnn_dir = repo_root / args.hgnn_run
    model, leaf_indices, leaf_to_cat, cat_to_parent = load_hgnn(hgnn_dir, device)

    photos_dir = repo_root / args.photos_dir
    sam_dir    = repo_root / args.sam_dir
    out_dir    = repo_root / args.out_dir

    photos = sorted(photos_dir.glob("*.jpg")) + sorted(photos_dir.glob("*.png"))
    if args.max_photos:
        photos = photos[:args.max_photos]
    print(f"Processing {len(photos)} photos...")

    total_merges = total_before = total_after = 0
    for i, photo_path in enumerate(photos, 1):
        photo_id  = photo_path.stem
        masks_path = sam_dir / "masks" / f"{photo_id}.json"
        if not masks_path.exists():
            continue

        masks_data = json.load(open(masks_path))
        photo_np   = np.array(Image.open(photo_path).convert("RGB"))

        output_masks, n_merges = merge_photo(
            photo_np, masks_data, model, leaf_indices, leaf_to_cat,
            cat_to_parent, device, args.proximity_px
        )
        total_merges += n_merges
        total_before += len(masks_data)
        total_after  += len(output_masks)

        # Save merged masks
        out_masks_dir = out_dir / "masks"
        out_masks_dir.mkdir(parents=True, exist_ok=True)
        save_data = []
        for md, label, conf in output_masks:
            entry = {k: v for k, v in md.items() if k != "segmentation"}
            entry["segmentation"]   = md["segmentation"].tolist() \
                if isinstance(md["segmentation"], np.ndarray) else md["segmentation"]
            entry["pred_label"]     = label
            entry["pred_conf"]      = conf
            save_data.append(entry)
        with open(out_masks_dir / f"{photo_id}.json", "w") as f:
            json.dump(save_data, f)

        if i % 20 == 0 or i == len(photos):
            print(f"  [{i}/{len(photos)}] merges so far: {total_merges}")

    print(f"\nDone.")
    print(f"  Merge events:    {total_merges}")
    print(f"  Avg masks before: {total_before/len(photos):.1f}")
    print(f"  Avg masks after:  {total_after/len(photos):.1f}")

    with open(out_dir / "merge_stats.json", "w") as f:
        json.dump({"total_merges": total_merges,
                   "n_photos": len(photos),
                   "avg_before": total_before / len(photos),
                   "avg_after":  total_after  / len(photos)}, f, indent=2)


if __name__ == "__main__":
    main()
