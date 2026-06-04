#!/usr/bin/env python3
"""
SAM + HGNN material segmentation pipeline.

Produces pixel-level material segmentation maps from a set of scene photos:
  1. Load pre-computed SAM masks for each photo (auto mode, ~64 masks/image)
  2. For each mask, extract the masked crop (non-mask pixels → ImageNet mean)
  3. Classify each crop with the HGNN → softmax confidence vector
  4. For each pixel: assign label from the covering mask with highest max-softmax
  5. Low-confidence pixels (max-softmax < conf_threshold) → 'other' (index 11)
  6. SAM-uncovered pixels → 'other'

Notes:
  - 'other' is the correct default for uncovered pixels: SAM is object/region-driven
    and is unlikely to generate proposals for genuinely ambiguous material regions,
    which are exactly the regions that should map to 'other'.
  - The confidence threshold avoids hard wrong predictions propagating into the
    pixel map when the classifier is uncertain.

Usage:
    python scripts/run_pipeline.py \\
        --sam-dir out/sam_eval_top200_matador_auto_vit_b \\
        --hgnn-run runs/minc_hgnn \\
        --out-dir out/pipeline_maps \\
        --photos data/external/minc/minc-s/photos

    # Evaluate against MINC-S GT segments after running:
    python scripts/run_pipeline.py --evaluate
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
NUM_CLASSES   = len(CATEGORIES)
OTHER_IDX     = CATEGORIES.index("other")     # 11
IMAGENET_MEAN = (123, 116, 103)

TRANSFORM = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model loading ──────────────────────────────────────────────────────────────

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

    leaves      = [n for n in graph.nodes if graph.out_degree(n) == 0]
    leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                dtype=torch.long, device=device)
    idx_to_node  = {v: k for k, v in node_to_idx.items()}
    leaf_to_cat  = torch.zeros(len(leaf_indices), dtype=torch.long, device=device)
    for pos, idx in enumerate(leaf_indices.tolist()):
        leaf_to_cat[pos] = CATEGORIES.index(idx_to_node[idx])

    return model, leaf_indices, leaf_to_cat


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
def classify_masks(model, leaf_indices, leaf_to_cat,
                   masks_data: list, photo_np: np.ndarray,
                   device: torch.device) -> list:
    """
    Classify each SAM mask crop with HGNN.

    Returns list of (label_idx, max_softmax_confidence) per mask.
    """
    results = []
    for mask_dict in masks_data:
        mask = mask_dict["segmentation"].astype(bool)
        crop = masked_crop(photo_np, mask)
        if crop is None:
            results.append((OTHER_IDX, 0.0))
            continue

        img    = TRANSFORM(crop).unsqueeze(0).to(device)
        logits = model(img)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        leaf_logits = logits[:, leaf_indices]
        probs = F.softmax(leaf_logits, dim=1)[0]
        max_conf, pos = probs.max(0)
        label = int(leaf_to_cat[pos].item())
        results.append((label, float(max_conf.item())))
    return results


# ── Reconciliation ─────────────────────────────────────────────────────────────

def reconcile(photo_np: np.ndarray, masks_data: list,
              classifications: list, conf_threshold: float) -> np.ndarray:
    """
    Build pixel-level label map from SAM masks + HGNN classifications.

    For each pixel, assign the label of the covering mask with the highest
    max-softmax confidence, provided it exceeds conf_threshold.
    Uncovered and low-confidence pixels → OTHER_IDX.
    """
    H, W   = photo_np.shape[:2]
    label_map = np.full((H, W), OTHER_IDX, dtype=np.int32)
    conf_map  = np.zeros((H, W), dtype=np.float32)

    for mask_dict, (label, conf) in zip(masks_data, classifications):
        if conf < conf_threshold:
            continue
        m = mask_dict["segmentation"].astype(bool)
        update = m & (conf > conf_map)
        label_map[update] = label
        conf_map[update]  = conf

    return label_map


# ── Pixel-level evaluation ────────────────────────────────────────────────────

def evaluate_pixel_map(label_map: np.ndarray, gt_segments: list) -> dict:
    """
    Evaluate a pixel-level label map against GT segment annotations.

    Reports: pixel accuracy (covered regions only), coverage fraction,
    per-class accuracy.
    """
    H, W = label_map.shape
    covered = label_map != OTHER_IDX
    coverage = float(covered.mean())

    correct = total = 0
    for seg in gt_segments:
        mask  = seg["mask"]
        label = seg["label_index"]
        if mask.shape != (H, W):
            continue
        in_covered = mask & covered
        if in_covered.sum() == 0:
            continue
        preds_here = label_map[in_covered]
        correct += (preds_here == label).sum()
        total   += in_covered.sum()

    pixel_acc = float(correct / total) if total > 0 else 0.0
    return {"pixel_acc": pixel_acc, "coverage": coverage,
            "correct": int(correct), "total": int(total)}


# ── Per-photo pipeline ─────────────────────────────────────────────────────────

def process_photo(photo_path: Path, sam_dir: Path, out_dir: Path,
                  model, leaf_indices, leaf_to_cat,
                  conf_threshold: float, device: torch.device) -> dict:
    """Run the full pipeline for a single photo."""
    photo_id = photo_path.stem
    masks_path = sam_dir / "masks" / f"{photo_id}.json"
    if not masks_path.exists():
        return {"photo_id": photo_id, "status": "no_masks"}

    masks_data = json.load(open(masks_path))
    photo_np   = np.array(Image.open(photo_path).convert("RGB"))

    classifications = classify_masks(model, leaf_indices, leaf_to_cat,
                                     masks_data, photo_np, device)
    label_map = reconcile(photo_np, masks_data, classifications, conf_threshold)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{photo_id}_labels.npy", label_map.astype(np.uint8))

    covered  = label_map != OTHER_IDX
    coverage = float(covered.mean())
    return {"photo_id": photo_id, "status": "ok",
            "n_masks": len(masks_data), "coverage": coverage}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sam-dir",        default="out/sam_eval_top200_matador_auto_vit_b")
    p.add_argument("--hgnn-run",       default="runs/minc_hgnn")
    p.add_argument("--photos-dir",     default="data/external/minc/minc-s/photos")
    p.add_argument("--out-dir",        default="out/pipeline_maps")
    p.add_argument("--conf-threshold", type=float, default=0.3,
                   help="Min max-softmax confidence; below this → 'other'")
    p.add_argument("--max-photos",     type=int, default=None,
                   help="Limit number of photos (for testing)")
    p.add_argument("--dry-run",        action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hgnn_dir = repo_root / args.hgnn_run
    print(f"Loading HGNN from {hgnn_dir}...")
    model, leaf_indices, leaf_to_cat = load_hgnn(hgnn_dir, device)

    photos_dir = repo_root / args.photos_dir
    sam_dir    = repo_root / args.sam_dir
    out_dir    = repo_root / args.out_dir

    photos = sorted(photos_dir.glob("*.jpg")) + sorted(photos_dir.glob("*.png"))
    if args.max_photos:
        photos = photos[:args.max_photos]
    print(f"Processing {len(photos)} photos...")

    if args.dry_run:
        print("Dry run OK.")
        return

    stats = []
    for i, photo_path in enumerate(photos, 1):
        result = process_photo(photo_path, sam_dir, out_dir,
                               model, leaf_indices, leaf_to_cat,
                               args.conf_threshold, device)
        stats.append(result)
        if i % 10 == 0 or i == len(photos):
            ok = [s for s in stats if s["status"] == "ok"]
            if ok:
                avg_cov = np.mean([s["coverage"] for s in ok])
                print(f"  [{i}/{len(photos)}] "
                      f"avg coverage={avg_cov:.3f}  ok={len(ok)}/{i}")

    ok_stats = [s for s in stats if s["status"] == "ok"]
    print(f"\nDone. {len(ok_stats)}/{len(photos)} photos processed.")
    if ok_stats:
        avg_cov = np.mean([s["coverage"] for s in ok_stats])
        avg_masks = np.mean([s["n_masks"] for s in ok_stats])
        print(f"  Avg pixel coverage:  {avg_cov:.3f}")
        print(f"  Avg masks/photo:     {avg_masks:.1f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pipeline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {out_dir / 'pipeline_stats.json'}")


if __name__ == "__main__":
    main()
