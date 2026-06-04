#!/usr/bin/env python3
"""
Evaluate patch classifiers on MINC-S real-world segment masks.

For each test segment in MINC-S:
  1. Load the photo and binary mask
  2. Crop the bounding-box of the mask region
  3. Optionally zero pixels outside mask (masked crop mode)
  4. Run through each model and record predicted label

Reports accuracy, CHD, Hier@d2 per model, plus per-class breakdown.
Also renders visualization overlays for the first N images.

Usage:
    python scripts/eval_minc_s_patch_classifiers.py \
        --models flat:runs/minc_flat/20260603_051833 \
                 hier:runs/minc_hier_smooth/20260603_063143 \
                 unif:runs/minc_unif_smooth/20260603_072359 \
        --dataset-root data/external/minc/minc-s \
        --taxonomy taxonomy/assets/minc-taxonomy.json \
        --out-dir runs/minc_s_eval \
        --visualize-n 10
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image, ImageDraw, ImageFont

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

CATEGORIES = [
    "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass",
    "hair", "leather", "metal", "mirror", "other", "painted", "paper",
    "plastic", "polishedstone", "skin", "sky", "stone", "tile",
    "wallpaper", "water", "wood",
]
NUM_CLASSES = len(CATEGORIES)  # 23


# --------------------------------------------------------------------------- #
# Taxonomy distance matrix
# --------------------------------------------------------------------------- #

def build_distance_matrix(taxonomy_path: str) -> np.ndarray:
    """Return NUM_CLASSES × NUM_CLASSES tree-distance matrix for leaf nodes."""
    import networkx as nx
    from taxonomy.tree import get_taxonomy

    g = get_taxonomy(taxonomy_path)
    undirected = g.to_undirected()

    D = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    for i, ci in enumerate(CATEGORIES):
        for j, cj in enumerate(CATEGORIES):
            if i == j:
                continue
            try:
                D[i, j] = nx.shortest_path_length(undirected, ci, cj)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                D[i, j] = 10.0  # large fallback
    return D


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(run_dir: Path, device: torch.device) -> nn.Module:
    model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# MINC-S segment loading
# --------------------------------------------------------------------------- #

def parse_segments(segments_txt: Path):
    """Returns list of (label_idx, photo_id, shape_id)."""
    entries = []
    with open(segments_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            entries.append((int(parts[0]), parts[1], parts[2]))
    return entries


def load_segment_crop(
    photos_dir: Path,
    segments_dir: Path,
    photo_id: str,
    shape_id: str,
    mask_crop: bool = True,
    pad_frac: float = 0.1,
) -> Image.Image | None:
    """
    Load masked crop for a single MINC-S segment.

    Returns PIL RGB image or None if files missing.
    """
    photo_path = photos_dir / f"{photo_id}.jpg"
    mask_path = segments_dir / f"{photo_id}_{shape_id}.png"

    if not photo_path.exists() or not mask_path.exists():
        return None

    photo = Image.open(photo_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Resize mask to photo size if needed
    if mask.size != photo.size:
        mask = mask.resize(photo.size, Image.NEAREST)

    mask_np = np.array(mask) > 127
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = photo.size[1], photo.size[0]
    pad_r = max(1, int((rmax - rmin + 1) * pad_frac))
    pad_c = max(1, int((cmax - cmin + 1) * pad_frac))
    rmin = max(0, rmin - pad_r)
    rmax = min(h - 1, rmax + pad_r)
    cmin = max(0, cmin - pad_c)
    cmax = min(w - 1, cmax + pad_c)

    crop = photo.crop((cmin, rmin, cmax + 1, rmax + 1))

    if mask_crop:
        crop_arr = np.array(crop)
        mask_crop_arr = mask_np[rmin:rmax+1, cmin:cmax+1]
        # Set pixels outside mask to ImageNet mean (neutral background)
        mean_rgb = np.array([123, 116, 103], dtype=np.uint8)
        crop_arr[~mask_crop_arr] = mean_rgb
        crop = Image.fromarray(crop_arr)

    return crop


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def preload_crops(
    entries,
    photos_dir: Path,
    segments_dir: Path,
    mask_crop: bool = True,
) -> tuple:
    """Pre-load all segment crops into memory (photo-sorted for cache efficiency).
    Returns (crops_list, labels_list) where crops_list[i] is PIL Image or None."""
    # Sort by photo_id to minimize photo re-opens
    indexed = list(enumerate(entries))
    indexed_sorted = sorted(indexed, key=lambda x: x[1][1])  # sort by photo_id

    crops = [None] * len(entries)
    labels = [None] * len(entries)
    missing = 0
    photo_cache: dict = {}

    errors = []
    for orig_idx, (label_idx, photo_id, shape_id) in indexed_sorted:
        photo_path = photos_dir / f"{photo_id}.jpg"
        mask_path = segments_dir / f"{photo_id}_{shape_id}.png"

        if not photo_path.exists() or not mask_path.exists():
            missing += 1
            continue

        try:
            # Cache decoded photo to avoid re-decoding for multiple segments
            if photo_id not in photo_cache:
                photo_cache = {photo_id: Image.open(photo_path).convert("RGB")}
            photo = photo_cache[photo_id]

            mask = Image.open(mask_path).convert("L")
            if mask.size != photo.size:
                mask = mask.resize(photo.size, Image.NEAREST)
            mask_np = np.array(mask) > 127
            rows = np.any(mask_np, axis=1)
            cols = np.any(mask_np, axis=0)
            if not rows.any():
                missing += 1
                continue

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            h, w = photo.size[1], photo.size[0]
            pad_frac = 0.1
            pad_r = max(1, int((rmax - rmin + 1) * pad_frac))
            pad_c = max(1, int((cmax - cmin + 1) * pad_frac))
            rmin = max(0, rmin - pad_r)
            rmax = min(h - 1, rmax + pad_r)
            cmin = max(0, cmin - pad_c)
            cmax = min(w - 1, cmax + pad_c)

            crop = photo.crop((cmin, rmin, cmax + 1, rmax + 1))
            if mask_crop:
                crop_arr = np.array(crop)
                mask_crop_arr = mask_np[rmin:rmax+1, cmin:cmax+1]
                mean_rgb = np.array([123, 116, 103], dtype=np.uint8)
                crop_arr[~mask_crop_arr] = mean_rgb
                crop = Image.fromarray(crop_arr)

            crops[orig_idx] = crop
            labels[orig_idx] = label_idx
        except Exception as e:
            errors.append(f"{photo_id}_{shape_id}: {e}")
            missing += 1
            photo_cache = {}  # clear cache in case the photo itself is corrupt

    if errors:
        print(f"  Errors ({len(errors)}): {errors[:5]}")
    print(f"  Pre-loaded crops: {sum(c is not None for c in crops)}/{len(entries)}, "
          f"missing: {missing}")
    return crops, labels


def evaluate_model(
    model: nn.Module,
    crops,
    labels_list,
    transform,
    device: torch.device,
    batch_size: int = 64,
):
    """Run inference on pre-loaded crops. Returns (preds, labels) arrays."""
    all_preds = []
    all_labels = []

    imgs_batch = []
    labels_batch = []

    def flush():
        if not imgs_batch:
            return
        batch = torch.stack(imgs_batch).to(device)
        with torch.no_grad():
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels_batch)
        imgs_batch.clear()
        labels_batch.clear()

    for crop, label_idx in zip(crops, labels_list):
        if crop is None or label_idx is None:
            continue
        imgs_batch.append(transform(crop))
        labels_batch.append(label_idx)
        if len(imgs_batch) >= batch_size:
            flush()

    flush()
    return np.array(all_preds), np.array(all_labels)


def stream_eval_all_models(
    entries,
    photos_dir: Path,
    segments_dir: Path,
    models: dict,
    transform,
    device: torch.device,
    mask_crop: bool = True,
    batch_size: int = 64,
) -> dict:
    """Stream through photos once, evaluate all models simultaneously.

    Never stores more than one photo in RAM. All models must fit in VRAM.
    Returns {model_name: (preds_array, labels_array)}.
    """
    indexed_sorted = sorted(enumerate(entries), key=lambda x: x[1][1])
    model_names = list(models.keys())

    # Accumulators per model
    batches = {n: [] for n in model_names}
    label_batches = {n: [] for n in model_names}
    all_preds = {n: [] for n in model_names}
    all_labels = {n: [] for n in model_names}

    def flush_batches():
        for name, model in models.items():
            if not batches[name]:
                continue
            batch = torch.stack(batches[name]).to(device)
            with torch.no_grad():
                logits = model(batch)
                preds = logits.argmax(dim=1).cpu().numpy()
            all_preds[name].extend(preds.tolist())
            all_labels[name].extend(label_batches[name])
            batches[name].clear()
            label_batches[name].clear()

    photo_cache = {}
    missing = 0
    ok = 0
    total = len(entries)

    for _orig_idx, (label_idx, photo_id, shape_id) in indexed_sorted:
        photo_path = photos_dir / f"{photo_id}.jpg"
        mask_path = segments_dir / f"{photo_id}_{shape_id}.png"
        if not photo_path.exists() or not mask_path.exists():
            missing += 1
            continue

        try:
            if photo_id not in photo_cache:
                photo_cache = {photo_id: Image.open(photo_path).convert("RGB")}
            photo = photo_cache[photo_id]

            mask = Image.open(mask_path).convert("L")
            if mask.size != photo.size:
                mask = mask.resize(photo.size, Image.NEAREST)
            mask_np = np.array(mask) > 127
            rows = np.any(mask_np, axis=1)
            cols = np.any(mask_np, axis=0)
            if not rows.any():
                missing += 1
                continue

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            h, w = photo.size[1], photo.size[0]
            pad_r = max(1, int((rmax - rmin + 1) * 0.1))
            pad_c = max(1, int((cmax - cmin + 1) * 0.1))
            rmin = max(0, rmin - pad_r)
            rmax = min(h - 1, rmax + pad_r)
            cmin = max(0, cmin - pad_c)
            cmax = min(w - 1, cmax + pad_c)

            crop = photo.crop((cmin, rmin, cmax + 1, rmax + 1))
            if mask_crop:
                crop_arr = np.array(crop)
                mask_crop_arr = mask_np[rmin:rmax+1, cmin:cmax+1]
                crop_arr[~mask_crop_arr] = np.array([123, 116, 103], dtype=np.uint8)
                crop = Image.fromarray(crop_arr)

            tensor = transform(crop)
            for name in model_names:
                batches[name].append(tensor)
                label_batches[name].append(label_idx)

            ok += 1
            if ok % batch_size == 0:
                flush_batches()
            if ok % 1000 == 0:
                print(f"  Progress: {ok}/{total} segments", flush=True)

        except Exception as e:
            missing += 1
            photo_cache = {}

    flush_batches()
    print(f"  Streamed {ok}/{total}, missing/errors: {missing}")
    return {n: (np.array(all_preds[n]), np.array(all_labels[n])) for n in model_names}


def compute_metrics(preds: np.ndarray, labels: np.ndarray, dist_matrix: np.ndarray):
    correct = (preds == labels)
    acc = correct.mean()

    dists = dist_matrix[labels, preds]
    chd = dists.mean()

    hier_d2 = (dists <= 2).mean()

    return {"accuracy": float(acc), "CHD": float(chd), "Hier@d2": float(hier_d2)}


def per_class_metrics(preds, labels, dist_matrix):
    results = {}
    for ci, cat in enumerate(CATEGORIES):
        mask = labels == ci
        if mask.sum() == 0:
            continue
        p = preds[mask]
        l = labels[mask]
        acc = (p == l).mean()
        dists = dist_matrix[l, p]
        chd = dists.mean()
        results[cat] = {"n": int(mask.sum()), "accuracy": float(acc), "CHD": float(chd)}
    return results


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

def visualize_predictions(
    model_names,
    model_preds_dict,
    entries,
    photos_dir,
    segments_dir,
    out_dir: Path,
    n: int = 10,
):
    """Save side-by-side overlays showing ground truth vs model predictions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shown = 0
    entry_idx = 0

    for label_idx, photo_id, shape_id in entries:
        if shown >= n:
            break
        photo_path = photos_dir / f"{photo_id}.jpg"
        mask_path = segments_dir / f"{photo_id}_{shape_id}.png"
        if not photo_path.exists() or not mask_path.exists():
            entry_idx += 1
            continue

        photo = Image.open(photo_path).convert("RGB").resize((400, 300))
        mask = Image.open(mask_path).convert("L").resize((400, 300), Image.NEAREST)
        mask_np = np.array(mask) > 127

        # Draw overlay: green for correct, red for wrong
        overlay = photo.copy()
        overlay_arr = np.array(overlay)
        # Highlight segment region with blue tint
        blue_tint = overlay_arr.copy()
        blue_tint[mask_np, 2] = np.clip(blue_tint[mask_np, 2] + 80, 0, 255)
        overlay_arr[mask_np] = blue_tint[mask_np]
        overlay = Image.fromarray(overlay_arr)

        # Text annotations
        draw = ImageDraw.Draw(overlay)
        gt_name = CATEGORIES[label_idx]
        draw.rectangle([0, 0, 400, 20], fill=(0, 0, 0, 180))
        draw.text((5, 3), f"GT: {gt_name}", fill=(255, 255, 255))

        y_off = 25
        for mname in model_names:
            if entry_idx < len(model_preds_dict[mname]):
                pred_idx = model_preds_dict[mname][entry_idx]
                pred_name = CATEGORIES[pred_idx]
                color = (80, 220, 80) if pred_idx == label_idx else (220, 80, 80)
                draw.rectangle([0, y_off, 400, y_off + 18], fill=(0, 0, 0, 160))
                draw.text((5, y_off + 2), f"{mname}: {pred_name}", fill=color)
                y_off += 20

        out_path = out_dir / f"vis_{shown:04d}_{photo_id}.jpg"
        overlay.save(out_path, quality=85)
        shown += 1
        entry_idx += 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models", nargs="+",
        default=[
            "flat:runs/minc_flat/20260603_051833",
            "hier:runs/minc_hier_smooth/20260603_063143",
            "unif:runs/minc_unif_smooth/20260603_072359",
        ],
        help="name:run_dir pairs",
    )
    p.add_argument("--dataset-root", default="data/external/minc/minc-s")
    p.add_argument("--taxonomy", default="taxonomy/assets/minc-taxonomy.json")
    p.add_argument("--out-dir", default="runs/minc_s_eval")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mask-crop", action="store_true", default=True,
                   help="Zero pixels outside mask (default: True)")
    p.add_argument("--no-mask-crop", dest="mask_crop", action="store_false")
    p.add_argument("--visualize-n", type=int, default=20)
    p.add_argument("--max-segments", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = repo / args.dataset_root
    photos_dir = dataset_root / "photos"
    segments_dir = dataset_root / "segments"
    segments_txt = dataset_root / "test-segments.txt"

    print("Loading taxonomy distances...")
    dist_matrix = build_distance_matrix(str(repo / args.taxonomy))

    print("Loading MINC-S segments...")
    entries = parse_segments(segments_txt)
    if args.max_segments:
        entries = entries[: args.max_segments]
    print(f"  {len(entries)} test segments")

    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load all models first, then stream photos once for all models simultaneously
    print("\nLoading models...")
    models_dict = {}
    for model_spec in args.models:
        name, run_dir_str = model_spec.split(":", 1)
        run_dir = repo / run_dir_str
        models_dict[name] = load_model(run_dir, device)
        print(f"  Loaded: {name}")

    mask_label = "masked" if args.mask_crop else "bbox_only"
    print(f"\nStreaming eval ({mask_label}, all models simultaneously)...")
    stream_results = stream_eval_all_models(
        entries, photos_dir, segments_dir,
        models_dict, transform, device,
        mask_crop=args.mask_crop, batch_size=args.batch_size,
    )

    results = {}
    all_preds_dict = {}

    for name, (preds, labels) in stream_results.items():
        metrics = compute_metrics(preds, labels, dist_matrix)
        per_class = per_class_metrics(preds, labels, dist_matrix)
        results[name] = {"metrics": metrics, "per_class": per_class}
        all_preds_dict[name] = preds
        print(f"  {name}: Accuracy={metrics['accuracy']:.4f}  CHD={metrics['CHD']:.4f}  Hier@d2={metrics['Hier@d2']:.4f}")

    # Save JSON results
    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Print summary table
    print("\n=== MINC-S Segmentation Evaluation Summary ===")
    print(f"{'Model':<12} {'Accuracy':>10} {'CHD':>8} {'Hier@d2':>10}")
    print("-" * 44)
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<12} {m['accuracy']:>10.4f} {m['CHD']:>8.4f} {m['Hier@d2']:>10.4f}")

    # Per-class table
    print("\n=== Per-Class Accuracy ===")
    model_names = list(results.keys())
    header = f"{'Category':<16}" + "".join(f" {n:>10}" for n in model_names)
    print(header)
    print("-" * (16 + 11 * len(model_names)))
    for cat in CATEGORIES:
        row = f"{cat:<16}"
        for name in model_names:
            pc = results[name]["per_class"].get(cat)
            if pc:
                row += f" {pc['accuracy']:>10.3f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # Visualize
    if args.visualize_n > 0:
        vis_dir = out_dir / "visualizations"
        print(f"\nGenerating {args.visualize_n} visualizations -> {vis_dir}")
        visualize_predictions(
            model_names=list(results.keys()),
            model_preds_dict=all_preds_dict,
            entries=entries,
            photos_dir=photos_dir,
            segments_dir=segments_dir,
            out_dir=vis_dir,
            n=args.visualize_n,
        )


if __name__ == "__main__":
    main()
