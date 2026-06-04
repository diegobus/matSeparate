#!/usr/bin/env python3
"""
End-to-end SAM + classifier evaluation on MINC-S.

For each GT segment:
  1. Load the best-matching SAM mask (from cached auto results, IoU > threshold)
  2. Crop the SAM mask region from the photo (with masking = gray outside)
  3. Classify with our patch classifiers
  4. Report accuracy vs GT label

This measures the full pipeline: SAM finds the region, classifier labels it.

Usage:
    python scripts/eval_sam_classification.py
    python scripts/eval_sam_classification.py --iou-threshold 0.5 --no-mask-crop
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from gnn_classifier.hgnn import HGNN
from taxonomy.tree import get_taxonomy

CATEGORIES = [
    "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass",
    "hair", "leather", "metal", "mirror", "other", "painted", "paper",
    "plastic", "polishedstone", "skin", "sky", "stone", "tile",
    "wallpaper", "water", "wood",
]
NUM_CLASSES = len(CATEGORIES)
IMAGENET_MEAN_RGB = np.array([123, 116, 103], dtype=np.uint8)


def build_distance_matrix(taxonomy_path):
    import networkx as nx
    from taxonomy.tree import get_taxonomy
    g = get_taxonomy(taxonomy_path)
    u = g.to_undirected()
    D = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
    for i, ci in enumerate(CATEGORIES):
        for j, cj in enumerate(CATEGORIES):
            if i != j:
                try:
                    D[i, j] = nx.shortest_path_length(u, ci, cj)
                except Exception:
                    D[i, j] = 10.0
    return D


def load_model(run_dir, device):
    model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
    ckpt = torch.load(Path(run_dir) / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_hgnn(run_dir, taxonomy_path, device):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    node_to_idx = ckpt["node_to_idx"]
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    g = get_taxonomy(taxonomy_path)
    node_order = list(nx.topological_sort(g))
    g_canon = nx.DiGraph()
    for n in node_order:
        g_canon.add_node(n, **g.nodes[n])
    for u, v in g.edges:
        g_canon.add_edge(u, v, **g.edges[u, v])

    model = HGNN(
        graph=g_canon,
        cnn_kwargs=dict(backbone=cfg["backbone"], pretrained=False,
                        output_dim=cfg["cnn_output_dim"], finetune=False),
        gnn_kwargs=dict(input_dim=cfg["cnn_output_dim"], hidden_dim=cfg["gnn_hidden_dim"],
                        output_dim=cfg["gnn_output_dim"], num_layers=cfg["gnn_layers"],
                        num_heads=1, skip_connection=True, dropout=cfg["dropout"]),
        dropout_prob=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    leaves = [n for n in g_canon.nodes if g_canon.out_degree(n) == 0]
    leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                dtype=torch.long, device=device)
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    leaf_to_cat = torch.zeros(len(leaf_indices), dtype=torch.long, device=device)
    for pos, idx in enumerate(leaf_indices.tolist()):
        leaf_to_cat[pos] = CATEGORIES.index(idx_to_node[idx])

    return model, leaf_indices, leaf_to_cat


def load_cached_masks(cache_dir, photo_id, model_type, image_shape):
    """Load cached SAM auto masks for a photo. Tries different shape suffixes."""
    h, w = image_shape
    pattern = f"{photo_id}__{model_type}__{h}x{w}.pkl"
    path = cache_dir / pattern
    if not path.exists():
        # Try any cached file for this photo_id
        matches = list(cache_dir.glob(f"{photo_id}__{model_type}__*.pkl"))
        if not matches:
            return None
        path = matches[0]
    with open(path, "rb") as f:
        return pickle.load(f)


def crop_sam_mask(photo_np, sam_mask_np, mask_crop=True, target_size=224):
    """Crop the SAM mask region from the photo. Optionally apply masking."""
    rows = np.any(sam_mask_np, axis=1)
    cols = np.any(sam_mask_np, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    crop = photo_np[rmin:rmax+1, cmin:cmax+1].copy()
    if mask_crop:
        local_mask = sam_mask_np[rmin:rmax+1, cmin:cmax+1]
        crop[~local_mask] = IMAGENET_MEAN_RGB

    return Image.fromarray(crop)


def compute_metrics(preds, labels, D):
    correct = (preds == labels)
    dists = D[labels, preds]
    return {
        "accuracy": float(correct.mean()),
        "CHD": float(dists.mean()),
        "Hier@d2": float((dists <= 2).mean()),
        "n": int(len(preds)),
    }


def per_class_metrics(preds, labels, D):
    res = {}
    for ci, cat in enumerate(CATEGORIES):
        mask = labels == ci
        if not mask.any():
            continue
        p, l = preds[mask], labels[mask]
        dists = D[l, p]
        res[cat] = {"n": int(mask.sum()), "accuracy": float((p == l).mean()), "CHD": float(dists.mean())}
    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sam-results-dir", default="out/sam_eval_top200_matador_auto_vit_b")
    p.add_argument("--taxonomy", default="taxonomy/assets/minc-taxonomy.json")
    p.add_argument("--out-dir", default="runs/sam_classification_eval")
    p.add_argument("--iou-threshold", type=float, default=0.25,
                   help="Min IoU for SAM match to be included (0=all)")
    p.add_argument("--mask-crop", action="store_true", default=True)
    p.add_argument("--no-mask-crop", dest="mask_crop", action="store_false")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-type", default="vit_b")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo = repo_root
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sam_dir = repo / args.sam_results_dir
    cache_dir = sam_dir / "cache" / "auto_masks"
    csv_path = sam_dir / "per_segment_metrics.csv"

    print(f"Loading SAM results from {sam_dir}")
    print(f"IoU threshold: {args.iou_threshold}, mask_crop: {args.mask_crop}")

    print("Building distance matrix...")
    D = build_distance_matrix(str(repo / args.taxonomy))

    # Load per-segment SAM results
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} segments from SAM results")

    # Filter by IoU threshold
    if args.iou_threshold > 0:
        rows = [r for r in rows if float(r["best_iou"]) >= args.iou_threshold]
    print(f"After IoU≥{args.iou_threshold} filter: {len(rows)} segments")

    # Discover flat ResNet models
    model_specs = []
    for name, base in [
        ("flat", "runs/minc_flat/20260603_051833"),
        ("hier", "runs/minc_hier_smooth/20260603_063143"),
    ]:
        run = repo / base
        if (run / "checkpoint_best.pt").exists():
            model_specs.append((name, run))

    for name, base in [("maskedaug", "runs/minc_masked_crop")]:
        candidates = sorted((repo / base).glob("*/"), reverse=True)
        for c in candidates:
            if (c / "checkpoint_best.pt").exists():
                model_specs.append((name, c))
                break

    # Discover HGNN runs
    hgnn_specs = []
    for name, base in [
        ("hgnn", "runs/minc_hgnn/20260603_092440"),
        ("hgnn_maskedaug", "runs/minc_hgnn_maskedaug/20260604_053406"),
    ]:
        run = repo / base
        if (run / "checkpoint_best.pt").exists() and (run / "config.json").exists():
            hgnn_specs.append((name, run))

    print(f"\nFlat models: {[n for n, _ in model_specs]}")
    print(f"HGNN models: {[n for n, _ in hgnn_specs]}")

    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Cache loaded photos
    photo_cache = {}

    def get_photo(photo_path_str):
        if photo_path_str not in photo_cache:
            if len(photo_cache) > 5:
                photo_cache.clear()
            photo_cache[photo_path_str] = np.array(
                Image.open(repo / photo_path_str).convert("RGB")
            )
        return photo_cache[photo_path_str]

    results = {}
    for model_name, run_dir in model_specs:
        print(f"\n[{model_name}] Loading model...")
        model = load_model(run_dir, device)

        all_preds, all_labels = [], []
        skipped = 0

        # Group by photo for cache efficiency
        from collections import defaultdict
        by_photo = defaultdict(list)
        for row in rows:
            by_photo[row["photo_id"]].append(row)

        for photo_id, photo_rows in by_photo.items():
            photo_np = get_photo(photo_rows[0]["photo_path"])
            H, W = photo_np.shape[:2]

            # Load cached SAM masks for this photo
            cached = load_cached_masks(cache_dir, photo_id, args.model_type, (H, W))
            if cached is None:
                skipped += len(photo_rows)
                continue

            # Build index: mask_index -> segmentation array
            mask_index = {}
            for m in cached:
                idx = m.get("mask_index", cached.index(m))
                seg = m.get("segmentation")
                if seg is not None:
                    mask_index[idx] = np.array(seg, dtype=bool)

            batch_imgs, batch_labels = [], []
            for row in photo_rows:
                best_idx = int(row["best_mask_index"])
                label_idx = int(row["label_index"])

                if best_idx not in mask_index:
                    skipped += 1
                    continue

                sam_mask = mask_index[best_idx]
                # SAM mask may be at different resolution than photo
                if sam_mask.shape != (H, W):
                    from PIL import Image as PILImage
                    sam_mask = np.array(
                        PILImage.fromarray(sam_mask.astype(np.uint8) * 255).resize(
                            (W, H), PILImage.NEAREST
                        )
                    ) > 127

                crop = crop_sam_mask(photo_np, sam_mask, args.mask_crop, args.image_size)
                if crop is None:
                    skipped += 1
                    continue

                batch_imgs.append(transform(crop))
                batch_labels.append(label_idx)

            if not batch_imgs:
                continue

            batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                preds = model(batch).argmax(1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(batch_labels)

        del model

        preds_np = np.array(all_preds)
        labels_np = np.array(all_labels)
        print(f"  Evaluated: {len(preds_np)}, skipped: {skipped}")
        if len(preds_np) == 0:
            continue

        m = compute_metrics(preds_np, labels_np, D)
        pc = per_class_metrics(preds_np, labels_np, D)
        results[model_name] = {"metrics": m, "per_class": pc}
        print(f"  Accuracy={m['accuracy']:.4f}  CHD={m['CHD']:.4f}  Hier@d2={m['Hier@d2']:.4f}")

    # HGNN evaluation
    for model_name, run_dir in hgnn_specs:
        print(f"\n[{model_name}] Loading HGNN...")
        hgnn_model, leaf_indices, leaf_to_cat = load_hgnn(
            run_dir, str(repo / args.taxonomy), device
        )

        all_preds, all_labels = [], []
        skipped = 0

        from collections import defaultdict as _dd
        by_photo2 = _dd(list)
        for row in rows:
            by_photo2[row["photo_id"]].append(row)

        for photo_id, photo_rows in by_photo2.items():
            photo_np = get_photo(photo_rows[0]["photo_path"])
            H, W = photo_np.shape[:2]
            cached = load_cached_masks(cache_dir, photo_id, args.model_type, (H, W))
            if cached is None:
                skipped += len(photo_rows)
                continue
            mask_index = {m.get("mask_index", cached.index(m)): np.array(m["segmentation"], dtype=bool)
                          for m in cached if m.get("segmentation") is not None}

            batch_imgs, batch_labels = [], []
            for row in photo_rows:
                best_idx = int(row["best_mask_index"])
                label_idx = int(row["label_index"])
                if best_idx not in mask_index:
                    skipped += 1
                    continue
                sam_mask = mask_index[best_idx]
                if sam_mask.shape != (H, W):
                    from PIL import Image as PILImage
                    sam_mask = np.array(PILImage.fromarray(sam_mask.astype(np.uint8) * 255).resize(
                        (W, H), PILImage.NEAREST)) > 127
                crop = crop_sam_mask(photo_np, sam_mask, args.mask_crop, args.image_size)
                if crop is None:
                    skipped += 1
                    continue
                batch_imgs.append(transform(crop))
                batch_labels.append(label_idx)

            if not batch_imgs:
                continue
            batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                logits = hgnn_model(batch)
                if logits.dim() == 1:  # batch_size=1 causes squeeze() to drop batch dim
                    logits = logits.unsqueeze(0)
                leaf_logits = logits[:, leaf_indices]
                pred_leaf_pos = leaf_logits.argmax(1)
                preds = leaf_to_cat[pred_leaf_pos].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_labels)

        del hgnn_model

        preds_np, labels_np = np.array(all_preds), np.array(all_labels)
        print(f"  Evaluated: {len(preds_np)}, skipped: {skipped}")
        if len(preds_np) == 0:
            continue
        m = compute_metrics(preds_np, labels_np, D)
        pc = per_class_metrics(preds_np, labels_np, D)
        results[model_name] = {"metrics": m, "per_class": pc}
        print(f"  Accuracy={m['accuracy']:.4f}  CHD={m['CHD']:.4f}  Hier@d2={m['Hier@d2']:.4f}")

    # Save results
    out_label = f"masked" if args.mask_crop else "bbox"
    out_file = out_dir / f"results_iou{args.iou_threshold:.2f}_{out_label}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Summary table
    print(f"\n=== SAM + Classifier End-to-End Accuracy ===")
    print(f"IoU threshold: {args.iou_threshold}, mask_crop: {args.mask_crop}")
    print(f"{'Model':<15} {'Accuracy':>10} {'CHD':>8} {'Hier@d2':>10} {'N':>6}")
    print("-" * 52)
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<15} {m['accuracy']:>10.4f} {m['CHD']:>8.4f} {m['Hier@d2']:>10.4f} {m['n']:>6}")

    print(f"\nNote: SAM auto recall@{args.iou_threshold:.2f} = "
          f"{sum(1 for r in rows)/len(list(open(csv_path)))*100:.0f}%")
    print("End-to-end accuracy = SAM recall × classifier accuracy (approximate)")

    # Per-class breakdown for flat model
    if "flat" in results:
        print(f"\n=== Per-class (flat model) ===")
        print(f"{'Category':<16} {'N':>5} {'Acc':>8} {'SAM IoU':>9}")
        print("-" * 42)

        # Load SAM per-class IoU from aggregate metrics
        agg = json.loads((sam_dir / "aggregate_metrics.json").read_text())
        sam_pc = agg["metrics"].get("per_class", {})

        for cat in sorted(results["flat"]["per_class"].keys()):
            pc = results["flat"]["per_class"][cat]
            sam_iou = sam_pc.get(cat, {}).get("mean_best_iou", float("nan"))
            print(f"{cat:<16} {pc['n']:>5} {pc['accuracy']:>8.3f} {sam_iou:>9.3f}")


if __name__ == "__main__":
    main()
