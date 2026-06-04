#!/usr/bin/env python3
"""
Evaluate trained HGNN patch classifier on MINC-S segmentation benchmark.

Runs the same protocol as eval_minc_s_patch_classifiers.py but handles the
HGNN's graph-output format: logits over all taxonomy nodes, leaf argmax.

Usage:
    python scripts/eval_minc_s_hgnn.py \
        --hgnn-run runs/minc_hgnn/20260603_092440 \
        --flat-run runs/minc_flat/20260603_051833 \
        --hier-run runs/minc_hier_smooth/20260603_063143
"""

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from gnn_classifier.hgnn import HGNN
from taxonomy.tree import get_taxonomy, get_hierarchy_levels

CATEGORIES = [
    "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass",
    "hair", "leather", "metal", "mirror", "other", "painted", "paper",
    "plastic", "polishedstone", "skin", "sky", "stone", "tile",
    "wallpaper", "water", "wood",
]
NUM_CLASSES = len(CATEGORIES)


def build_distance_matrix(taxonomy_path: str) -> np.ndarray:
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
                D[i, j] = 10.0
    return D


def load_flat_model(run_dir: Path, device: torch.device) -> nn.Module:
    model = timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES)
    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_hgnn_model(run_dir: Path, taxonomy_path: str, device: torch.device):
    """Load HGNN and return (model, leaf_indices, leaf_to_cat_idx)."""
    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    node_to_idx = ckpt["node_to_idx"]

    g = get_taxonomy(taxonomy_path)
    node_order = list(nx.topological_sort(g))
    g_canon = nx.DiGraph()
    for n in node_order:
        g_canon.add_node(n, **g.nodes[n])
    for u, v in g.edges:
        g_canon.add_edge(u, v, **g.edges[u, v])

    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    model = HGNN(
        graph=g_canon,
        cnn_kwargs=dict(
            backbone=cfg["backbone"],
            pretrained=False,
            output_dim=cfg["cnn_output_dim"],
            finetune=False,
        ),
        gnn_kwargs=dict(
            input_dim=cfg["cnn_output_dim"],
            hidden_dim=cfg["gnn_hidden_dim"],
            output_dim=cfg["gnn_output_dim"],
            num_layers=cfg["gnn_layers"],
            num_heads=1,
            skip_connection=True,
            dropout=cfg["dropout"],
        ),
        dropout_prob=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    leaves = [n for n in g_canon.nodes if g_canon.out_degree(n) == 0]
    leaf_indices = torch.tensor(
        sorted([node_to_idx[n] for n in leaves]), dtype=torch.long, device=device
    )
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    # Map leaf position → CATEGORIES index
    leaf_to_cat = torch.zeros(len(leaf_indices), dtype=torch.long, device=device)
    for pos, idx in enumerate(leaf_indices.tolist()):
        cat_name = idx_to_node[idx]
        leaf_to_cat[pos] = CATEGORIES.index(cat_name)

    return model, leaf_indices, leaf_to_cat


def stream_segment_crops(entries, photos_dir, segments_dir, mask_crop=True):
    """Yield (tensor_list_ready_to_stack, label_idx) per segment. One photo in RAM at a time."""
    indexed_sorted = sorted(enumerate(entries), key=lambda x: x[1][1])
    photo_cache = {}
    missing = 0

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
            yield crop, label_idx
        except Exception as e:
            missing += 1
            photo_cache = {}

    print(f"  Streamed segments, missing/errors: {missing}")


def stream_eval_all(flat_models, hgnn_model, leaf_indices, leaf_to_cat,
                    entries, photos_dir, segments_dir,
                    transform, device, mask_crop=True, batch_size=64):
    """Stream through photos once, run all models simultaneously."""
    flat_names = list(flat_models.keys())
    all_preds = {n: [] for n in flat_names}
    all_labels = {n: [] for n in flat_names}
    hgnn_preds, hgnn_labels = [], []

    batches = {n: [] for n in flat_names}
    lbls = {n: [] for n in flat_names}
    h_batch, h_lbls = [], []

    def flush():
        for name, model in flat_models.items():
            if not batches[name]:
                continue
            b = torch.stack(batches[name]).to(device)
            with torch.no_grad():
                p = model(b).argmax(1).cpu().tolist()
            all_preds[name].extend(p)
            all_labels[name].extend(lbls[name])
            batches[name].clear()
            lbls[name].clear()
        if h_batch:
            b = torch.stack(h_batch).to(device)
            with torch.no_grad():
                logits = hgnn_model(b)
                lp = logits[:, leaf_indices].argmax(1)
                p = leaf_to_cat[lp].cpu().tolist()
            hgnn_preds.extend(p)
            hgnn_labels.extend(h_lbls)
            h_batch.clear()
            h_lbls.clear()

    ok = 0
    for crop, lbl in stream_segment_crops(entries, photos_dir, segments_dir, mask_crop):
        t = transform(crop)
        for name in flat_names:
            batches[name].append(t)
            lbls[name].append(lbl)
        h_batch.append(t)
        h_lbls.append(lbl)
        ok += 1
        if ok % batch_size == 0:
            flush()

    flush()
    print(f"  Evaluated {ok} segments")
    results = {n: (np.array(all_preds[n]), np.array(all_labels[n])) for n in flat_names}
    results["hgnn"] = (np.array(hgnn_preds), np.array(hgnn_labels))
    return results


def compute_metrics(preds, labels, D):
    correct = preds == labels
    dists = D[labels, preds]
    return {
        "accuracy": float(correct.mean()),
        "CHD": float(dists.mean()),
        "Hier@d2": float((dists <= 2).mean()),
        "n": int(len(preds)),
    }


def per_class_metrics(preds, labels, D):
    out = {}
    for ci, cat in enumerate(CATEGORIES):
        mask = labels == ci
        if not mask.any():
            continue
        p, l = preds[mask], labels[mask]
        out[cat] = {
            "n": int(mask.sum()),
            "accuracy": float((p == l).mean()),
            "CHD": float(D[l, p].mean()),
        }
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hgnn-run", default="runs/minc_hgnn/20260603_092440")
    p.add_argument("--flat-run", default="runs/minc_flat/20260603_051833")
    p.add_argument("--hier-run", default="runs/minc_hier_smooth/20260603_063143")
    p.add_argument("--dataset-root", default="data/external/minc/minc-s")
    p.add_argument("--taxonomy", default="taxonomy/assets/minc-taxonomy.json")
    p.add_argument("--out-dir", default="runs/minc_s_hgnn_eval")
    p.add_argument("--mask-crop", action="store_true", default=True)
    p.add_argument("--no-mask-crop", dest="mask_crop", action="store_false")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-segments", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    repo = repo_root
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = repo / args.dataset_root
    photos_dir = dataset_root / "photos"
    segments_dir = dataset_root / "segments"
    segments_txt = dataset_root / "test-segments.txt"

    print("Loading taxonomy distances...")
    D = build_distance_matrix(str(repo / args.taxonomy))

    print("Loading MINC-S segments...")
    entries = []
    with open(segments_txt) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                entries.append((int(parts[0]), parts[1], parts[2]))
    if args.max_segments:
        entries = entries[:args.max_segments]
    print(f"  {len(entries)} segments")

    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load all models, then stream photos once for all models simultaneously
    print("\nLoading flat models...")
    flat_models = {}
    for name, run_str in [("flat", args.flat_run), ("hier", args.hier_run)]:
        flat_models[name] = load_flat_model(repo / run_str, device)
        print(f"  Loaded: {name}")

    print("Loading HGNN...")
    hgnn_model, leaf_indices, leaf_to_cat = load_hgnn_model(
        repo / args.hgnn_run, str(repo / args.taxonomy), device
    )

    mask_label = "masked" if args.mask_crop else "bbox_only"
    print(f"\nStreaming eval ({mask_label}, all models simultaneously)...")
    stream_results = stream_eval_all(
        flat_models, hgnn_model, leaf_indices, leaf_to_cat,
        entries, photos_dir, segments_dir,
        transform, device, mask_crop=args.mask_crop,
    )

    results = {}
    for name, (preds, labels) in stream_results.items():
        m = compute_metrics(preds, labels, D)
        results[name] = {"metrics": m, "per_class": per_class_metrics(preds, labels, D)}
        print(f"  {name}: n={m['n']}  Acc={m['accuracy']:.4f}  CHD={m['CHD']:.4f}  Hier@d2={m['Hier@d2']:.4f}")

    with open(out_dir / f"results_{mask_label}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== MINC-S HGNN Eval ({mask_label}) ===")
    print(f"{'Model':<8} {'Accuracy':>10} {'CHD':>8} {'Hier@d2':>10}")
    print("-" * 40)
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<8} {m['accuracy']:>10.4f} {m['CHD']:>8.4f} {m['Hier@d2']:>10.4f}")

    if "flat" in results and "hgnn" in results:
        flat_m = results["flat"]["metrics"]
        hgnn_m = results["hgnn"]["metrics"]
        print(f"\nHGNN vs flat: acc Δ={hgnn_m['accuracy']-flat_m['accuracy']:+.4f}, "
              f"CHD Δ={hgnn_m['CHD']-flat_m['CHD']:+.4f}, "
              f"Hier@d2 Δ={hgnn_m['Hier@d2']-flat_m['Hier@d2']:+.4f}")


if __name__ == "__main__":
    main()
