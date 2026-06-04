#!/usr/bin/env python3
"""
Evaluate material classifiers on MINC-S scene segments.

Two evaluation protocols:

  Section 1 — SAM-matched segments (IoU ≥ 0.5):
    Only GT segments for which SAM produced a mask with IoU ≥ 0.5 are included
    (751 / 1,067 GT segments).  Each classifier receives the SAM crop with
    non-mask pixels filled to ImageNet mean (masked crop format).

  Section 2 — All GT segments:
    All 6,917 MINC-S GT segment masks, two crop modes:
      masked  — non-segment pixels → ImageNet mean
      bbox    — full bounding box, no masking

Models evaluated (one or more, via --models):
  flat      — ResNet50 flat CE baseline
  hierloss  — ResNet50 with hierarchical greedy_loss (38-node head)
  maskdrop  — ResNet50 trained with MaskDropAugment
  hgnn      — HGNN with taxonomy GNN + greedy_loss (main model)

Metrics reported: Accuracy, CHD (Confusional Hierarchy Distance), Hier@d2.

Usage:
    # Section 1 only (SAM-matched)
    python scripts/eval_classifiers.py --section 1

    # Both sections, all models
    python scripts/eval_classifiers.py --section all

    # Specific models / SAM dir
    python scripts/eval_classifiers.py --models flat hgnn \\
        --sam-dir out/sam_eval_all1654_auto_vit_b
"""

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.minc import MINC2500Dataset
from gnn_classifier.hgnn import HGNN
from taxonomy.tree import get_taxonomy

CATEGORIES    = MINC2500Dataset.CATEGORIES
NUM_CLASSES   = len(CATEGORIES)                     # 23
IMAGENET_MEAN = (123, 116, 103)
ALL_MODELS    = ["flat", "hierloss", "maskdrop", "hgnn"]


# ── Tree-distance matrix ──────────────────────────────────────────────────────

def build_tree_distances() -> np.ndarray:
    tax_path = repo_root / "taxonomy/assets/minc-taxonomy.json"
    g  = get_taxonomy(str(tax_path)).to_undirected()
    dist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    for i, ci in enumerate(CATEGORIES):
        if ci not in g:
            continue
        lengths = nx.single_source_shortest_path_length(g, ci)
        for j, cj in enumerate(CATEGORIES):
            dist[i, j] = lengths.get(cj, 0)
    return dist

TREE_DIST = build_tree_distances()


# ── Crop helpers ──────────────────────────────────────────────────────────────

def crop_masked(photo: np.ndarray, mask: np.ndarray) -> Image.Image | None:
    """Bounding box crop with non-mask pixels → ImageNet mean."""
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


def crop_bbox(photo: np.ndarray, mask: np.ndarray) -> Image.Image | None:
    """Bounding box crop, no masking."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return Image.fromarray(photo[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1])


def load_mask(mask_path: str, H: int, W: int) -> np.ndarray:
    p = Path(mask_path)
    if not p.is_absolute():
        p = repo_root / p
    m = np.array(Image.open(p))
    if m.shape != (H, W):
        m = np.array(Image.fromarray((m.astype(np.uint8) * 255))
                       .resize((W, H), Image.NEAREST)) > 127
    return m.astype(bool)


# ── Model loaders ─────────────────────────────────────────────────────────────

def _latest_ckpt(run_dir: Path) -> Path:
    runs = sorted(run_dir.glob("*/checkpoint_best.pt"))
    if not runs:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return runs[-1]


def load_flat_model(run_dir: Path, device: torch.device):
    """Load flat or maskdrop ResNet50 (timm model, direct argmax → CATEGORIES index)."""
    ckpt_path = _latest_ckpt(run_dir)
    cfg_path  = ckpt_path.parent / "config.json"
    backbone  = json.load(open(cfg_path)).get("backbone", "resnet50") if cfg_path.exists() else "resnet50"
    model = timm.create_model(backbone, pretrained=False, num_classes=NUM_CLASSES)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), "flat"


def load_hierloss_model(run_dir: Path, device: torch.device):
    """
    Load Flat+HierLoss model (38-node head, CE training on leaf subset).

    CE training on leaf_logits[k] vs label k forces the mapping:
      argmax(logits[:, leaf_indices]) → CATEGORIES index directly.
    """
    ckpt_path = _latest_ckpt(run_dir)
    cfg       = json.load(open(ckpt_path.parent / "config.json"))
    node_to_idx_path = ckpt_path.parent / "node_index.json"
    node_to_idx = json.load(open(node_to_idx_path))["node_to_idx"]

    backbone = cfg.get("backbone", "resnet50")
    graph     = get_taxonomy(str(repo_root / "taxonomy/assets/minc-taxonomy.json"))
    node_order = list(nx.topological_sort(graph))
    num_nodes  = len(node_order)
    base = timm.create_model(backbone, pretrained=False, num_classes=0)
    model = nn.Sequential(base, nn.Linear(base.num_features, num_nodes))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    leaves      = [n for n in graph.nodes if graph.out_degree(n) == 0]
    leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                dtype=torch.long, device=device)
    return model, "hierloss", leaf_indices


def load_hgnn_model(run_dir: Path, device: torch.device):
    """
    Load HGNN (greedy_loss trained).

    greedy_loss does NOT enforce argmax-position == CATEGORIES-index, so we
    need leaf_to_cat to map: argmax(logits[:, leaf_indices]) → CATEGORIES index.
    """
    ckpt_path = _latest_ckpt(run_dir)
    cfg       = json.load(open(ckpt_path.parent / "config.json"))
    ckpt      = torch.load(ckpt_path, map_location=device)
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
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    leaf_to_cat = torch.zeros(len(leaf_indices), dtype=torch.long, device=device)
    for pos, idx in enumerate(leaf_indices.tolist()):
        leaf_to_cat[pos] = CATEGORIES.index(idx_to_node[idx])

    return model, "hgnn", leaf_indices, leaf_to_cat


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, model_type: str, img_tensor: torch.Tensor,
            leaf_indices=None, leaf_to_cat=None) -> int:
    """Return CATEGORIES index prediction for a single image tensor."""
    logits = model(img_tensor.unsqueeze(0))
    if model_type in ("hgnn", "hierloss"):
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        leaf_logits = logits[:, leaf_indices]
        pos = int(leaf_logits.argmax(1).item())
        if leaf_to_cat is not None:           # hgnn (greedy_loss)
            return int(leaf_to_cat[pos].item())
        return pos                             # hierloss (CE → direct correspondence)
    return int(logits.argmax(1).item())        # flat / maskdrop


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> dict:
    n = len(preds)
    acc     = float((preds == gts).mean())
    chd     = float(np.mean([TREE_DIST[gts[i], preds[i]] for i in range(n)]))
    hier_d2 = float(np.mean([TREE_DIST[gts[i], preds[i]] <= 2 for i in range(n)]))

    # Per-class accuracy for bar/radar plots
    per_class = {}
    for ci, name in enumerate(CATEGORIES):
        mask = gts == ci
        if mask.sum() == 0:
            continue
        per_class[name] = {"acc": float((preds[mask] == ci).mean()), "n": int(mask.sum())}

    return {"accuracy": acc, "CHD": chd, "Hier@d2": hier_d2, "n": n,
            "per_class": per_class}


def print_table(results: dict, title: str):
    print(f"\n{title}")
    print(f"{'Model':<16} {'Accuracy':>10} {'CHD':>8} {'Hier@d2':>10} {'N':>6}")
    print("-" * 54)
    for name, m in results.items():
        print(f"{name:<16} {m['accuracy']:>10.4f} {m['CHD']:>8.4f} "
              f"{m['Hier@d2']:>10.4f} {m['n']:>6}")


# ── Evaluation runners ────────────────────────────────────────────────────────

def run_section1(model_specs: list, sam_dir: Path, iou_threshold: float,
                 transform: T.Compose, device: torch.device) -> dict:
    """Section 1: SAM-matched GT segments (masked crop)."""
    df = pd.read_csv(sam_dir / "per_segment_metrics.csv")
    df = df[df["best_iou"] >= iou_threshold].copy()
    print(f"\nSection 1: {len(df)} segments (SAM IoU ≥ {iou_threshold}, masked crop)")

    results = {}
    for spec in model_specs:
        name, model, model_type = spec[0], spec[1], spec[2]
        leaf_indices = spec[3] if len(spec) > 3 else None
        leaf_to_cat  = spec[4] if len(spec) > 4 else None

        preds, gts, skipped = [], [], 0
        for _, row in df.iterrows():
            ppath = Path(row["photo_path"])
            if not ppath.is_absolute():
                ppath = repo_root / ppath
            if not ppath.exists():
                skipped += 1; continue

            photo_np = np.array(Image.open(ppath).convert("RGB"))
            mask     = load_mask(row["mask_path"], *photo_np.shape[:2])
            crop     = crop_masked(photo_np, mask)
            if crop is None:
                skipped += 1; continue

            img = transform(crop).to(device)
            preds.append(predict(model, model_type, img, leaf_indices, leaf_to_cat))
            gts.append(int(row["label_index"]))

        m = compute_metrics(np.array(preds), np.array(gts))
        results[name] = m
        print(f"  {name}: acc={m['accuracy']:.4f}  CHD={m['CHD']:.4f}  "
              f"Hier@d2={m['Hier@d2']:.4f}  (n={m['n']}, skipped={skipped})")
    return results


def run_section2(model_specs: list, gt_csv: Path, transform: T.Compose,
                 device: torch.device) -> dict:
    """Section 2: All GT segments, masked and bbox crop modes."""
    df = pd.read_csv(gt_csv)
    print(f"\nSection 2: {len(df)} GT segments (all, masked + bbox crop)")

    results_masked = {}
    results_bbox   = {}
    for spec in model_specs:
        name, model, model_type = spec[0], spec[1], spec[2]
        leaf_indices = spec[3] if len(spec) > 3 else None
        leaf_to_cat  = spec[4] if len(spec) > 4 else None

        preds_m, preds_b, gts_all, skipped = [], [], [], 0
        for _, row in df.iterrows():
            ppath = Path(row["photo_path"])
            if not ppath.is_absolute():
                ppath = repo_root / ppath
            if not ppath.exists():
                skipped += 1; continue

            photo_np = np.array(Image.open(ppath).convert("RGB"))
            mask     = load_mask(row["mask_path"], *photo_np.shape[:2])
            cm = crop_masked(photo_np, mask)
            cb = crop_bbox(photo_np, mask)
            if cm is None or cb is None:
                skipped += 1; continue

            img_m = transform(cm).to(device)
            img_b = transform(cb).to(device)
            preds_m.append(predict(model, model_type, img_m, leaf_indices, leaf_to_cat))
            preds_b.append(predict(model, model_type, img_b, leaf_indices, leaf_to_cat))
            gts_all.append(int(row["label_index"]))

        gts = np.array(gts_all)
        results_masked[name] = compute_metrics(np.array(preds_m), gts)
        results_bbox[name]   = compute_metrics(np.array(preds_b), gts)
        print(f"  {name}: masked acc={results_masked[name]['accuracy']:.4f}  "
              f"bbox acc={results_bbox[name]['accuracy']:.4f}  "
              f"(n={len(gts)}, skipped={skipped})")

    return {"masked": results_masked, "bbox": results_bbox}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    p.add_argument("--section", choices=["1", "2", "all"], default="all")
    p.add_argument("--sam-dir",         default="out/sam_eval_all1654_auto_vit_b")
    p.add_argument("--iou-threshold",   type=float, default=0.5)
    p.add_argument("--gt-csv",          default=None,
                   help="CSV of all GT segments for Section 2 "
                        "(default: <sam-dir>/per_segment_metrics.csv with all rows)")
    p.add_argument("--flat-run",        default="runs/minc_flat")
    p.add_argument("--hierloss-run",    default="runs/minc_hierloss")
    p.add_argument("--maskdrop-run",    default="runs/minc_maskdrop")
    p.add_argument("--hgnn-run",        default="runs/minc_hgnn")
    p.add_argument("--out",             default="runs/eval_classifiers.json")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Load requested models ─────────────────────────────────────────────────
    model_specs = []
    run_dirs = {
        "flat":     (repo_root / args.flat_run,     "flat"),
        "hierloss": (repo_root / args.hierloss_run, "hierloss"),
        "maskdrop": (repo_root / args.maskdrop_run, "flat"),
        "hgnn":     (repo_root / args.hgnn_run,     "hgnn"),
    }

    for name in args.models:
        run_dir, mtype = run_dirs[name]
        if not run_dir.exists():
            print(f"  [{name}] run dir not found — skipping ({run_dir})")
            continue
        try:
            if mtype == "flat":
                model, _ = load_flat_model(run_dir, device)
                model_specs.append((name, model, "flat"))
            elif mtype == "hierloss":
                model, _, leaf_indices = load_hierloss_model(run_dir, device)
                model_specs.append((name, model, "hierloss", leaf_indices))
            elif mtype == "hgnn":
                model, _, leaf_indices, leaf_to_cat = load_hgnn_model(run_dir, device)
                model_specs.append((name, model, "hgnn", leaf_indices, leaf_to_cat))
            print(f"  Loaded: {name}")
        except Exception as e:
            print(f"  [{name}] failed to load — skipping ({e})")

    if not model_specs:
        print("No models loaded. Check run directories.")
        return

    # ── Run evaluations ───────────────────────────────────────────────────────
    sam_dir = repo_root / args.sam_dir
    gt_csv  = Path(args.gt_csv) if args.gt_csv else sam_dir / "per_segment_metrics.csv"
    all_results = {}

    if args.section in ("1", "all"):
        s1 = run_section1(model_specs, sam_dir, args.iou_threshold, transform, device)
        all_results["section1_sam_matched"] = s1
        print_table(s1, "Section 1 — SAM-matched segments (masked crop)")

    if args.section in ("2", "all"):
        s2 = run_section2(model_specs, gt_csv, transform, device)
        all_results["section2_gt_masked"] = s2["masked"]
        all_results["section2_gt_bbox"]   = s2["bbox"]
        print_table(s2["masked"], "Section 2 — All GT segments (masked crop)")
        print_table(s2["bbox"],   "Section 2 — All GT segments (bbox crop)")

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
