#!/usr/bin/env python3
"""
Run all five HGNN ablation studies on MINC-S GT segments.

Ablation 1 — Graph vs parameters:
  Compare hgnn_ce (HGNN true taxonomy, CE loss) vs mlp_head (ResNet50 + 4-layer
  MLP, ~same non-CNN params).  Tests whether the performance gain comes from the
  graph structure itself, not just extra parameters.

Ablation 2 — Graph topology:
  Compare hgnn_ce (true taxonomy tree) vs random_tree (random spanning tree,
  same nodes) vs full_graph (all-to-all edges).  Tests whether the specific
  semantic hierarchy matters, or just the GNN mechanism.

Ablation 3 — Cross-parent error reduction:
  Group MINC-S predictions by parent taxonomy node and compare within-parent
  accuracy and cross-parent error rates between Flat ResNet50 and HGNN.

Ablation 4 — Context removal:
  Evaluate both Flat and HGNN at five mask-drop levels (0–100%) to test whether
  HGNN's advantage grows as scene context is removed.

Ablation 5 — Bootstrap confidence intervals:
  1,000-sample bootstrap over 751 MINC-S GT segments to compute 95% CIs for the
  HGNN vs Flat accuracy gain and CHD reduction.

Usage:
    python scripts/eval_ablations.py              # all ablations
    python scripts/eval_ablations.py --ablation 1
    python scripts/eval_ablations.py --ablation 3 4 5
"""

import argparse
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.minc import MINC2500Dataset
from gnn_classifier.hgnn import HGNN, ImageEncoder
from taxonomy.tree import get_taxonomy

CATEGORIES  = MINC2500Dataset.CATEGORIES
NUM_CLASSES = len(CATEGORIES)
IMAGENET_MEAN = (123, 116, 103)


# ── Taxonomy helpers ──────────────────────────────────────────────────────────

def build_tree_distances() -> np.ndarray:
    g = get_taxonomy(str(repo_root / "taxonomy/assets/minc-taxonomy.json")).to_undirected()
    dist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    for i, ci in enumerate(CATEGORIES):
        if ci not in g:
            continue
        lengths = nx.single_source_shortest_path_length(g, ci)
        for j, cj in enumerate(CATEGORIES):
            dist[i, j] = lengths.get(cj, 0)
    return dist

TREE_DIST = build_tree_distances()

# Parent groupings for Ablation 3
PARENT_GROUPS = {
    "masonry":       ["brick", "stone", "tile", "ceramic"],
    "vitreous":      ["glass", "mirror"],
    "textile":       ["carpet", "fabric", "wallpaper"],
    "vegetation":    ["foliage"],
    "wood_derived":  ["wood", "paper"],
    "metal":         ["metal", "polishedstone"],
    "synthetic":     ["plastic", "painted"],
    "organic":       ["food"],
    "animal_derived":["leather", "skin", "hair"],
    "natural":       ["sky", "water"],
    "other":         ["other"],
}


# ── Model definitions (ablation variants) ────────────────────────────────────

class MLPClassifier(nn.Module):
    def __init__(self, num_classes=23, backbone="resnet50", pretrained=False, dropout=0.1):
        super().__init__()
        self.encoder = ImageEncoder(output_dim=512, backbone=backbone,
                                    pretrained=pretrained, finetune=False)
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ── Model loading ─────────────────────────────────────────────────────────────

def _latest_ckpt(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("*/checkpoint_best.pt"))
    if ckpts:
        return ckpts[-1]
    direct = run_dir / "checkpoint_best.pt"
    if direct.exists():
        return direct
    raise FileNotFoundError(f"No checkpoint in {run_dir}")


def load_hgnn_main(run_dir: Path, device: torch.device):
    """Load the main HGNN (greedy_loss trained). Returns (model, leaf_indices, leaf_to_cat)."""
    ckpt_path   = _latest_ckpt(run_dir)
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


def load_flat_model(run_dir: Path, device: torch.device):
    """Load flat ResNet50 baseline. Returns (model,)."""
    import timm
    ckpt_path = _latest_ckpt(run_dir)
    cfg_path  = ckpt_path.parent / "config.json"
    backbone  = json.load(open(cfg_path)).get("backbone", "resnet50") if cfg_path.exists() else "resnet50"
    model = timm.create_model(backbone, pretrained=False, num_classes=NUM_CLASSES)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def load_ablation_variant(run_dir: Path, device: torch.device):
    """
    Load a graph ablation variant (CE-trained HGNN or MLP).

    For CE-trained HGNNs: argmax(logits[:, leaf_indices]) == CATEGORIES index directly.
    For MLP: argmax(logits) == CATEGORIES index directly.
    Returns (model, is_hgnn, leaf_indices).
    """
    from scripts.train_ablation_variants import (
        MLPClassifier, MLPMatchedClassifier, make_random_tree, make_full_graph, canonicalize
    )
    ckpt_path = run_dir / "checkpoint_best.pt"
    cfg       = json.load(open(run_dir / "config.json"))
    ckpt      = torch.load(ckpt_path, map_location=device)
    variant   = cfg["variant"]

    if variant == "mlp_head":
        model = MLPClassifier(num_classes=23, backbone=cfg["backbone"],
                              pretrained=False, dropout=cfg["dropout"])
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device).eval(), False, None

    if variant == "mlp_matched":
        # 38-logit MLP; extract leaf logits exactly like CE-trained HGNNs
        tax_path   = repo_root / "taxonomy/assets/minc-taxonomy.json"
        g_raw      = get_taxonomy(str(tax_path))
        node_to_idx = ckpt["node_to_idx"]
        leaves      = [n for n in g_raw.nodes if g_raw.out_degree(n) == 0]
        leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                    dtype=torch.long, device=device)
        model = MLPMatchedClassifier(num_nodes=cfg.get("num_nodes", 38),
                                     backbone=cfg["backbone"],
                                     pretrained=False, dropout=cfg["dropout"])
        model.load_state_dict(ckpt["model_state_dict"])
        return model.to(device).eval(), True, leaf_indices

    # HGNN variant — rebuild the exact graph topology used during training
    tax_path = repo_root / "taxonomy/assets/minc-taxonomy.json"
    g_raw    = get_taxonomy(str(tax_path))
    node_order = list(nx.topological_sort(g_raw))
    g_canon  = nx.DiGraph()
    for n in node_order:
        g_canon.add_node(n, **g_raw.nodes[n])
    for u, v in g_raw.edges:
        g_canon.add_edge(u, v, **g_raw.edges[u, v])

    if variant == "random_tree":
        g_use = canonicalize(make_random_tree(list(g_canon.nodes), seed=42))
        for n in g_use.nodes:
            g_use.nodes[n].update(g_canon.nodes.get(n, {}))
    elif variant == "full_graph":
        g_use = make_full_graph(list(g_canon.nodes))
        for n in g_use.nodes:
            g_use.nodes[n].update(g_canon.nodes.get(n, {}))
    else:
        g_use = g_canon

    model = HGNN(
        graph=g_use,
        cnn_kwargs=dict(backbone=cfg["backbone"], pretrained=False,
                        output_dim=cfg["cnn_output_dim"], finetune=False),
        gnn_kwargs=dict(input_dim=cfg["cnn_output_dim"], hidden_dim=cfg["gnn_hidden_dim"],
                        output_dim=cfg["gnn_output_dim"], num_layers=cfg["gnn_layers"],
                        num_heads=1, skip_connection=True, dropout=cfg["dropout"]),
        dropout_prob=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    node_to_idx  = ckpt["node_to_idx"]
    leaves       = [n for n in g_canon.nodes if g_canon.out_degree(n) == 0]
    leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                dtype=torch.long, device=device)
    return model, True, leaf_indices


# ── Crop helpers ──────────────────────────────────────────────────────────────

def load_mask(mask_path: str, H: int, W: int) -> np.ndarray:
    p = Path(mask_path)
    if not p.is_absolute():
        p = repo_root / p
    m = np.array(Image.open(p))
    if m.shape != (H, W):
        m = np.array(Image.fromarray((m.astype(np.uint8) * 255))
                       .resize((W, H), Image.NEAREST)) > 127
    return m.astype(bool)


def crop_with_level(photo: np.ndarray, mask: np.ndarray,
                    context_removed: float) -> Image.Image | None:
    """
    Bounding-box crop where `context_removed` fraction of non-mask pixels are
    set to ImageNet mean (0 = full context, 1 = full masking).
    """
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    crop = photo[r0:r1, c0:c1].copy()
    m    = mask[r0:r1, c0:c1]
    if context_removed > 0:
        outside = ~m
        if context_removed < 1.0:
            # Randomly keep (1-context_removed) of outside pixels
            keep = np.random.rand(*outside.shape) > context_removed
            outside = outside & ~keep
        for ch, mv in enumerate(IMAGENET_MEAN):
            crop[:, :, ch][outside] = mv
    return Image.fromarray(crop)


# ── Predict helpers ───────────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(model, crops: list, transform: T.Compose, device: torch.device,
                  is_hgnn: bool, leaf_indices=None,
                  leaf_to_cat=None) -> list:
    """Classify a list of PIL images; return list of CATEGORIES indices."""
    preds = []
    for crop in crops:
        img    = transform(crop).unsqueeze(0).to(device)
        logits = model(img)
        if is_hgnn:
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            leaf_logits = logits[:, leaf_indices]
            pos = int(leaf_logits.argmax(1).item())
            pred = int(leaf_to_cat[pos].item()) if leaf_to_cat is not None else pos
        else:
            pred = int(logits.argmax(1).item())
        preds.append(pred)
    return preds


def load_segment_data(df: pd.DataFrame) -> tuple:
    """Load photos, masks, and GT labels for a segment DataFrame."""
    photos, masks, gts = [], [], []
    skipped = 0
    for _, row in df.iterrows():
        ppath = Path(row["photo_path"])
        if not ppath.is_absolute():
            ppath = repo_root / ppath
        if not ppath.exists():
            skipped += 1; continue
        photo_np = np.array(Image.open(ppath).convert("RGB"))
        mask     = load_mask(row["mask_path"], *photo_np.shape[:2])
        photos.append(photo_np)
        masks.append(mask)
        gts.append(int(row["label_index"]))
    if skipped:
        print(f"  Warning: skipped {skipped} missing photos")
    return photos, masks, np.array(gts)


def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> dict:
    n = len(preds)
    acc     = float((preds == gts).mean())
    chd     = float(np.mean([TREE_DIST[gts[i], preds[i]] for i in range(n)]))
    hier_d2 = float(np.mean([TREE_DIST[gts[i], preds[i]] <= 2 for i in range(n)]))

    per_class = {}
    for ci, name in enumerate(CATEGORIES):
        mask = gts == ci
        if mask.sum() == 0:
            continue
        per_class[name] = {"acc": float((preds[mask] == ci).mean()), "n": int(mask.sum())}

    return {"accuracy": acc, "CHD": chd, "Hier@d2": hier_d2, "n": n,
            "per_class": per_class}


# ── Ablation runners ──────────────────────────────────────────────────────────

def ablation_1_and_2(args, df: pd.DataFrame, transform: T.Compose,
                     device: torch.device) -> dict:
    """
    Ablations 1 & 2: evaluate mlp_head, hgnn_ce, random_tree, full_graph
    on SAM-matched segments (IoU ≥ threshold, masked crop).
    """
    variants = ["mlp_head", "mlp_matched", "hgnn_ce", "random_tree", "full_graph"]
    abl_dir  = repo_root / args.ablations_dir
    results  = {}

    # Also include the main HGNN for comparison
    hgnn_dir = repo_root / args.hgnn_run
    try:
        model, leaf_indices, leaf_to_cat = load_hgnn_main(hgnn_dir, device)
        photos, masks, gts = load_segment_data(df)
        crops = [c for ph, m in zip(photos, masks)
                 if (c := crop_with_level(ph, m, 1.0)) is not None]
        gts_f = np.array([gt for ph, m, gt in zip(photos, masks, gts)
                          if crop_with_level(ph, m, 1.0) is not None])
        preds = predict_batch(model, crops, transform, device,
                              is_hgnn=True, leaf_indices=leaf_indices, leaf_to_cat=leaf_to_cat)
        results["hgnn_greedy"] = compute_metrics(np.array(preds), gts_f)
        del model; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [hgnn_greedy] skipped: {e}")

    for variant in variants:
        vdir = abl_dir / variant
        if not (vdir / "checkpoint_best.pt").exists() or not (vdir / "config.json").exists():
            print(f"  [{variant}] not trained — skipping")
            continue
        try:
            model, is_hgnn, leaf_indices = load_ablation_variant(vdir, device)
        except Exception as e:
            print(f"  [{variant}] load failed: {e}")
            continue

        photos, masks, gts = load_segment_data(df)
        crops = []
        gts_f = []
        for ph, m, gt in zip(photos, masks, gts):
            c = crop_with_level(ph, m, 1.0)
            if c is not None:
                crops.append(c)
                gts_f.append(gt)

        preds = predict_batch(model, crops, transform, device,
                              is_hgnn=is_hgnn, leaf_indices=leaf_indices)
        results[variant] = compute_metrics(np.array(preds), np.array(gts_f))
        cfg = json.load(open(vdir / "config.json"))
        results[variant]["minc_val_acc"] = cfg.get("best_val_acc", None)
        del model; torch.cuda.empty_cache()
        print(f"  {variant}: {results[variant]}")

    return results


def ablation_3(args, df: pd.DataFrame, transform: T.Compose,
               device: torch.device) -> dict:
    """
    Ablation 3: cross-parent error rates for Flat vs HGNN.

    Groups predictions by parent taxonomy node and reports within-parent
    accuracy and cross-parent error rate for each model.
    """
    # Build per-category parent lookup
    cat_to_parent = {}
    for parent, cats in PARENT_GROUPS.items():
        for c in cats:
            if c in CATEGORIES:
                cat_to_parent[CATEGORIES.index(c)] = parent

    results = {}
    for name, loader in [
        ("flat", lambda: (load_flat_model(repo_root / args.flat_run, device), False, None, None)),
        ("hgnn", lambda: load_hgnn_main(repo_root / args.hgnn_run, device) + (True,)),
    ]:
        try:
            loaded = loader()
            if name == "flat":
                model, is_hgnn, leaf_indices, leaf_to_cat = loaded[0], False, None, None
            else:
                model, leaf_indices, leaf_to_cat = loaded[0], loaded[1], loaded[2]
                is_hgnn = True
        except Exception as e:
            print(f"  [{name}] load failed: {e}"); continue

        photos, masks, gts = load_segment_data(df)
        crops, gts_f = [], []
        for ph, m, gt in zip(photos, masks, gts):
            c = crop_with_level(ph, m, 1.0)
            if c is not None:
                crops.append(c); gts_f.append(gt)

        preds = predict_batch(model, crops, transform, device,
                              is_hgnn=is_hgnn, leaf_indices=leaf_indices,
                              leaf_to_cat=leaf_to_cat)

        gts_arr = np.array(gts_f)
        preds_arr = np.array(preds)
        within, cross, total = 0, 0, 0
        per_parent = {}
        for parent, cats in PARENT_GROUPS.items():
            cat_idxs = [CATEGORIES.index(c) for c in cats if c in CATEGORIES]
            mask_gt  = np.isin(gts_arr, cat_idxs)
            if mask_gt.sum() == 0:
                continue
            p_sub, g_sub = preds_arr[mask_gt], gts_arr[mask_gt]
            in_parent = np.isin(p_sub, cat_idxs)
            per_parent[parent] = {
                "within_acc": float(in_parent.mean()),
                "n": int(mask_gt.sum()),
            }
            within += in_parent.sum()
            cross  += (~in_parent).sum()
            total  += len(p_sub)

        # Full confusion matrix (NUM_CLASSES × NUM_CLASSES) for plotting
        conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for g, p in zip(gts_arr, preds_arr):
            conf_matrix[g, p] += 1

        results[name] = {
            "within_parent_acc": float(within / total),
            "cross_parent_err":  float(cross / total),
            "per_parent": per_parent,
            "confusion_matrix": conf_matrix.tolist(),
            "categories": CATEGORIES,
            "n": total,
        }
        del model; torch.cuda.empty_cache()
        print(f"  {name}: within={results[name]['within_parent_acc']:.4f}  "
              f"cross={results[name]['cross_parent_err']:.4f}")

    return results


def ablation_4(args, df: pd.DataFrame, transform: T.Compose,
               device: torch.device) -> dict:
    """
    Ablation 4: HGNN vs Flat accuracy at five mask-drop levels.

    Tests whether HGNN's advantage grows as scene context is removed,
    which would confirm the GNN's prototype-matching is most valuable
    when external cues are absent.
    """
    levels  = [0.0, 0.25, 0.50, 0.75, 1.0]
    results = {lvl: {} for lvl in levels}

    photos, masks, gts = load_segment_data(df)
    gts = np.array(gts)

    for name, loader, is_hgnn_flag in [
        ("flat", lambda: (load_flat_model(repo_root / args.flat_run, device), None, None), False),
        ("hgnn", lambda: load_hgnn_main(repo_root / args.hgnn_run, device),               True),
    ]:
        try:
            loaded   = loader()
            model    = loaded[0]
            li       = loaded[1] if is_hgnn_flag else None
            l2c      = loaded[2] if is_hgnn_flag else None
        except Exception as e:
            print(f"  [{name}] load failed: {e}"); continue

        for lvl in levels:
            crops, gts_f = [], []
            for ph, m, gt in zip(photos, masks, gts):
                c = crop_with_level(ph, m, lvl)
                if c is not None:
                    crops.append(c); gts_f.append(gt)

            preds = predict_batch(model, crops, transform, device,
                                  is_hgnn=is_hgnn_flag, leaf_indices=li, leaf_to_cat=l2c)
            results[lvl][name] = compute_metrics(np.array(preds), np.array(gts_f))

        del model; torch.cuda.empty_cache()

    # Print table
    print(f"\n  {'Context removed':>20} {'Flat acc':>10} {'HGNN acc':>10} {'Gain':>8}")
    print("  " + "-" * 52)
    for lvl in levels:
        fa = results[lvl].get("flat", {}).get("accuracy", float("nan"))
        ha = results[lvl].get("hgnn", {}).get("accuracy", float("nan"))
        print(f"  {lvl*100:>19.0f}%  {fa:>10.4f} {ha:>10.4f} {ha-fa:>+8.4f}")

    return results


def ablation_5(args, df: pd.DataFrame, transform: T.Compose,
               device: torch.device, n_bootstrap: int = 1000) -> dict:
    """
    Ablation 5: Bootstrap 95% confidence intervals for accuracy gain.

    Resamples the 751 MINC-S GT segments (with replacement) 1,000 times
    and reports the CI for HGNN − Flat accuracy and CHD reduction.
    """
    photos, masks, gts = load_segment_data(df)
    gts = np.array(gts)

    all_preds = {}
    for name, loader, is_hgnn_flag in [
        ("flat", lambda: (load_flat_model(repo_root / args.flat_run, device), None, None), False),
        ("hgnn", lambda: load_hgnn_main(repo_root / args.hgnn_run, device),               True),
    ]:
        try:
            loaded = loader()
            model  = loaded[0]
            li     = loaded[1] if is_hgnn_flag else None
            l2c    = loaded[2] if is_hgnn_flag else None
        except Exception as e:
            print(f"  [{name}] load failed: {e}"); continue

        crops, valid_gts = [], []
        for ph, m, gt in zip(photos, masks, gts):
            c = crop_with_level(ph, m, 1.0)
            if c is not None:
                crops.append(c); valid_gts.append(gt)

        preds = predict_batch(model, crops, transform, device,
                              is_hgnn=is_hgnn_flag, leaf_indices=li, leaf_to_cat=l2c)
        all_preds[name] = np.array(preds)
        del model; torch.cuda.empty_cache()

    if "flat" not in all_preds or "hgnn" not in all_preds:
        return {"error": "one or both models failed to load"}

    gts_valid = np.array(valid_gts)
    n = len(gts_valid)

    # Bootstrap
    rng = np.random.default_rng(42)
    acc_flat_bs, acc_hgnn_bs, chd_flat_bs, chd_hgnn_bs = [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        g_s  = gts_valid[idx]
        pf_s = all_preds["flat"][idx]
        ph_s = all_preds["hgnn"][idx]
        acc_flat_bs.append((pf_s == g_s).mean())
        acc_hgnn_bs.append((ph_s == g_s).mean())
        chd_flat_bs.append(np.mean([TREE_DIST[g_s[i], pf_s[i]] for i in range(n)]))
        chd_hgnn_bs.append(np.mean([TREE_DIST[g_s[i], ph_s[i]] for i in range(n)]))

    acc_flat_bs = np.array(acc_flat_bs)
    acc_hgnn_bs = np.array(acc_hgnn_bs)
    chd_flat_bs = np.array(chd_flat_bs)
    chd_hgnn_bs = np.array(chd_hgnn_bs)
    gain_acc = acc_hgnn_bs - acc_flat_bs
    gain_chd = chd_flat_bs - chd_hgnn_bs   # positive = HGNN lower CHD (better)

    point_acc_flat = float((all_preds["flat"] == gts_valid).mean())
    point_acc_hgnn = float((all_preds["hgnn"] == gts_valid).mean())
    point_chd_flat = float(np.mean([TREE_DIST[gts_valid[i], all_preds["flat"][i]] for i in range(n)]))
    point_chd_hgnn = float(np.mean([TREE_DIST[gts_valid[i], all_preds["hgnn"][i]] for i in range(n)]))

    result = {
        "flat_acc":       {"mean": point_acc_flat,
                           "ci_95": [float(np.percentile(acc_flat_bs, 2.5)),
                                     float(np.percentile(acc_flat_bs, 97.5))]},
        "hgnn_acc":       {"mean": point_acc_hgnn,
                           "ci_95": [float(np.percentile(acc_hgnn_bs, 2.5)),
                                     float(np.percentile(acc_hgnn_bs, 97.5))]},
        "acc_gain":       {"mean": float(gain_acc.mean()),
                           "ci_95": [float(np.percentile(gain_acc, 2.5)),
                                     float(np.percentile(gain_acc, 97.5))]},
        "flat_chd":       {"mean": point_chd_flat,
                           "ci_95": [float(np.percentile(chd_flat_bs, 2.5)),
                                     float(np.percentile(chd_flat_bs, 97.5))]},
        "hgnn_chd":       {"mean": point_chd_hgnn,
                           "ci_95": [float(np.percentile(chd_hgnn_bs, 2.5)),
                                     float(np.percentile(chd_hgnn_bs, 97.5))]},
        "chd_reduction":  {"mean": float(gain_chd.mean()),
                           "ci_95": [float(np.percentile(gain_chd, 2.5)),
                                     float(np.percentile(gain_chd, 97.5))]},
        "p_hgnn_gt_flat": float((gain_acc > 0).mean()),
        "n": n, "n_bootstrap": n_bootstrap,
    }

    print(f"\n  Flat acc:  {point_acc_flat:.4f}  "
          f"[{result['flat_acc']['ci_95'][0]:.4f}, {result['flat_acc']['ci_95'][1]:.4f}]")
    print(f"  HGNN acc:  {point_acc_hgnn:.4f}  "
          f"[{result['hgnn_acc']['ci_95'][0]:.4f}, {result['hgnn_acc']['ci_95'][1]:.4f}]")
    print(f"  Gain:     {gain_acc.mean():+.4f}  "
          f"[{result['acc_gain']['ci_95'][0]:+.4f}, {result['acc_gain']['ci_95'][1]:+.4f}]")
    print(f"  P(HGNN > flat) = {result['p_hgnn_gt_flat']:.3f}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", nargs="+", type=int, choices=[1, 2, 3, 4, 5],
                   default=[1, 2, 3, 4, 5], help="Which ablations to run")
    p.add_argument("--sam-dir",        default="out/sam_eval_all1654_auto_vit_b")
    p.add_argument("--iou-threshold",  type=float, default=0.5)
    p.add_argument("--flat-run",       default="runs/minc_flat")
    p.add_argument("--hgnn-run",       default="runs/minc_hgnn")
    p.add_argument("--ablations-dir",  default="runs/graph_ablations")
    p.add_argument("--n-bootstrap",    type=int, default=1000)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--out",            default="runs/eval_ablations.json")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    sam_dir = repo_root / args.sam_dir
    df = pd.read_csv(sam_dir / "per_segment_metrics.csv")
    df_matched = df[df["best_iou"] >= args.iou_threshold].copy()
    print(f"SAM-matched segments (IoU ≥ {args.iou_threshold}): {len(df_matched)}")

    all_results = {}

    if 1 in args.ablation or 2 in args.ablation:
        print("\n── Ablations 1 & 2: graph structure and topology ───────────────")
        all_results["ablation_1_2"] = ablation_1_and_2(args, df_matched, transform, device)

    if 3 in args.ablation:
        print("\n── Ablation 3: cross-parent error reduction ─────────────────────")
        all_results["ablation_3"] = ablation_3(args, df_matched, transform, device)

    if 4 in args.ablation:
        print("\n── Ablation 4: context removal levels ───────────────────────────")
        all_results["ablation_4"] = ablation_4(args, df_matched, transform, device)

    if 5 in args.ablation:
        print("\n── Ablation 5: bootstrap confidence intervals ───────────────────")
        all_results["ablation_5"] = ablation_5(
            args, df_matched, transform, device, n_bootstrap=args.n_bootstrap
        )

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
