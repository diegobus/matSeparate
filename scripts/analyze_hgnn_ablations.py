#!/usr/bin/env python3
"""
Post-hoc analysis ablations (no training required).

Ablation 3 – Severe-mistake reduction
  Confusion matrices grouped by parent taxonomy node.
  Shows whether HGNN reduces cross-parent confusions more than within-parent.

Ablation 4 – Context removal levels
  Evaluates HGNN vs flat at mask-drop severities: 0% (bbox), 25%, 50%, 75%, 100%.
  Uses the MINC-S GT segment masks (clean GT, not SAM-retrieved).

Ablation 5 – Stability (bootstrap 95% CIs)
  Resamples MINC-S GT segments 1000× to get confidence intervals on accuracy/CHD.

Usage:
    python scripts/analyze_hgnn_ablations.py
    python scripts/analyze_hgnn_ablations.py --ablation 3   # single ablation
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import torchvision.transforms as T
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from gnn_classifier.hgnn import HGNN, ImageEncoder
from taxonomy.tree import get_taxonomy
import networkx as nx

CATEGORIES = [
    'brick','carpet','ceramic','fabric','foliage','food','glass','hair',
    'leather','metal','mirror','other','painted','paper','plastic',
    'polishedstone','skin','sky','stone','tile','wallpaper','water','wood',
]
NUM_CLASSES = len(CATEGORIES)
IMAGENET_MEAN = (123, 116, 103)

# Parent groupings from taxonomy
PARENT_GROUPS = {
    'masonry':        ['brick', 'tile', 'ceramic'],
    'raw_stone':      ['stone', 'polishedstone'],
    'vitreous':       ['glass', 'mirror'],
    'metal':          ['metal'],
    'synthetic':      ['plastic', 'painted', 'wallpaper'],
    'textile':        ['fabric', 'carpet'],
    'wood_derived':   ['wood', 'paper'],
    'animal_derived': ['leather', 'hair', 'skin'],
    'living':         ['foliage', 'food'],
    'fluid':          ['water', 'sky'],
    'amorphous':      ['other'],
}
CAT_TO_PARENT = {cat: parent for parent, cats in PARENT_GROUPS.items() for cat in cats}


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_flat(run_dir, device):
    """Load flat ResNet50 from runs/minc_flat."""
    import timm, torch.nn as nn
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / 'checkpoint_best.pt', map_location=device)
    cfg = json.load(open(run_dir / 'config.json'))
    model = timm.create_model(cfg.get('backbone', 'resnet50'), pretrained=False, num_classes=23)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


def load_hgnn_model(run_dir, taxonomy_path, device):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / 'checkpoint_best.pt', map_location=device)
    node_to_idx = ckpt['node_to_idx']
    cfg = json.load(open(run_dir / 'config.json'))
    g = get_taxonomy(str(taxonomy_path))
    node_order = list(nx.topological_sort(g))
    g_canon = nx.DiGraph()
    for n in node_order:
        g_canon.add_node(n, **g.nodes[n])
    for u, v in g.edges:
        g_canon.add_edge(u, v, **g.edges[u, v])
    model = HGNN(
        graph=g_canon,
        cnn_kwargs=dict(backbone=cfg['backbone'], pretrained=False,
                        output_dim=cfg['cnn_output_dim'], finetune=False),
        gnn_kwargs=dict(input_dim=cfg['cnn_output_dim'], hidden_dim=cfg['gnn_hidden_dim'],
                        output_dim=cfg['gnn_output_dim'], num_layers=cfg['gnn_layers'],
                        num_heads=1, skip_connection=True, dropout=cfg['dropout']),
        dropout_prob=cfg['dropout'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    leaves = [n for n in g_canon.nodes if g_canon.out_degree(n) == 0]
    leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                dtype=torch.long, device=device)
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    leaf_to_cat = torch.zeros(len(leaf_indices), dtype=torch.long, device=device)
    for pos, idx in enumerate(leaf_indices.tolist()):
        leaf_to_cat[pos] = CATEGORIES.index(idx_to_node[idx])
    return model, leaf_indices, leaf_to_cat


# ── Crop helpers ───────────────────────────────────────────────────────────────

def bbox_crop(photo_np, mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None
    return Image.fromarray(photo_np[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1])


def masked_crop(photo_np, mask, drop_frac=1.0):
    """Crop to bbox; fill (drop_frac * 100)% of non-mask pixels with ImageNet mean."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    crop = photo_np[r0:r1, c0:c1].copy()
    m = mask[r0:r1, c0:c1]
    if drop_frac > 0:
        out_mask = ~m
        if drop_frac < 1.0:
            # Randomly keep (1 - drop_frac) of non-mask pixels
            keep = np.random.random(out_mask.shape) > drop_frac
            out_mask = out_mask & ~keep
        for ch, mv in enumerate(IMAGENET_MEAN):
            crop[:, :, ch][out_mask] = mv
    return Image.fromarray(crop)


# ── Inference helpers ──────────────────────────────────────────────────────────

def predict_batch(model, crops, transform, device, is_hgnn, leaf_indices=None, leaf_to_cat=None):
    """
    Returns (preds [N], softmax [N, 23]).
    leaf_to_cat: pass for greedy_loss HGNNs to remap argmax pos → CATEGORIES index.
                 Leave None for CE-trained models and MLP (position == CATEGORIES index).
    """
    if not crops:
        return np.array([], dtype=int), np.zeros((0, NUM_CLASSES))
    imgs = torch.stack([transform(c) for c in crops]).to(device)
    with torch.no_grad():
        logits = model(imgs)
        if is_hgnn:
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            logits = logits[:, leaf_indices]
        probs = F.softmax(logits, dim=1).cpu().numpy()
    pos_preds = probs.argmax(1)
    if leaf_to_cat is not None:
        preds = np.array([int(leaf_to_cat[p].item()) for p in pos_preds])
    else:
        preds = pos_preds
    return preds, probs


def load_gt_mask(mask_path, H, W):
    p = Path(mask_path)
    if not p.is_absolute():
        p = repo_root / p
    mask = np.array(Image.open(p))
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255)
                        .resize((W, H), Image.NEAREST)) > 127
    return mask.astype(bool)


# ── Tree distance for CHD ─────────────────────────────────────────────────────

def build_tree_distances():
    """Return 23x23 matrix of tree distances between leaf classes."""
    g = get_taxonomy(str(repo_root / 'taxonomy/assets/minc-taxonomy.json'))
    ug = g.to_undirected()
    dist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=float)
    for i, ci in enumerate(CATEGORIES):
        if ci not in ug:
            continue
        lengths = nx.single_source_shortest_path_length(ug, ci)
        for j, cj in enumerate(CATEGORIES):
            dist[i, j] = lengths.get(cj, 0)
    return dist


TREE_DIST = build_tree_distances()


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation 3: Confusion matrices grouped by parent
# ═══════════════════════════════════════════════════════════════════════════════

def ablation3_confusion(args, device):
    """Compare within-parent vs cross-parent error rates: flat vs HGNN."""
    print("\n" + "="*60)
    print("Ablation 3: Severe-mistake reduction (confusion by parent)")
    print("="*60)

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    flat_dir  = repo_root / 'runs/minc_flat'
    hgnn_dir  = repo_root / 'runs/minc_hgnn/20260603_092440'
    tax_path  = repo_root / 'taxonomy/assets/minc-taxonomy.json'
    sam_dir   = repo_root / 'out/sam_eval_top200_matador_auto_vit_b'

    flat_run = sorted((flat_dir).glob('*/checkpoint_best.pt'))
    if not flat_run:
        print("  [skip] flat checkpoint not found")
        return
    flat_model = load_flat(flat_run[0].parent, device)

    hgnn_model, leaf_indices, leaf_to_cat = load_hgnn_model(hgnn_dir, tax_path, device)

    df = pd.read_csv(sam_dir / 'per_segment_metrics.csv')
    df = df[df['best_iou'] >= 0.5].copy()
    print(f"  GT segments (IoU≥0.5): {len(df)}")

    # confusion[gt_parent][pred_parent] counts
    parent_names = list(PARENT_GROUPS.keys())
    P = len(parent_names)
    p2i = {p: i for i, p in enumerate(parent_names)}

    conf_flat = np.zeros((P, P), dtype=int)
    conf_hgnn = np.zeros((P, P), dtype=int)

    for _, row in df.iterrows():
        ppath = Path(row['photo_path'])
        if not ppath.is_absolute():
            ppath = repo_root / ppath
        if not ppath.exists():
            continue
        photo_np = np.array(Image.open(ppath).convert('RGB'))
        H, W = photo_np.shape[:2]

        mask = load_gt_mask(row['mask_path'], H, W)
        crop = masked_crop(photo_np, mask, drop_frac=1.0)
        if crop is None:
            continue

        gt_cat   = int(row['label_index'])
        gt_pname = CAT_TO_PARENT.get(CATEGORIES[gt_cat], 'amorphous')
        gt_pi    = p2i[gt_pname]

        # flat
        preds_f, _ = predict_batch(flat_model, [crop], transform, device, is_hgnn=False)
        pr_pname_f = CAT_TO_PARENT.get(CATEGORIES[preds_f[0]], 'amorphous')
        conf_flat[gt_pi, p2i[pr_pname_f]] += 1

        # hgnn
        preds_h, _ = predict_batch(hgnn_model, [crop], transform, device,
                                   is_hgnn=True, leaf_indices=leaf_indices,
                                   leaf_to_cat=leaf_to_cat)
        pr_pname_h = CAT_TO_PARENT.get(CATEGORIES[preds_h[0]], 'amorphous')
        conf_hgnn[gt_pi, p2i[pr_pname_h]] += 1

    # Summarise: within-parent vs cross-parent accuracy
    def summarise(conf, name):
        total   = conf.sum()
        correct = np.diag(conf).sum()
        within  = correct / max(total, 1)
        # cross-parent errors: off-diagonal / total
        cross   = (total - correct) / max(total, 1)
        print(f"\n  {name}  (total={total})")
        print(f"    Within-parent accuracy:  {within:.4f}")
        print(f"    Cross-parent error rate: {cross:.4f}")
        print(f"\n  Parent-level confusion (rows=GT, cols=pred):")
        header = " " * 16 + "  ".join(f"{p[:6]:>6}" for p in parent_names)
        print("  " + header)
        for i, pname in enumerate(parent_names):
            row_sum = conf[i].sum()
            if row_sum == 0:
                continue
            row_str = "  ".join(f"{conf[i,j]:>6}" for j in range(P))
            diag_pct = conf[i,i]/row_sum*100
            print(f"  {pname:<14}  {row_str}  ({diag_pct:.0f}%)")

    summarise(conf_flat, "Flat ResNet50")
    summarise(conf_hgnn, "HGNN")

    # Save
    out = repo_root / 'runs/ablation3_confusion.json'
    with open(out, 'w') as f:
        json.dump({'flat': conf_flat.tolist(), 'hgnn': conf_hgnn.tolist(),
                   'parent_names': parent_names}, f, indent=2)
    print(f"\n  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation 4: Context removal levels
# ═══════════════════════════════════════════════════════════════════════════════

def ablation4_context(args, device):
    """Accuracy vs mask-drop severity: 0% (bbox) → 25 → 50 → 75 → 100% (full mask)."""
    print("\n" + "="*60)
    print("Ablation 4: Context removal levels")
    print("="*60)

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    flat_dir  = repo_root / 'runs/minc_flat'
    hgnn_dir  = repo_root / 'runs/minc_hgnn/20260603_092440'
    tax_path  = repo_root / 'taxonomy/assets/minc-taxonomy.json'
    sam_dir   = repo_root / 'out/sam_eval_top200_matador_auto_vit_b'

    flat_run = sorted((flat_dir).glob('*/checkpoint_best.pt'))
    if not flat_run:
        print("  [skip] flat checkpoint not found")
        return
    flat_model = load_flat(flat_run[0].parent, device)
    hgnn_model, leaf_indices, leaf_to_cat = load_hgnn_model(hgnn_dir, tax_path, device)

    df = pd.read_csv(sam_dir / 'per_segment_metrics.csv')
    df = df[df['best_iou'] >= 0.5].copy()

    DROP_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]
    results = {drop: {'flat': {'correct': 0, 'n': 0},
                      'hgnn': {'correct': 0, 'n': 0}} for drop in DROP_LEVELS}

    for _, row in df.iterrows():
        ppath = Path(row['photo_path'])
        if not ppath.is_absolute():
            ppath = repo_root / ppath
        if not ppath.exists():
            continue
        photo_np = np.array(Image.open(ppath).convert('RGB'))
        H, W = photo_np.shape[:2]
        mask = load_gt_mask(row['mask_path'], H, W)
        gt = int(row['label_index'])

        for drop in DROP_LEVELS:
            crop = masked_crop(photo_np, mask, drop_frac=drop)
            if crop is None:
                continue

            pf, _ = predict_batch(flat_model, [crop], transform, device, is_hgnn=False)
            ph, _ = predict_batch(hgnn_model, [crop], transform, device,
                                  is_hgnn=True, leaf_indices=leaf_indices,
                                  leaf_to_cat=leaf_to_cat)

            results[drop]['flat']['correct'] += int(pf[0] == gt)
            results[drop]['flat']['n']       += 1
            results[drop]['hgnn']['correct'] += int(ph[0] == gt)
            results[drop]['hgnn']['n']       += 1

    print(f"\n  {'Drop%':>6}  {'Flat acc':>10}  {'HGNN acc':>10}  {'Gain':>8}")
    print("  " + "-" * 42)
    for drop in DROP_LEVELS:
        r = results[drop]
        f_acc = r['flat']['correct'] / max(r['flat']['n'], 1)
        h_acc = r['hgnn']['correct'] / max(r['hgnn']['n'], 1)
        print(f"  {drop*100:>5.0f}%  {f_acc:>10.4f}  {h_acc:>10.4f}  {h_acc-f_acc:>+8.4f}")

    out = repo_root / 'runs/ablation4_context.json'
    with open(out, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\n  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation 5: Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════════════

def ablation5_bootstrap(args, device):
    """1000-sample bootstrap 95% CIs on accuracy and CHD for flat vs HGNN."""
    print("\n" + "="*60)
    print("Ablation 5: Bootstrap confidence intervals (n=1000)")
    print("="*60)

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    flat_dir  = repo_root / 'runs/minc_flat'
    hgnn_dir  = repo_root / 'runs/minc_hgnn/20260603_092440'
    tax_path  = repo_root / 'taxonomy/assets/minc-taxonomy.json'
    sam_dir   = repo_root / 'out/sam_eval_top200_matador_auto_vit_b'

    flat_run = sorted((flat_dir).glob('*/checkpoint_best.pt'))
    if not flat_run:
        print("  [skip] flat checkpoint not found")
        return
    flat_model = load_flat(flat_run[0].parent, device)
    hgnn_model, leaf_indices, leaf_to_cat = load_hgnn_model(hgnn_dir, tax_path, device)

    df = pd.read_csv(sam_dir / 'per_segment_metrics.csv')
    df = df[df['best_iou'] >= 0.5].copy()
    print(f"  Collecting predictions on {len(df)} segments...")

    flat_correct = []
    flat_chd     = []
    hgnn_correct = []
    hgnn_chd     = []

    for _, row in df.iterrows():
        ppath = Path(row['photo_path'])
        if not ppath.is_absolute():
            ppath = repo_root / ppath
        if not ppath.exists():
            flat_correct.append(np.nan); flat_chd.append(np.nan)
            hgnn_correct.append(np.nan); hgnn_chd.append(np.nan)
            continue
        photo_np = np.array(Image.open(ppath).convert('RGB'))
        H, W = photo_np.shape[:2]
        mask = load_gt_mask(row['mask_path'], H, W)
        crop = masked_crop(photo_np, mask, drop_frac=1.0)
        gt = int(row['label_index'])

        if crop is None:
            flat_correct.append(np.nan); flat_chd.append(np.nan)
            hgnn_correct.append(np.nan); hgnn_chd.append(np.nan)
            continue

        pf, _ = predict_batch(flat_model, [crop], transform, device, is_hgnn=False)
        ph, _ = predict_batch(hgnn_model, [crop], transform, device,
                              is_hgnn=True, leaf_indices=leaf_indices,
                              leaf_to_cat=leaf_to_cat)

        flat_correct.append(float(pf[0] == gt))
        flat_chd.append(float(TREE_DIST[gt, pf[0]]))
        hgnn_correct.append(float(ph[0] == gt))
        hgnn_chd.append(float(TREE_DIST[gt, ph[0]]))

    flat_correct = np.array(flat_correct)
    flat_chd     = np.array(flat_chd)
    hgnn_correct = np.array(hgnn_correct)
    hgnn_chd     = np.array(hgnn_chd)

    # Remove NaNs
    valid = ~np.isnan(flat_correct)
    flat_correct = flat_correct[valid]
    flat_chd     = flat_chd[valid]
    hgnn_correct = hgnn_correct[valid]
    hgnn_chd     = hgnn_chd[valid]
    N = len(flat_correct)
    print(f"  Valid segments: {N}")

    rng = np.random.default_rng(args.seed)
    n_boot = args.n_bootstrap
    boot_stats = {'flat_acc': [], 'hgnn_acc': [], 'flat_chd': [], 'hgnn_chd': [], 'delta_acc': [], 'delta_chd': []}

    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot_stats['flat_acc'].append(flat_correct[idx].mean())
        boot_stats['hgnn_acc'].append(hgnn_correct[idx].mean())
        boot_stats['flat_chd'].append(flat_chd[idx].mean())
        boot_stats['hgnn_chd'].append(hgnn_chd[idx].mean())
        boot_stats['delta_acc'].append(hgnn_correct[idx].mean() - flat_correct[idx].mean())
        boot_stats['delta_chd'].append(flat_chd[idx].mean() - hgnn_chd[idx].mean())

    def ci(arr):
        a = np.array(arr)
        return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)

    print(f"\n  {'Metric':<22} {'Mean':>8}  {'95% CI':>20}")
    print("  " + "-" * 56)
    for key, label in [
        ('flat_acc',   'Flat accuracy'),
        ('hgnn_acc',   'HGNN accuracy'),
        ('delta_acc',  'Accuracy gain (HGNN−flat)'),
        ('flat_chd',   'Flat CHD'),
        ('hgnn_chd',   'HGNN CHD'),
        ('delta_chd',  'CHD reduction (flat−HGNN)'),
    ]:
        mean, lo, hi = ci(boot_stats[key])
        print(f"  {label:<22} {mean:>8.4f}  [{lo:.4f}, {hi:.4f}]")

    # Is the gain significantly > 0?
    delta_arr = np.array(boot_stats['delta_acc'])
    p_positive = (delta_arr > 0).mean()
    print(f"\n  P(HGNN acc > flat acc) = {p_positive:.3f}  "
          f"({'significant' if p_positive > 0.95 else 'not significant at 95%'})")

    out = repo_root / 'runs/ablation5_bootstrap.json'
    with open(out, 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in boot_stats.items()}, f, indent=2)
    print(f"\n  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ablation', type=int, choices=[3, 4, 5], default=None,
                   help='Run only one ablation (default: all)')
    p.add_argument('--n-bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    to_run = [args.ablation] if args.ablation else [3, 4, 5]
    if 3 in to_run:
        ablation3_confusion(args, device)
    if 4 in to_run:
        ablation4_context(args, device)
    if 5 in to_run:
        ablation5_bootstrap(args, device)


if __name__ == '__main__':
    main()
