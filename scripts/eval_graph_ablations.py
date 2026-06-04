#!/usr/bin/env python3
"""
SAM + classifier evaluation for graph ablation variants.

Evaluates all trained graph ablation models on MINC-S GT segments
matched by SAM (IoU >= 0.5, masked crop), reporting accuracy / CHD / Hier@d2.

Models evaluated:
  hgnn_original  -- existing HGNN (greedy_loss), the main result
  mlp_head       -- ResNet50 + MLP, parameter-matched (ablation 1)
  hgnn_ce        -- HGNN with true taxonomy graph, CE loss (ablation 1 baseline)
  random_tree    -- HGNN with random spanning tree (ablation 2)
  full_graph     -- HGNN with fully-connected graph (ablation 2)

Usage:
    python scripts/eval_graph_ablations.py
    python scripts/eval_graph_ablations.py --iou-threshold 0.5
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import networkx as nx
import pandas as pd
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from gnn_classifier.hgnn import HGNN, ImageEncoder
from taxonomy.tree import get_taxonomy

CATEGORIES = [
    'brick','carpet','ceramic','fabric','foliage','food','glass','hair',
    'leather','metal','mirror','other','painted','paper','plastic',
    'polishedstone','skin','sky','stone','tile','wallpaper','water','wood',
]
NUM_CLASSES = len(CATEGORIES)
IMAGENET_MEAN = (123, 116, 103)


# ── Tree distance matrix ───────────────────────────────────────────────────────

def build_tree_distances():
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


# ── Model definitions ──────────────────────────────────────────────────────────

import torch.nn as nn

class MLPClassifier(nn.Module):
    """Matches definition in train_graph_ablations.py."""
    def __init__(self, num_classes=23, backbone='resnet50', pretrained=False, dropout=0.1):
        super().__init__()
        self.encoder = ImageEncoder(
            output_dim=512, backbone=backbone, pretrained=pretrained, finetune=False
        )
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ── Model loaders ──────────────────────────────────────────────────────────────

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


def load_hgnn_ablation(run_dir, device):
    """Load HGNN ablation variant — uses reference node_to_idx since graph structure varies."""
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / 'checkpoint_best.pt', map_location=device)
    cfg = json.load(open(run_dir / 'config.json'))

    # For graph ablations, always use the taxonomy graph for the HGNN architecture
    # (the ablated topology is baked into the checkpoint weights, but we need the
    # canonical taxonomy graph to reconstruct the same HGNN object for loading).
    tax_path = repo_root / 'taxonomy/assets/minc-taxonomy.json'
    g = get_taxonomy(str(tax_path))
    node_order = list(nx.topological_sort(g))
    g_canon = nx.DiGraph()
    for n in node_order:
        g_canon.add_node(n, **g.nodes[n])
    for u, v in g.edges:
        g_canon.add_edge(u, v, **g.edges[u, v])

    # For random_tree / full_graph, we need to rebuild the exact same graph topology
    # that was used during training so the edge buffers match the saved state_dict.
    variant = cfg.get('variant', 'hgnn_ce')
    if variant == 'random_tree':
        from scripts.train_graph_ablations import make_random_tree, canonicalize
        g_use = canonicalize(make_random_tree(list(g_canon.nodes), seed=42))
        for n in g_use.nodes:
            g_use.nodes[n].update(g_canon.nodes.get(n, {}))
    elif variant == 'full_graph':
        from scripts.train_graph_ablations import make_full_graph
        g_use = make_full_graph(list(g_canon.nodes))
        for n in g_use.nodes:
            g_use.nodes[n].update(g_canon.nodes.get(n, {}))
    else:
        g_use = g_canon

    model = HGNN(
        graph=g_use,
        cnn_kwargs=dict(backbone=cfg['backbone'], pretrained=False,
                        output_dim=cfg['cnn_output_dim'], finetune=False),
        gnn_kwargs=dict(input_dim=cfg['cnn_output_dim'], hidden_dim=cfg['gnn_hidden_dim'],
                        output_dim=cfg['gnn_output_dim'], num_layers=cfg['gnn_layers'],
                        num_heads=1, skip_connection=True, dropout=cfg['dropout']),
        dropout_prob=cfg['dropout'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    # Always use canonical node_to_idx for leaf extraction
    node_to_idx = ckpt['node_to_idx']
    leaves = [n for n in g_canon.nodes if g_canon.out_degree(n) == 0]
    leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                dtype=torch.long, device=device)
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    leaf_to_cat = torch.zeros(len(leaf_indices), dtype=torch.long, device=device)
    for pos, idx in enumerate(leaf_indices.tolist()):
        leaf_to_cat[pos] = CATEGORIES.index(idx_to_node[idx])
    return model, leaf_indices, leaf_to_cat


def load_mlp_model(run_dir, device):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / 'checkpoint_best.pt', map_location=device)
    cfg = json.load(open(run_dir / 'config.json'))
    model = MLPClassifier(num_classes=23, backbone=cfg['backbone'],
                          pretrained=False, dropout=cfg['dropout'])
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval()


# ── Crop helper ────────────────────────────────────────────────────────────────

def crop_masked(photo_np, mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    crop = photo_np[r0:r1, c0:c1].copy()
    m = mask[r0:r1, c0:c1]
    for ch, mv in enumerate(IMAGENET_MEAN):
        crop[:, :, ch][~m] = mv
    return Image.fromarray(crop)


def load_gt_mask(mask_path, H, W):
    p = Path(mask_path)
    if not p.is_absolute():
        p = repo_root / p
    mask = np.array(Image.open(p))
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255)
                        .resize((W, H), Image.NEAREST)) > 127
    return mask.astype(bool)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model, is_hgnn, leaf_indices, df, transform, device, leaf_to_cat=None):
    """
    Evaluate a model on GT segments from df.

    leaf_to_cat: if provided (for greedy_loss-trained HGNNs), maps argmax position
                 in sorted leaf_indices back to CATEGORIES index.  For CE-trained
                 models (ablation variants + MLP) the argmax already equals the
                 CATEGORIES index directly, so leave as None.
    """
    preds, gts = [], []
    skipped = 0

    for _, row in df.iterrows():
        ppath = Path(row['photo_path'])
        if not ppath.is_absolute():
            ppath = repo_root / ppath
        if not ppath.exists():
            skipped += 1
            continue

        photo_np = np.array(Image.open(ppath).convert('RGB'))
        H, W = photo_np.shape[:2]
        mask = load_gt_mask(row['mask_path'], H, W)
        crop = crop_masked(photo_np, mask)
        if crop is None:
            skipped += 1
            continue

        img = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img)
            if is_hgnn:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                logits = logits[:, leaf_indices]
            pos = int(logits.argmax(1).item())
            pred = int(leaf_to_cat[pos].item()) if leaf_to_cat is not None else pos

        preds.append(pred)
        gts.append(int(row['label_index']))

    return np.array(preds), np.array(gts), skipped


def compute_metrics(preds, gts):
    n = len(preds)
    acc = (preds == gts).mean()
    chd = np.mean([TREE_DIST[gts[i], preds[i]] for i in range(n)])
    hier_d2 = np.mean([TREE_DIST[gts[i], preds[i]] <= 2 for i in range(n)])
    return {'accuracy': float(acc), 'CHD': float(chd), 'Hier@d2': float(hier_d2), 'n': n}


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sam-dir', default='out/sam_eval_top200_matador_auto_vit_b')
    p.add_argument('--iou-threshold', type=float, default=0.5)
    p.add_argument('--out', default='runs/graph_ablation_eval.json')
    p.add_argument('--dry-run', action='store_true', help='Run on first 50 segments only')
    return p.parse_args()


MODEL_SPECS = [
    ('hgnn_original', 'hgnn',        'runs/minc_hgnn/20260603_092440'),
    ('mlp_head',      'mlp',         'runs/graph_ablations/mlp_head'),
    ('hgnn_ce',       'hgnn_ablat',  'runs/graph_ablations/hgnn_ce'),
    ('random_tree',   'hgnn_ablat',  'runs/graph_ablations/random_tree'),
    ('full_graph',    'hgnn_ablat',  'runs/graph_ablations/full_graph'),
]


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tax_path = repo_root / 'taxonomy/assets/minc-taxonomy.json'
    sam_dir  = repo_root / args.sam_dir

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(sam_dir / 'per_segment_metrics.csv')
    df = df[df['best_iou'] >= args.iou_threshold].copy()
    if args.dry_run:
        df = df.head(50)
    print(f"Evaluating on {len(df)} segments (IoU≥{args.iou_threshold})")

    results = {}
    for name, model_type, run_path in MODEL_SPECS:
        run_dir = repo_root / run_path
        if not (run_dir / 'checkpoint_best.pt').exists() or not (run_dir / 'config.json').exists():
            print(f"\n  [{name}] checkpoint or config not ready — skipping")
            continue

        print(f"\n  Loading {name} ({model_type})...")
        l2c = None  # leaf_to_cat needed only for greedy_loss HGNN (hgnn_original)
        if model_type == 'hgnn':
            model, leaf_indices, l2c = load_hgnn_model(run_dir, tax_path, device)
            is_hgnn = True
        elif model_type == 'mlp':
            model = load_mlp_model(run_dir, device)
            leaf_indices = None
            is_hgnn = False
        else:  # hgnn_ablat — CE-trained, argmax position == CATEGORIES index directly
            model, leaf_indices, _ = load_hgnn_ablation(run_dir, device)
            is_hgnn = True

        print(f"  Evaluating {name}...")
        preds, gts, skipped = evaluate_model(model, is_hgnn, leaf_indices, df, transform, device,
                                             leaf_to_cat=l2c)
        m = compute_metrics(preds, gts)
        results[name] = m
        print(f"  {name}: acc={m['accuracy']:.4f}  CHD={m['CHD']:.4f}  Hier@d2={m['Hier@d2']:.4f}  (n={m['n']}, skipped={skipped})")

        del model
        torch.cuda.empty_cache()

    # Print table
    print(f"\n{'='*65}")
    print(f"{'Model':<18} {'Accuracy':>10} {'CHD':>8} {'Hier@d2':>10} {'N':>6}")
    print("-" * 55)
    for name, m in results.items():
        print(f"{name:<18} {m['accuracy']:>10.4f} {m['CHD']:>8.4f} {m['Hier@d2']:>10.4f} {m['n']:>6}")

    print(f"\nNote: all evaluated on SAM-matched segments (IoU≥{args.iou_threshold}, masked crop)")

    out_path = repo_root / args.out
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
