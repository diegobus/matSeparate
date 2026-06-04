#!/usr/bin/env python3
"""
Graph structure ablations for HGNN material classification.

Trains four variants on MINC-2500 fold-1, all with CE on leaf logits,
so the loss is not a confound:

  mlp_head     -- ResNet50 + 4-layer MLP (~same non-CNN params as HGNN)
  hgnn_ce      -- HGNN with true taxonomy graph (CE loss; baseline for ablations)
  random_tree  -- HGNN with a random spanning tree over the same 38 nodes
  full_graph   -- HGNN with fully-connected edges between all 38 nodes

All share the same backbone (ResNet50, pretrained ImageNet), 5-epoch training,
identical optimiser/LR/augmentation, and are evaluated on MINC-2500 val.

Usage:
    python scripts/train_graph_ablations.py
    python scripts/train_graph_ablations.py --variant mlp_head   # single variant
    python scripts/train_graph_ablations.py --epochs 10
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.minc import MINC2500Dataset
from gnn_classifier.hgnn import HGNN, ImageEncoder
from taxonomy.tree import get_taxonomy

VARIANTS = ['mlp_head', 'hgnn_ce', 'random_tree', 'full_graph']


# ── MLP head model ─────────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """ResNet50 + 4-layer MLP (~400K non-CNN params, comparable to HGNN's 616K)."""
    def __init__(self, num_classes=23, backbone='resnet50', pretrained=True, dropout=0.1):
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

    def num_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6


# ── Graph topology constructors ─────────────────────────────────────────────────

def make_random_tree(nodes, root='root', seed=0):
    """Random spanning tree over the same node set; valid rooted DAG."""
    rng = random.Random(seed)
    non_root = [n for n in nodes if n != root]
    rng.shuffle(non_root)
    g = nx.DiGraph()
    g.add_node(root)
    for i, node in enumerate(non_root):
        g.add_node(node)
        # Pick a random parent from nodes already added
        candidates = [root] + non_root[:i]
        g.add_edge(rng.choice(candidates), node)
    return g


def make_full_graph(nodes, root='root'):
    """Fully-connected directed graph (each node → every other node).
    Structured as root → all, plus all-to-all among taxonomy nodes, to
    give HGNN._init_graph something valid to digest."""
    g = nx.DiGraph()
    node_list = list(nodes)
    for n in node_list:
        g.add_node(n)
    non_root = [n for n in node_list if n != root]
    # Root → all
    for n in non_root:
        g.add_edge(root, n)
    # All-to-all among non-root (directed both ways; to_undirected handles dedup)
    for i, u in enumerate(non_root):
        for v in non_root[i+1:]:
            g.add_edge(u, v)
            g.add_edge(v, u)
    return g


def canonicalize(g):
    node_order = list(nx.topological_sort(g)) if nx.is_directed_acyclic_graph(g) else list(g.nodes)
    g2 = nx.DiGraph()
    for n in node_order:
        g2.add_node(n, **g.nodes[n])
    for u, v in g.edges:
        g2.add_edge(u, v, **g.edges[u, v])
    return g2


def build_hgnn(graph, cfg, pretrained=True):
    return HGNN(
        graph=graph,
        cnn_kwargs=dict(backbone=cfg['backbone'], pretrained=pretrained,
                        output_dim=cfg['cnn_output_dim'], finetune=False),
        gnn_kwargs=dict(input_dim=cfg['cnn_output_dim'],
                        hidden_dim=cfg['gnn_hidden_dim'],
                        output_dim=cfg['gnn_output_dim'],
                        num_layers=cfg['gnn_layers'],
                        num_heads=1, skip_connection=True,
                        dropout=cfg['dropout']),
        dropout_prob=cfg['dropout'],
    )


# ── Dataset helpers ─────────────────────────────────────────────────────────────

def make_loaders(data_root, fold, batch_size, image_size, num_workers):
    data_root = Path(data_root)
    aug = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2, 0.05),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize(256), T.CenterCrop(image_size), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = MINC2500Dataset(data_root, data_root / 'labels' / f'train{fold}.txt', transform=aug)
    val_ds   = MINC2500Dataset(data_root, data_root / 'labels' / f'validate{fold}.txt', transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ── Forward pass helpers ─────────────────────────────────────────────────────────

def forward_hgnn_leaf(model, imgs, leaf_indices):
    logits = model(imgs)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return logits[:, leaf_indices]   # [B, 23]


def forward_mlp(model, imgs):
    return model(imgs)               # [B, 23]


# ── One epoch ──────────────────────────────────────────────────────────────────

def run_epoch(model, loader, leaf_indices, optimiser, device, is_hgnn, train=True):
    model.train() if train else model.eval()
    total_loss = total_correct = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)        # [B] int, 0-22

            if is_hgnn:
                leaf_logits = forward_hgnn_leaf(model, imgs, leaf_indices)
            else:
                leaf_logits = forward_mlp(model, imgs)

            loss = F.cross_entropy(leaf_logits, labels)

            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss    += loss.item() * imgs.size(0)
            total_correct += (leaf_logits.argmax(1) == labels).sum().item()
            n             += imgs.size(0)

    return total_loss / n, total_correct / n


# ── Train one variant ──────────────────────────────────────────────────────────

def train_variant(variant, args, cfg, taxonomy_g, leaf_indices, device):
    run_dir = repo_root / args.runs_dir / variant
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    if variant == 'mlp_head':
        model = MLPClassifier(num_classes=23, backbone=cfg['backbone'],
                               pretrained=True, dropout=cfg['dropout'])
        is_hgnn = False
        non_cnn = sum(p.numel() for p in model.head.parameters()) / 1e3
        print(f"  MLP head params: {non_cnn:.0f}K non-backbone")
    else:
        if variant == 'hgnn_ce':
            g = taxonomy_g
        elif variant == 'random_tree':
            g = canonicalize(make_random_tree(list(taxonomy_g.nodes), seed=args.seed))
            # Carry over node attributes from taxonomy so HGNN initialisation works
            for n in g.nodes:
                g.nodes[n].update(taxonomy_g.nodes.get(n, {}))
        elif variant == 'full_graph':
            g = make_full_graph(list(taxonomy_g.nodes))
            for n in g.nodes:
                g.nodes[n].update(taxonomy_g.nodes.get(n, {}))
        model = build_hgnn(g, cfg, pretrained=True)
        is_hgnn = True
        cnn_p = sum(p.numel() for p in model.cnn.parameters())
        non_cnn = (sum(p.numel() for p in model.parameters()) - cnn_p) / 1e3
        print(f"  {variant} non-CNN params: {non_cnn:.0f}K")

    model = model.to(device)

    train_loader, val_loader = make_loaders(
        repo_root / args.data_root, args.fold, args.batch_size,
        args.image_size, args.num_workers
    )

    # Differential LR: backbone 0.1×, head 1.0×
    if is_hgnn:
        cnn_params  = list(model.cnn.parameters())
        head_params = [p for p in model.parameters() if not any(p is q for q in cnn_params)]
    else:
        cnn_params  = list(model.encoder.parameters())
        head_params = list(model.head.parameters())

    optimiser = torch.optim.AdamW([
        {'params': cnn_params,  'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best_val_acc = 0.0
    best_ckpt = run_dir / 'checkpoint_best.pt'

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, leaf_indices, optimiser, device, is_hgnn, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   leaf_indices, None,      device, is_hgnn, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:2d}/{args.epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | {elapsed:.0f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save = {'model_state_dict': model.state_dict(), 'variant': variant,
                    'val_acc': vl_acc, 'epoch': epoch, 'cfg': cfg,
                    'is_hgnn': is_hgnn}
            if is_hgnn:
                # Store node_to_idx so eval scripts can load it
                ckpt_ref = repo_root / 'runs/minc_hgnn/20260603_092440/checkpoint_best.pt'
                if ckpt_ref.exists():
                    ref = torch.load(ckpt_ref, map_location='cpu')
                    save['node_to_idx'] = ref['node_to_idx']
            torch.save(save, best_ckpt)

    # Save config
    with open(run_dir / 'config.json', 'w') as f:
        json.dump({**cfg, 'variant': variant, 'best_val_acc': best_val_acc,
                   'epochs': args.epochs, 'is_hgnn': is_hgnn}, f, indent=2)

    print(f"  Best val acc: {best_val_acc:.4f}  →  {best_ckpt}")
    return best_val_acc


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root',   default='data/external/minc/minc-2500')
    p.add_argument('--taxonomy',    default='taxonomy/assets/minc-taxonomy.json')
    p.add_argument('--fold',        type=int,   default=1)
    p.add_argument('--epochs',      type=int,   default=5)
    p.add_argument('--batch-size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight-decay',type=float, default=5e-4)
    p.add_argument('--image-size',  type=int,   default=224)
    p.add_argument('--num-workers', type=int,   default=4)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--runs-dir',    default='runs/graph_ablations')
    p.add_argument('--backbone',    default='resnet50')
    p.add_argument('--cnn-output-dim', type=int, default=512)
    p.add_argument('--gnn-hidden-dim', type=int, default=256)
    p.add_argument('--gnn-output-dim', type=int, default=128)
    p.add_argument('--gnn-layers',     type=int, default=2)
    p.add_argument('--dropout',        type=float, default=0.1)
    p.add_argument('--variant', choices=VARIANTS + ['all'], default='all',
                   help='Which variant to train (default: all sequentially)')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    taxonomy_path = repo_root / args.taxonomy
    g_raw = get_taxonomy(str(taxonomy_path))
    node_order = list(nx.topological_sort(g_raw))
    taxonomy_g = nx.DiGraph()
    for n in node_order:
        taxonomy_g.add_node(n, **g_raw.nodes[n])
    for u, v in g_raw.edges:
        taxonomy_g.add_edge(u, v, **g_raw.edges[u, v])

    # Leaf indices (same for all HGNN variants since we use true taxonomy node_to_idx)
    ref_ckpt_path = repo_root / 'runs/minc_hgnn/20260603_092440/checkpoint_best.pt'
    ref_ckpt = torch.load(ref_ckpt_path, map_location='cpu')
    node_to_idx = ref_ckpt['node_to_idx']
    leaves = [n for n in taxonomy_g.nodes if taxonomy_g.out_degree(n) == 0]
    leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                dtype=torch.long, device=device)
    print(f"Taxonomy: {taxonomy_g.number_of_nodes()} nodes, {len(leaves)} leaves")

    cfg = {
        'backbone': args.backbone,
        'cnn_output_dim': args.cnn_output_dim,
        'gnn_hidden_dim': args.gnn_hidden_dim,
        'gnn_output_dim': args.gnn_output_dim,
        'gnn_layers': args.gnn_layers,
        'dropout': args.dropout,
    }

    variants_to_run = VARIANTS if args.variant == 'all' else [args.variant]
    results = {}
    for variant in variants_to_run:
        print(f"\n{'='*60}")
        print(f"Training: {variant}")
        print(f"{'='*60}")
        acc = train_variant(variant, args, cfg, taxonomy_g, leaf_indices, device)
        results[variant] = acc

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Best val acc':>14}")
    print("-" * 36)
    for v, acc in results.items():
        print(f"{v:<20} {acc:>14.4f}")


if __name__ == '__main__':
    main()
