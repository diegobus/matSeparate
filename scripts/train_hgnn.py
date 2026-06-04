#!/usr/bin/env python3
"""
Train the Hierarchical GNN (HGNN) material classifier on MINC-2500.

Architecture:
  ResNet50 backbone (ImageNet pretrained, frozen) → 512-d projection
  → prototype embeddings for each of the 38 taxonomy nodes
  → 2-layer Graph Attention Network over the material taxonomy tree
  → global mean pool → 128-d → 38 logits (one per taxonomy node)

Loss:
  greedy_loss: at each taxonomy level, cross-entropy over the children of
  the predicted parent node. This encodes the hierarchy directly into the
  gradient signal rather than treating all classes as equidistant.

The backbone is frozen during training; only the GNN head learns. This is
intentional — the goal is to test what graph structure adds over a fixed
representation, not to fine-tune the visual encoder.

Usage:
    python scripts/train_hgnn.py
    python scripts/train_hgnn.py --epochs 10 --batch-size 32
    python scripts/train_hgnn.py --mask-prob 0.5  # enable MaskDropAugment
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.minc import MINC2500Dataset
from gnn_classifier.hgnn import HGNN
from gnn_classifier.loss import greedy_loss
from taxonomy.tree import get_taxonomy, get_hierarchy_levels

CATEGORIES = MINC2500Dataset.CATEGORIES
IMAGENET_MEAN_RGB = (123, 116, 103)


# ── Optional augmentation ─────────────────────────────────────────────────────

class MaskDropAugment:
    """
    Randomly fill pixels outside a random ellipse with ImageNet mean.

    Simulates the masked-crop format used at MINC-S eval time, where non-segment
    pixels are filled with ImageNet mean before the crop is classified. Only used
    when --mask-prob > 0; disabled by default for the main HGNN training run.
    """

    def __init__(self, p: float = 0.5, min_scale: float = 0.3,
                 max_scale: float = 0.9, margin: float = 0.1):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.margin = margin

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        W, H = img.size
        cx = random.uniform(self.margin * W, (1 - self.margin) * W)
        cy = random.uniform(self.margin * H, (1 - self.margin) * H)
        rx = random.uniform(self.min_scale * W / 2, self.max_scale * W / 2)
        ry = random.uniform(self.min_scale * H / 2, self.max_scale * H / 2)
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).ellipse(
            [int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry)], fill=255
        )
        inside = np.array(mask) > 127
        arr = np.array(img).copy()
        arr[~inside] = np.array(IMAGENET_MEAN_RGB, dtype=np.uint8)
        return Image.fromarray(arr)


# ── Helpers ───────────────────────────────────────────────────────────────────

def canonicalize_graph(g: nx.DiGraph) -> nx.DiGraph:
    node_order = list(nx.topological_sort(g))
    g2 = nx.DiGraph()
    for n in node_order:
        g2.add_node(n, **g.nodes[n])
    for u, v in g.edges:
        g2.add_edge(u, v, **g.edges[u, v])
    return g2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Parse args ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",        default="data/external/minc/minc-2500")
    p.add_argument("--taxonomy",         default="taxonomy/assets/minc-taxonomy.json")
    p.add_argument("--fold",             type=int,   default=1)
    p.add_argument("--epochs",           type=int,   default=10)
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight-decay",     type=float, default=5e-4)
    p.add_argument("--image-size",       type=int,   default=224)
    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--backbone",         default="resnet50")
    p.add_argument("--cnn-output-dim",   type=int,   default=512)
    p.add_argument("--gnn-hidden-dim",   type=int,   default=256)
    p.add_argument("--gnn-output-dim",   type=int,   default=128)
    p.add_argument("--gnn-layers",       type=int,   default=2)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--mask-prob",        type=float, default=0.0,
                   help="MaskDropAugment probability during training (0 = disabled)")
    p.add_argument("--runs-dir",         default="runs/minc_hgnn")
    p.add_argument("--dry-run",          action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Taxonomy ──────────────────────────────────────────────────────────────
    tax_path = repo_root / args.taxonomy
    graph = canonicalize_graph(get_taxonomy(str(tax_path)))
    node_to_idx = {n: i for i, n in enumerate(graph.nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    leaves = [n for n in graph.nodes if graph.out_degree(n) == 0]
    leaf_indices = torch.tensor(
        sorted([node_to_idx[n] for n in leaves]), dtype=torch.long, device=device
    )
    levels_dict = get_hierarchy_levels(graph, root_name="root")
    hierarchy_levels = [
        np.array([node_to_idx[n] for n in levels_dict[l]], dtype=np.int64)
        for l in sorted(levels_dict.keys())
    ]
    print(f"Taxonomy: {graph.number_of_nodes()} nodes, {len(leaves)} leaves")

    # ── Run directory ─────────────────────────────────────────────────────────
    run_dir = repo_root / args.runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    with open(run_dir / "node_index.json", "w") as f:
        json.dump({"node_to_idx": node_to_idx, "idx_to_node": idx_to_node}, f, indent=2)

    # ── Transforms ────────────────────────────────────────────────────────────
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    sz = args.image_size
    extra_aug = [MaskDropAugment(p=args.mask_prob)] if args.mask_prob > 0 else []
    train_tf = T.Compose([
        T.Resize(sz + 32), T.RandomCrop(sz),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        *extra_aug,
        T.ToTensor(), T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(sz + 32), T.CenterCrop(sz), T.ToTensor(), T.Normalize(mean, std),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────────
    root = repo_root / args.data_root
    train_ds = MINC2500Dataset(root, root / "labels" / f"train{args.fold}.txt",
                               graph=graph, transform=train_tf, node_to_idx=node_to_idx)
    val_ds   = MINC2500Dataset(root, root / "labels" / f"validate{args.fold}.txt",
                               graph=graph, transform=val_tf,   node_to_idx=node_to_idx)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    if args.dry_run:
        print("Dry run OK.")
        return

    def collate_fn(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "target_multihot": torch.stack([b["target_multihot"] for b in batch]),
            "leaf_idx": torch.tensor([b["leaf_idx"] for b in batch], dtype=torch.long),
        }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HGNN(
        graph=graph,
        cnn_kwargs=dict(backbone=args.backbone, pretrained=True,
                        output_dim=args.cnn_output_dim, finetune=False),
        gnn_kwargs=dict(input_dim=args.cnn_output_dim, hidden_dim=args.gnn_hidden_dim,
                        output_dim=args.gnn_output_dim, num_layers=args.gnn_layers,
                        num_heads=1, skip_connection=True, dropout=args.dropout),
        dropout_prob=args.dropout,
    ).to(device)

    # Differential LR: GNN head 1× lr, frozen backbone 0.1× (handles fine-tune if enabled)
    cnn_params  = list(model.cnn.parameters())
    head_params = [p for p in model.parameters() if not any(p is q for q in cnn_params)]
    optimizer = torch.optim.AdamW([
        {"params": cnn_params,  "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    metrics_all = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        tr_loss = tr_correct = tr_n = 0
        for batch in train_loader:
            imgs     = batch["image"].to(device)
            multihot = batch["target_multihot"].to(device)
            logits   = model(imgs)
            loss     = greedy_loss(logits, multihot, hierarchy_levels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                pred = logits[:, leaf_indices].argmax(1)
                true = multihot[:, leaf_indices].argmax(1)
                tr_correct += (pred == true).sum().item()
            tr_loss += loss.item() * imgs.size(0)
            tr_n    += imgs.size(0)

        scheduler.step()

        # Validate
        model.eval()
        vl_loss = vl_correct = vl_n = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs     = batch["image"].to(device)
                multihot = batch["target_multihot"].to(device)
                logits   = model(imgs)
                loss     = greedy_loss(logits, multihot, hierarchy_levels)
                pred = logits[:, leaf_indices].argmax(1)
                true = multihot[:, leaf_indices].argmax(1)
                vl_correct += (pred == true).sum().item()
                vl_loss += loss.item() * imgs.size(0)
                vl_n    += imgs.size(0)

        tr_acc = tr_correct / tr_n
        vl_acc = vl_correct / vl_n
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss={tr_loss/tr_n:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss/vl_n:.4f} acc={vl_acc:.4f} | {elapsed:.1f}s")

        metrics_all.append({"epoch": epoch,
                             "train_loss": tr_loss / tr_n, "train_acc": tr_acc,
                             "val_loss": vl_loss / vl_n,   "val_acc": vl_acc})

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": vl_acc,
                "node_to_idx": node_to_idx,
            }, run_dir / "checkpoint_best.pt")

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_all, f, indent=2)

    print(f"\nBest val leaf acc: {best_val_acc:.4f}")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
