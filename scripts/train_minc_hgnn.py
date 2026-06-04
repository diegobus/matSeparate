#!/usr/bin/env python3
"""
Hierarchical GNN classifier on MINC-2500 with 23-class taxonomy.

Trains the same HGNN architecture used for Matador-C1, but with the
MINC-23 material hierarchy. Compares against the flat ResNet50 baseline.

Usage:
    python scripts/train_minc_hgnn.py
    python scripts/train_minc_hgnn.py --epochs 10 --batch-size 32
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
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

IMAGENET_MEAN_RGB = (123, 116, 103)


class MaskDropAugment:
    """Randomly mask a non-rectangular region of the image with ImageNet mean."""

    def __init__(self, p=0.5, min_scale=0.3, max_scale=0.9, margin=0.1):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.margin = margin

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        W, H = img.size
        cx = random.uniform(self.margin * W, (1 - self.margin) * W)
        cy = random.uniform(self.margin * H, (1 - self.margin) * H)
        rx = random.uniform(self.min_scale * W / 2, self.max_scale * W / 2)
        ry = random.uniform(self.min_scale * H / 2, self.max_scale * H / 2)
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        x0 = max(0, int(cx - rx))
        y0 = max(0, int(cy - ry))
        x1 = min(W, int(cx + rx))
        y1 = min(H, int(cy + ry))
        draw.ellipse([x0, y0, x1, y1], fill=255)
        mask_np = np.array(mask) > 127
        img_arr = np.array(img)
        bg = np.array(IMAGENET_MEAN_RGB, dtype=np.uint8)
        img_arr[~mask_np] = bg
        return Image.fromarray(img_arr)

from datasets.minc import MINC2500Dataset
from gnn_classifier.hgnn import HGNN
from gnn_classifier.loss import greedy_loss
from taxonomy.tree import get_taxonomy, get_hierarchy_levels

# Monkey-patch global_mean_pool for safety
import gnn_classifier.hgnn as _hgnn_mod
_orig_gmp = _hgnn_mod.global_mean_pool
def _safe_gmp(x, batch, size=None):
    return _orig_gmp(x, batch, size=size)
_hgnn_mod.global_mean_pool = _safe_gmp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/external/minc/minc-2500")
    p.add_argument("--taxonomy", default="taxonomy/assets/minc-taxonomy.json")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", default="runs/minc_hgnn")
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--cnn-output-dim", type=int, default=512)
    p.add_argument("--gnn-hidden-dim", type=int, default=256)
    p.add_argument("--gnn-output-dim", type=int, default=128)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mask-prob", type=float, default=0.0,
                   help="Probability of MaskDropAugment per sample (0=disabled)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonicalize_graph(graph, node_order):
    g = nx.DiGraph()
    for n in node_order:
        g.add_node(n, **graph.nodes[n])
    for u, v in graph.edges:
        g.add_edge(u, v, **graph.edges[u, v])
    return g


def compute_metrics(logits, multihot, leaf_indices, node_to_idx):
    """Compute leaf accuracy and hierarchical accuracy."""
    # leaf accuracy: argmax over leaf logits == true leaf
    leaf_logits = logits[:, leaf_indices]
    pred_leaf_pos = leaf_logits.argmax(dim=1)
    pred_leaf_idx = leaf_indices[pred_leaf_pos]

    true_leaf_idx = torch.tensor(
        [node_to_idx[c] for c in
         [MINC2500Dataset.CATEGORIES[i] for i in
          multihot[:, leaf_indices].argmax(dim=1).cpu().tolist()]],
        device=logits.device
    )
    leaf_acc = (pred_leaf_idx == true_leaf_idx).float().mean().item()

    # hierarchical: fraction of predicted path bits that are correct
    probs = torch.sigmoid(logits)
    pred_path = (probs > 0.5).float()
    hier_acc = ((pred_path * multihot).sum(1) / multihot.sum(1).clamp(min=1)).mean().item()

    return leaf_acc, hier_acc


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build taxonomy graph
    graph = get_taxonomy(args.taxonomy)
    node_order = list(nx.topological_sort(graph))
    graph = canonicalize_graph(graph, node_order)
    node_to_idx = {n: i for i, n in enumerate(node_order)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    leaves = [n for n in graph.nodes if graph.out_degree(n) == 0]
    leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                dtype=torch.long, device=device)

    levels_dict = get_hierarchy_levels(graph, root_name="root")
    hierarchy_levels = [
        np.array([node_to_idx[n] for n in levels_dict[l]], dtype=np.int64)
        for l in sorted(levels_dict.keys())
    ]

    num_nodes = graph.number_of_nodes()
    print(f"Taxonomy: {num_nodes} nodes, {len(leaves)} leaves")

    # Setup run directory
    from datetime import datetime
    run_dir = Path(args.runs_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Save node index
    with open(run_dir / "node_index.json", "w") as f:
        json.dump({"node_to_idx": node_to_idx, "idx_to_node": idx_to_node}, f, indent=2)

    # Transforms
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    mask_aug = [MaskDropAugment(p=args.mask_prob)] if args.mask_prob > 0 else []
    train_tf = T.Compose([
        T.Resize(args.image_size + 32),
        T.RandomCrop(args.image_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        *mask_aug,
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(args.image_size + 32),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    root = Path(args.data_root)
    train_ds = MINC2500Dataset(root, root / "labels" / f"train{args.fold}.txt",
                               graph=graph, transform=train_tf, node_to_idx=node_to_idx)
    val_ds   = MINC2500Dataset(root, root / "labels" / f"validate{args.fold}.txt",
                               graph=graph, transform=val_tf, node_to_idx=node_to_idx)

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
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn)

    # Build model — cnn_output_dim must equal gnn input_dim (projection layer bridges them)
    model = HGNN(
        graph=graph,
        cnn_kwargs=dict(
            backbone=args.backbone,
            pretrained=True,
            output_dim=args.cnn_output_dim,
            finetune=False,
        ),
        gnn_kwargs=dict(
            input_dim=args.cnn_output_dim,
            hidden_dim=args.gnn_hidden_dim,
            output_dim=args.gnn_output_dim,
            num_layers=args.gnn_layers,
            num_heads=1,
            skip_connection=True,
            dropout=args.dropout,
        ),
        dropout_prob=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    metrics_all = []
    best_val_leaf_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        total_loss = 0.0
        total_leaf_correct = 0
        n = 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            multihot = batch["target_multihot"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = greedy_loss(logits, multihot, hierarchy_levels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                leaf_logits = logits[:, leaf_indices]
                pred_leaf_pos = leaf_logits.argmax(dim=1)
                true_leaf_pos = multihot[:, leaf_indices].argmax(dim=1)
                total_leaf_correct += (pred_leaf_pos == true_leaf_pos).sum().item()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        train_loss = total_loss / n
        train_leaf_acc = total_leaf_correct / n
        scheduler.step()

        # --- Val ---
        model.eval()
        val_loss = 0.0
        val_leaf_correct = 0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                multihot = batch["target_multihot"].to(device)

                logits = model(imgs)
                loss = greedy_loss(logits, multihot, hierarchy_levels)
                val_loss += loss.item() * imgs.size(0)

                leaf_logits = logits[:, leaf_indices]
                pred_leaf_pos = leaf_logits.argmax(dim=1)
                true_leaf_pos = multihot[:, leaf_indices].argmax(dim=1)
                val_leaf_correct += (pred_leaf_pos == true_leaf_pos).sum().item()
                val_n += imgs.size(0)

        val_loss = val_loss / val_n
        val_leaf_acc = val_leaf_correct / val_n
        elapsed = time.time() - t0

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_leaf_acc": train_leaf_acc,
            "val_loss": val_loss,
            "val_leaf_acc": val_leaf_acc,
            "time_sec": elapsed,
        }
        metrics_all.append(metrics)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss={train_loss:.4f} leaf_acc={train_leaf_acc:.4f} | "
              f"val loss={val_loss:.4f} leaf_acc={val_leaf_acc:.4f} | "
              f"{elapsed:.1f}s")

        if val_leaf_acc > best_val_leaf_acc:
            best_val_leaf_acc = val_leaf_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_leaf_acc": val_leaf_acc,
                "node_to_idx": node_to_idx,
            }, run_dir / "checkpoint_best.pt")

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_all, f, indent=2)

    print(f"\nBest val leaf acc: {best_val_leaf_acc:.4f}")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
