#!/usr/bin/env python3
"""
Flat ResNet50 trained with hierarchical greedy_loss (no GNN).

Key ablation: uses the same hierarchical loss as the HGNN but without
the graph attention network. Tests whether improvement from HGNN
comes from (a) the GNN architecture or (b) the hierarchical loss function.

Architecture:
  ResNet50 backbone → Linear head → 38 logits (all hierarchy nodes)

Loss:
  Same greedy_loss as HGNN (hierarchical softmax + binary CE on full path)

Usage:
    python scripts/train_minc_flat_hierloss.py
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
import torchvision.transforms as T
import timm
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.minc import MINC2500Dataset
from gnn_classifier.loss import greedy_loss
from taxonomy.tree import get_taxonomy, get_hierarchy_levels

CATEGORIES = MINC2500Dataset.CATEGORIES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/external/minc/minc-2500")
    p.add_argument("--taxonomy", default="taxonomy/assets/minc-taxonomy.json")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", default="runs/minc_flat_hierloss")
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--feat-dim", type=int, default=512)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def canonicalize_graph(graph, node_order):
    g = nx.DiGraph()
    for n in node_order: g.add_node(n, **graph.nodes[n])
    for u, v in graph.edges: g.add_edge(u, v, **graph.edges[u, v])
    return g


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build taxonomy
    graph = get_taxonomy(args.taxonomy)
    node_order = list(nx.topological_sort(graph))
    graph = canonicalize_graph(graph, node_order)
    node_to_idx = {n: i for i, n in enumerate(node_order)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    num_nodes = graph.number_of_nodes()

    leaves = sorted([n for n in graph.nodes if graph.out_degree(n) == 0])
    leaf_indices = torch.tensor([node_to_idx[n] for n in leaves], dtype=torch.long, device=device)

    levels_dict = get_hierarchy_levels(graph, root_name="root")
    hierarchy_levels = [
        np.array([node_to_idx[n] for n in levels_dict[l]], dtype=np.int64)
        for l in sorted(levels_dict.keys())
    ]

    print(f"Taxonomy: {num_nodes} nodes, {len(leaves)} leaves")

    from datetime import datetime
    run_dir = Path(args.runs_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "node_index.json", "w") as f:
        json.dump({"node_to_idx": node_to_idx, "idx_to_node": idx_to_node}, f, indent=2)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = T.Compose([
        T.Resize(args.image_size + 32), T.RandomCrop(args.image_size),
        T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(args.image_size + 32), T.CenterCrop(args.image_size),
        T.ToTensor(), T.Normalize(mean, std),
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
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    # Model: ResNet50 → feature projection → num_nodes logits
    backbone = timm.create_model(args.backbone, pretrained=True, num_classes=0).to(device)
    feat_dim = backbone.num_features
    head = nn.Sequential(
        nn.Linear(feat_dim, args.feat_dim),
        nn.ReLU(),
        nn.Linear(args.feat_dim, num_nodes),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    metrics_all = []
    best_val_leaf_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        backbone.train(); head.train()
        total_loss, leaf_correct, n = 0.0, 0, 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            multihot = batch["target_multihot"].to(device)

            optimizer.zero_grad()
            feats = backbone(imgs)
            logits = head(feats)  # [B, num_nodes]
            loss = greedy_loss(logits, multihot, hierarchy_levels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()

            with torch.no_grad():
                lf = logits[:, leaf_indices]
                pred_pos = lf.argmax(1)
                true_pos = multihot[:, leaf_indices].argmax(1)
                leaf_correct += (pred_pos == true_pos).sum().item()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        train_loss = total_loss / n
        train_leaf_acc = leaf_correct / n
        scheduler.step()

        backbone.eval(); head.eval()
        val_loss, val_leaf_correct, val_n = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                multihot = batch["target_multihot"].to(device)
                feats = backbone(imgs)
                logits = head(feats)
                loss = greedy_loss(logits, multihot, hierarchy_levels)
                val_loss += loss.item() * imgs.size(0)
                lf = logits[:, leaf_indices]
                pred_pos = lf.argmax(1)
                true_pos = multihot[:, leaf_indices].argmax(1)
                val_leaf_correct += (pred_pos == true_pos).sum().item()
                val_n += imgs.size(0)

        val_loss /= val_n
        val_leaf_acc = val_leaf_correct / val_n
        elapsed = time.time() - t0

        metrics = {
            "epoch": epoch, "train_loss": train_loss, "train_leaf_acc": train_leaf_acc,
            "val_loss": val_loss, "val_leaf_acc": val_leaf_acc, "time_sec": elapsed,
        }
        metrics_all.append(metrics)
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train {train_loss:.4f}/{train_leaf_acc:.4f} | "
              f"val {val_loss:.4f}/{val_leaf_acc:.4f} | {elapsed:.1f}s")

        if val_leaf_acc > best_val_leaf_acc:
            best_val_leaf_acc = val_leaf_acc
            torch.save({
                "epoch": epoch, "backbone_state": backbone.state_dict(),
                "head_state": head.state_dict(), "val_leaf_acc": val_leaf_acc,
            }, run_dir / "checkpoint_best.pt")

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_all, f, indent=2)

    print(f"\nBest val leaf acc: {best_val_leaf_acc:.4f}")

    # Test evaluation
    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone_state"])
    head.load_state_dict(ckpt["head_state"])
    backbone.eval(); head.eval()

    test_ds_eval = MINC2500Dataset(root, root / "labels" / f"test{args.fold}.txt",
                                   graph=graph, transform=val_tf, node_to_idx=node_to_idx)
    test_loader_eval = DataLoader(test_ds_eval, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    test_correct, test_n = 0, 0
    with torch.no_grad():
        for batch in test_loader_eval:
            imgs = batch["image"].to(device)
            multihot = batch["target_multihot"].to(device)
            lf = head(backbone(imgs))[:, leaf_indices]
            true_pos = multihot[:, leaf_indices].argmax(1)
            test_correct += (lf.argmax(1) == true_pos).sum().item()
            test_n += imgs.size(0)
    test_acc = test_correct / test_n
    print(f"Test leaf acc (best ckpt): {test_acc:.4f}")
    with open(run_dir / "test_results.json", "w") as f:
        json.dump({"test_leaf_acc": test_acc, "best_val_leaf_acc": best_val_leaf_acc}, f, indent=2)

    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
