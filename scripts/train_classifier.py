#!/usr/bin/env python3
"""
Train flat ResNet50 baselines on MINC-2500 23-class patch classification.

Three model variants:
  flat      -- ResNet50 + flat 23-class CE head (standard baseline)
  hierloss  -- ResNet50 + 38-node head trained with greedy hierarchical loss
  maskdrop  -- ResNet50 + flat 23-class CE head + MaskDropAugment (closes domain
               gap to MINC-S masked-crop evaluation format)

All share the same backbone, optimizer, and augmentation pipeline except where
noted. Checkpoints are saved under runs/minc_<model>/<timestamp>/.

Usage:
    python scripts/train_classifier.py --model flat
    python scripts/train_classifier.py --model maskdrop --epochs 10 --batch-size 64
    python scripts/train_classifier.py --model hierloss --epochs 10
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.minc import MINC2500Dataset
from gnn_classifier.loss import greedy_loss
from taxonomy.tree import get_taxonomy, get_hierarchy_levels

CATEGORIES = MINC2500Dataset.CATEGORIES
NUM_CLASSES = len(CATEGORIES)          # 23
IMAGENET_MEAN_RGB = (123, 116, 103)    # matches eval-time masking


# ── Augmentation ──────────────────────────────────────────────────────────────

class MaskDropAugment:
    """
    Simulate the masked-crop format used at eval time on MINC-S segments.

    With probability `p`, draws a random ellipse mask and fills pixels outside
    it with ImageNet mean. This exposes the model to the same "context-removed"
    input format it will see during segmentation evaluation, closing the domain
    gap between MINC-2500 training patches and irregular MINC-S segment crops.
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
        ellipse_mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(ellipse_mask)
        draw.ellipse([int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry)], fill=255)
        inside = np.array(ellipse_mask) > 127
        arr = np.array(img).copy()
        arr[~inside] = np.array(IMAGENET_MEAN_RGB, dtype=np.uint8)
        return Image.fromarray(arr)


# ── Model builders ────────────────────────────────────────────────────────────

def build_flat_model(backbone: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Standard timm model with flat classification head."""
    return timm.create_model(backbone, pretrained=True, num_classes=num_classes)


def build_hierloss_model(backbone: str, num_nodes: int) -> nn.Module:
    """ResNet50 backbone with a linear head projecting to all taxonomy nodes."""
    base = timm.create_model(backbone, pretrained=True, num_classes=0)
    feat_dim = base.num_features
    head = nn.Linear(feat_dim, num_nodes)
    return nn.Sequential(base, head)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              is_hierloss: bool = False, leaf_indices=None, hierarchy_levels=None):
    model.train() if train else model.eval()
    total_loss = correct = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            imgs = batch["image"].to(device)
            if is_hierloss:
                multihot = batch["target_multihot"].to(device)
                logits = model(imgs)
                if isinstance(logits, (list, tuple)):
                    logits = logits[-1]
                loss = greedy_loss(logits, multihot, hierarchy_levels)
                # Leaf accuracy for monitoring
                leaf_logits = logits[:, leaf_indices]
                pred = leaf_logits.argmax(1)
                true = multihot[:, leaf_indices].argmax(1)
                correct += (pred == true).sum().item()
            else:
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                correct += (logits.argmax(1) == labels).sum().item()

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

    return total_loss / n, correct / n


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["flat", "hierloss", "maskdrop"],
                   default="flat", help="Which classifier variant to train")
    p.add_argument("--data-root",    default="data/external/minc/minc-2500")
    p.add_argument("--taxonomy",     default="taxonomy/assets/minc-taxonomy.json",
                   help="Only used for --model hierloss")
    p.add_argument("--fold",         type=int,   default=1)
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size",   type=int,   default=224)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--backbone",     default="resnet50")
    p.add_argument("--mask-prob",    type=float, default=0.5,
                   help="MaskDropAugment probability (only for --model maskdrop)")
    p.add_argument("--runs-dir",     default=None,
                   help="Override output directory (default: runs/minc_<model>)")
    p.add_argument("--dry-run",      action="store_true")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}")

    runs_dir = Path(args.runs_dir) if args.runs_dir else Path(f"runs/minc_{args.model}")
    run_dir = runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ── Taxonomy (needed only for hierloss) ───────────────────────────────────
    taxonomy_g = leaf_indices = hierarchy_levels = None
    num_nodes = NUM_CLASSES
    if args.model == "hierloss":
        import networkx as nx
        graph = get_taxonomy(str(repo_root / args.taxonomy))
        node_order = list(nx.topological_sort(graph))
        node_to_idx = {n: i for i, n in enumerate(node_order)}
        num_nodes = len(node_order)
        leaves = [n for n in graph.nodes if graph.out_degree(n) == 0]
        leaf_indices = torch.tensor(
            sorted([node_to_idx[n] for n in leaves]), dtype=torch.long, device=device
        )
        levels_dict = get_hierarchy_levels(graph, root_name="root")
        hierarchy_levels = [
            np.array([node_to_idx[n] for n in levels_dict[l]], dtype=np.int64)
            for l in sorted(levels_dict.keys())
        ]
        print(f"Taxonomy: {num_nodes} nodes, {len(leaves)} leaves")
        with open(run_dir / "node_index.json", "w") as f:
            json.dump({"node_to_idx": node_to_idx}, f, indent=2)

    # ── Transforms ───────────────────────────────────────────────────────────
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    sz = args.image_size
    extra_aug = [MaskDropAugment(p=args.mask_prob)] if args.model == "maskdrop" else []
    train_tf = T.Compose([
        T.Resize(sz + 32),
        T.RandomCrop(sz),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        *extra_aug,
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(sz + 32), T.CenterCrop(sz), T.ToTensor(), T.Normalize(mean, std),
    ])

    # ── Datasets ─────────────────────────────────────────────────────────────
    root = repo_root / args.data_root
    is_hierloss = args.model == "hierloss"

    def make_ds(split_file, tf):
        if is_hierloss:
            import networkx as nx
            graph = get_taxonomy(str(repo_root / args.taxonomy))
            node_order = list(nx.topological_sort(graph))
            node_to_idx_local = {n: i for i, n in enumerate(node_order)}
            return MINC2500Dataset(root, root / "labels" / split_file,
                                   graph=graph, transform=tf, node_to_idx=node_to_idx_local)
        return MINC2500Dataset(root, root / "labels" / split_file, transform=tf)

    train_ds = make_ds(f"train{args.fold}.txt", train_tf)
    val_ds   = make_ds(f"validate{args.fold}.txt", val_tf)
    test_ds  = make_ds(f"test{args.fold}.txt", val_tf)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    if args.dry_run:
        print("Dry run OK.")
        return

    def collate_hier(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "target_multihot": torch.stack([b["target_multihot"] for b in batch]),
        }

    collate = collate_hier if is_hierloss else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    # ── Model ─────────────────────────────────────────────────────────────────
    if is_hierloss:
        model = build_hierloss_model(args.backbone, num_nodes).to(device)
    else:
        model = build_flat_model(args.backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    with open(run_dir / "config.json", "w") as f:
        json.dump({**vars(args), "num_nodes": num_nodes}, f, indent=2)

    # ── Training ──────────────────────────────────────────────────────────────
    metrics_all = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True, is_hierloss=is_hierloss,
                                    leaf_indices=leaf_indices, hierarchy_levels=hierarchy_levels)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, None, device,
                                    train=False, is_hierloss=is_hierloss,
                                    leaf_indices=leaf_indices, hierarchy_levels=hierarchy_levels)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | {time.time()-t0:.1f}s")

        metrics_all.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                             "val_loss": vl_loss, "val_acc": vl_acc})

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_acc": vl_acc}
            if is_hierloss:
                save["node_to_idx"] = node_to_idx  # noqa: F821
            torch.save(save, run_dir / "checkpoint_best.pt")

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_all, f, indent=2)

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te_loss, te_acc = run_epoch(model, test_loader, criterion, None, device,
                                train=False, is_hierloss=is_hierloss,
                                leaf_indices=leaf_indices, hierarchy_levels=hierarchy_levels)
    print(f"\nBest val acc: {best_val_acc:.4f}  |  Test acc: {te_acc:.4f}")
    with open(run_dir / "test_results.json", "w") as f:
        json.dump({"test_acc": te_acc, "best_val_acc": best_val_acc}, f, indent=2)

    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
