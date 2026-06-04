#!/usr/bin/env python3
"""
Flat ResNet50 baseline for MINC-2500 23-class patch classification.

Usage:
    python scripts/train_minc_flat.py
    python scripts/train_minc_flat.py --epochs 10 --batch-size 64
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
from torch.utils.data import DataLoader
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.minc import MINC2500Dataset

CATEGORIES = MINC2500Dataset.CATEGORIES
NUM_CLASSES = len(CATEGORIES)  # 23


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/external/minc/minc-2500")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", default="runs/minc_flat")
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(backbone: str) -> nn.Module:
    model = timm.create_model(backbone, pretrained=True, num_classes=NUM_CLASSES)
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Setup run directory
    from datetime import datetime
    run_dir = Path(args.runs_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Transforms
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tf = T.Compose([
        T.Resize(args.image_size + 32),
        T.RandomCrop(args.image_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
    train_ds = MINC2500Dataset(root, root / "labels" / f"train{args.fold}.txt", transform=train_tf)
    val_ds   = MINC2500Dataset(root, root / "labels" / f"validate{args.fold}.txt", transform=val_tf)
    test_ds  = MINC2500Dataset(root, root / "labels" / f"test{args.fold}.txt", transform=val_tf)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    if args.dry_run:
        print("Dry run OK, exiting.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Save config
    config = vars(args)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_all = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            n += imgs.size(0)

        train_loss = total_loss / n
        train_acc  = correct / n

        # Val
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_n += imgs.size(0)

        val_loss = val_loss / val_n
        val_acc  = val_correct / val_n
        elapsed  = time.time() - t0

        scheduler.step()

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_sec": elapsed,
        }
        metrics_all.append(metrics)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, run_dir / "checkpoint_best.pt")

        # Save latest metrics
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_all, f, indent=2)

    print(f"\nBest val acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # Test-set evaluation with best checkpoint
    ckpt = torch.load(run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_correct, test_n = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(imgs)
            test_correct += (logits.argmax(1) == labels).sum().item()
            test_n += imgs.size(0)
    test_acc = test_correct / test_n
    print(f"Test acc (best ckpt): {test_acc:.4f}")

    with open(run_dir / "test_results.json", "w") as f:
        json.dump({"test_acc": test_acc, "best_epoch": best_epoch,
                   "best_val_acc": best_val_acc}, f, indent=2)

    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
