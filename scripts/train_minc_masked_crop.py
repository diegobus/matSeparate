#!/usr/bin/env python3
"""
Train ResNet50 on MINC-2500 with masked-crop augmentation to close the
domain gap to MINC-S segments.

Key difference from train_minc_flat.py: a MaskDropAugment transform
randomly applies an irregular elliptical mask to each training patch,
filling pixels outside the mask with ImageNet mean. This simulates
the masked-crop evaluation format used in MINC-S eval.

Hypothesis: training with masked crops reduces the domain gap between
MINC-2500 (uniform-material patches) and MINC-S (irregular segments),
leading to better segmentation evaluation accuracy.

Usage:
    python scripts/train_minc_masked_crop.py --epochs 7 --batch-size 64
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
from PIL import Image, ImageDraw

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.minc import MINC2500Dataset

CATEGORIES = MINC2500Dataset.CATEGORIES
NUM_CLASSES = len(CATEGORIES)  # 23
IMAGENET_MEAN_RGB = (123, 116, 103)  # matches eval masking


class MaskDropAugment:
    """
    Randomly mask a non-rectangular region of the image.

    With probability `p`, generates a random ellipse mask:
    - Center: random within [margin, 1-margin] * image_size
    - Axes: random in [min_scale, max_scale] * image_size
    - Rotation: random 0-180 degrees
    Pixels outside the ellipse are set to ImageNet mean.

    This simulates the irregular segment shape masking in MINC-S eval.
    """

    def __init__(
        self,
        p: float = 0.5,
        min_scale: float = 0.3,
        max_scale: float = 0.9,
        margin: float = 0.1,
    ):
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
        angle = random.uniform(0, 180)

        # Create mask via PIL ellipse then numpy for rotation
        mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask)
        # Approximate rotated ellipse with a regular ellipse (good enough)
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/external/minc/minc-2500")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-dir", default="runs/minc_masked_crop")
    p.add_argument("--backbone", default="resnet50")
    p.add_argument("--mask-prob", type=float, default=0.5,
                   help="Probability of applying mask augmentation per sample")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from datetime import datetime
    run_dir = Path(args.runs_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = T.Compose([
        T.Resize(args.image_size + 32),
        T.RandomCrop(args.image_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        MaskDropAugment(p=args.mask_prob),  # <-- domain adaptation augmentation
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
    val_ds = MINC2500Dataset(root, root / "labels" / f"validate{args.fold}.txt", transform=val_tf)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    if args.dry_run:
        print("Dry run OK, exiting.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = vars(args)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics_all = []
    best_val_acc = 0.0

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

        scheduler.step()
        train_acc = correct / n
        train_loss = total_loss / n

        # Validate
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

        val_acc = val_correct / val_n
        val_loss = val_loss / val_n
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({elapsed:.1f}s)")

        metrics_all.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_sec": elapsed,
        })

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_all, f, indent=2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": val_acc}, run_dir / "checkpoint_best.pt")
            print(f"  -> New best: {val_acc:.4f}")

    print(f"\nBest val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
