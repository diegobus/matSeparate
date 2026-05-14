#!/usr/bin/env python3
"""
Evaluate a trained flat ResNet50 checkpoint on a Matador-C1 split.

Usage:
    python scripts/eval_c1_resnet.py \
        --run-dir runs/c1_resnet50_baseline/20250513_123000 \
        --split test
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.matador import MatadorC1Dataset
from scripts.train_c1_resnet import _build_subset_dataset, _accuracy


def _collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for key in keys:
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            shapes = {tuple(v.shape) for v in values}
            if len(shapes) == 1:
                out[key] = torch.stack(values)
            else:
                out[key] = values
        else:
            out[key] = values
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 C1 checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Load config and class mapping from run dir
    with open(args.run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(args.run_dir / "class_to_idx.json") as f:
        class_to_idx = json.load(f)

    device_str = args.device
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) else (device_str if device_str != "auto" else "cpu")
    )
    print(f"Device: {device}")

    split_csv = Path(config[f"{args.split}_split"])
    ds = _build_subset_dataset(split_csv, class_to_idx, config)
    loader = DataLoader(
        ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_fn,
    )

    import timm
    model = timm.create_model(
        config["model"]["backbone"],
        pretrained=False,
        num_classes=len(class_to_idx),
    )
    checkpoint = torch.load(args.run_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_top5 = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            total_acc += _accuracy(logits, targets) * images.size(0)

            # top-5 accuracy
            _, pred_top5 = logits.topk(k=min(5, logits.size(1)), dim=1)
            correct_top5 = (pred_top5 == targets.unsqueeze(1)).any(dim=1).float().sum().item()
            total_top5 += correct_top5
            count += images.size(0)

    print(f"{args.split.upper()} results:")
    print(f"  Loss:   {total_loss / count:.4f}")
    print(f"  Top-1:  {total_acc / count:.4f}")
    print(f"  Top-5:  {total_top5 / count:.4f}")


if __name__ == "__main__":
    main()
