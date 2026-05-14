#!/usr/bin/env python3
"""
Train a flat ResNet50 baseline on Matador-C1 leaf labels.

Usage:
    python scripts/train_c1_resnet.py \
        --config configs/experiments/c1_resnet50_baseline.yaml

    python scripts/train_c1_resnet.py --dry-run
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.matador import MatadorC1Dataset


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_transform(image_size: int, mean, std):
    return T.Compose([
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
        T.Normalize(mean=mean, std=std),
    ])


def _build_class_to_idx(split_csv: Path) -> dict:
    """Map sorted c1_label strings to integer indices."""
    import csv
    labels = set()
    with open(split_csv, newline="") as f:
        for r in csv.DictReader(f):
            labels.add(r["c1_label"])
    return {label: i for i, label in enumerate(sorted(labels))}


def _build_subset_dataset(split_csv: Path, class_to_idx: dict, config: dict):
    # We need to filter the manifest to only rows in the split.
    # The split CSVs are derived from manifest and contain all columns.
    import csv
    rows = []
    with open(split_csv, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Minimal wrapper that reads from extracted root or tar.
    # We'll reuse MatadorC1Dataset but override __len__/__getitem__.
    # Simpler: create a small wrapper class.
    base_ds = MatadorC1Dataset(
        manifest_csv=config["manifest_csv"],
        taxonomy_json=config["taxonomy_json"],
        appearance_tar=config.get("appearance_tar") or None,
        extracted_root=config.get("extracted_root") or None,
        transform=None,
    )

    # Build index from sample_id to base dataset index
    sample_id_to_idx = {base_ds.samples[i]["sample_id"]: i for i in range(len(base_ds))}

    class SubsetWrapper:
        def __init__(self, base_ds, split_rows, class_to_idx, transform):
            self.base_ds = base_ds
            self.rows = split_rows
            self.class_to_idx = class_to_idx
            self.transform = transform

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            base_idx = sample_id_to_idx[row["sample_id"]]
            sample = self.base_ds[base_idx]
            if self.transform:
                sample["image"] = self.transform(sample["image"])
            # Replace target with flat class index
            sample["target"] = torch.tensor(self.class_to_idx[sample["c1_label"]], dtype=torch.long)
            return sample

    return SubsetWrapper(base_ds, rows, class_to_idx, _build_transform(
        config["training"]["image_size"],
        config["data"]["mean"],
        config["data"]["std"],
    ))


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# --------------------------------------------------------------------------- #
# Collate
# --------------------------------------------------------------------------- #

def _collate_fn(batch):
    """Collate dict samples: stack fixed-size tensors, keep lists/variable-length as lists."""
    keys = batch[0].keys()
    out = {}
    for key in keys:
        values = [d[key] for d in batch]
        if isinstance(values[0], torch.Tensor):
            # Only stack if all shapes match; otherwise keep as list of tensors
            shapes = {tuple(v.shape) for v in values}
            if len(shapes) == 1:
                out[key] = torch.stack(values)
            else:
                out[key] = values
        else:
            out[key] = values
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_acc += _accuracy(logits, targets) * images.size(0)
        count += images.size(0)

    return total_loss / count, total_acc / count


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            total_acc += _accuracy(logits, targets) * images.size(0)
            count += images.size(0)
    return total_loss / count, total_acc / count


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train flat ResNet50 C1 baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/c1_resnet50_baseline.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Load two batches, run one fwd/bwd, one val batch, exit.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/auto).")
    args = parser.parse_args()

    config = _load_config(args.config)
    _set_seed(config["training"]["seed"])

    device_str = args.device or config["training"].get("device", "auto")
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else (device_str if device_str != "auto" else "cpu"))
    print(f"Device: {device}")

    # Class mapping
    class_to_idx = _build_class_to_idx(Path(config["train_split"]))
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes}")

    # Datasets
    train_ds = _build_subset_dataset(Path(config["train_split"]), class_to_idx, config)
    val_ds = _build_subset_dataset(Path(config["val_split"]), class_to_idx, config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,  # safe for tar / CPU
        drop_last=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_fn,
    )

    # Model
    import timm
    model = timm.create_model(
        config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        num_classes=num_classes,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Dry run --------------------------------------------------------------- #
    if args.dry_run:
        print("\n--- DRY RUN ---")
        print("Loading two train batches...")
        train_iter = iter(train_loader)
        batch1 = next(train_iter)
        batch2 = next(train_iter)
        print(f"Batch1 image shape: {batch1['image'].shape}, target shape: {batch1['target'].shape}")
        print(f"Batch2 image shape: {batch2['image'].shape}, target shape: {batch2['target'].shape}")

        print("Forward + backward on batch1...")
        model.train()
        optimizer.zero_grad()
        out = model(batch1["image"].to(device))
        loss = criterion(out, batch1["target"].to(device))
        loss.backward()
        optimizer.step()
        print(f"  Loss: {loss.item():.4f}")

        print("Evaluating one val batch...")
        model.eval()
        with torch.no_grad():
            vbatch = next(iter(val_loader))
            vout = model(vbatch["image"].to(device))
            vloss = criterion(vout, vbatch["target"].to(device))
            vacc = _accuracy(vout, vbatch["target"].to(device))
            print(f"  Val loss: {vloss.item():.4f}, Val acc: {vacc:.4f}")

        print("\nDry run complete. Exiting.")
        return

    # Full training --------------------------------------------------------- #
    runs_dir = Path(config["logging"]["runs_dir"]) / config["experiment_name"]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save config and class mapping
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(run_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    best_val_acc = -1.0
    metrics_log = []
    num_epochs = config["training"]["num_epochs"]
    log_interval = config["logging"]["log_interval"]

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        metrics_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_sec": elapsed,
        })

        if epoch % log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{num_epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                  f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
            }, run_dir / "checkpoint_best.pt")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
