#!/usr/bin/env python3
"""
Train a dual-encoder ResNet baseline using local appearance patches and global context images.
"""

import argparse
import csv
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
from torch.utils.tensorboard import SummaryWriter

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.matador import MatadorC1GlobalDataset


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_device(device_str: str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _build_transform(image_size: int, mean, std, train: bool):
    if train:
        return T.Compose([
            T.Resize(image_size, antialias=True),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
        T.Normalize(mean=mean, std=std),
    ])


def _build_class_to_idx(split_csv: Path) -> dict:
    labels = set()
    with open(split_csv, newline="") as f:
        for r in csv.DictReader(f):
            labels.add(r["c1_label"])
    return {label: i for i, label in enumerate(sorted(labels))}


def _load_split_rows(split_csv: Path):
    with open(split_csv, newline="") as f:
        return list(csv.DictReader(f))


def _validate_global_manifest(manifest_csv: Path):
    with open(manifest_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Global manifest is empty: {manifest_csv}")
    missing = [r["sample_id"] for r in rows if not r.get("context_path")]
    texture_context = [
        r["sample_id"] for r in rows if "texture_img" in r.get("context_path", "")
    ]
    non_context = [
        r["sample_id"] for r in rows if "context_img" not in r.get("context_path", "")
    ]
    if missing:
        raise ValueError(
            f"Global manifest has {len(missing)} rows without context_path, "
            f"e.g. {missing[:5]}"
        )
    if texture_context:
        raise ValueError(
            "Global manifest appears to use local swatches as context: "
            f"{len(texture_context)} rows contain texture_img, e.g. {texture_context[:5]}"
        )
    if non_context:
        raise ValueError(
            f"Global manifest has {len(non_context)} rows outside context_img, "
            f"e.g. {non_context[:5]}"
        )
    print(f"Validated global manifest: {len(rows)} rows, all context_path under context_img")


class _GlobalSubset:
    def __init__(self, base_ds, split_rows, class_to_idx, sample_id_to_idx):
        self.base_ds = base_ds
        self.rows = split_rows
        self.class_to_idx = class_to_idx
        self.sample_id_to_idx = sample_id_to_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        sample = self.base_ds[self.sample_id_to_idx[row["sample_id"]]]
        sample["target"] = torch.tensor(self.class_to_idx[sample["c1_label"]], dtype=torch.long)
        return sample


def _build_dataset(split_csv: Path, class_to_idx: dict, config: dict, train: bool):
    data_cfg = config["data"]
    app_root = data_cfg.get("appearance_extracted_root") or data_cfg.get("extracted_root")
    app_tar = data_cfg.get("appearance_tar")
    if app_root:
        app_tar = None

    context_root = data_cfg.get("context_extracted_root")
    context_tar = data_cfg.get("context_tar")
    if context_root:
        context_tar = None

    local_transform = _build_transform(
        config["training"]["local_image_size"],
        data_cfg["mean"],
        data_cfg["std"],
        train=train,
    )
    context_transform = _build_transform(
        config["training"]["context_image_size"],
        data_cfg["mean"],
        data_cfg["std"],
        train=train,
    )
    base_ds = MatadorC1GlobalDataset(
        manifest_csv=data_cfg["manifest_csv"],
        taxonomy_json=data_cfg["taxonomy_json"],
        appearance_tar=app_tar or None,
        extracted_root=app_root or None,
        context_tar=context_tar or None,
        context_extracted_root=context_root or None,
        transform=local_transform,
        context_transform=context_transform,
    )
    sample_id_to_idx = {base_ds.samples[i]["sample_id"]: i for i in range(len(base_ds))}
    return _GlobalSubset(base_ds, _load_split_rows(split_csv), class_to_idx, sample_id_to_idx)


def _collate_fn(batch):
    return {
        "local_image": torch.stack([b["local_image"] for b in batch]),
        "context_image": torch.stack([b["context_image"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "sample_id": [b["sample_id"] for b in batch],
    }


class DualEncoderResNet(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        import timm

        self.local_encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.context_encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.local_encoder.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, local_image, context_image):
        local_feat = self.local_encoder(local_image)
        context_feat = self.context_encoder(context_image)
        return self.classifier(torch.cat([local_feat, context_feat], dim=1))


def _load_local_checkpoint(model: DualEncoderResNet, checkpoint_path: str | None):
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.local_encoder.load_state_dict(state, strict=False)
    print(f"Warm-started local encoder from {checkpoint_path}")
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for batch in loader:
        local_images = batch["local_image"].to(device)
        context_images = batch["context_image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(local_images, context_images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_acc += _accuracy(logits, targets) * bs
        count += bs
    return total_loss / count, total_acc / count


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            local_images = batch["local_image"].to(device)
            context_images = batch["context_image"].to(device)
            targets = batch["target"].to(device)
            logits = model(local_images, context_images)
            loss = criterion(logits, targets)
            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_acc += _accuracy(logits, targets) * bs
            count += bs
    return total_loss / count, total_acc / count


def main():
    parser = argparse.ArgumentParser(description="Train dual-encoder global Matador-C1 ResNet.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/c1_global_resnet50.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    config = _load_config(args.config)
    _set_seed(config["training"]["seed"])
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers

    device = _resolve_device(args.device or config["training"].get("device", "auto"))
    print(f"Device: {device}")

    _validate_global_manifest(Path(config["data"]["manifest_csv"]))

    train_split = Path(config["data"]["train_split"])
    val_split = Path(config["data"]["val_split"])
    class_to_idx = _build_class_to_idx(train_split)
    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes}")

    train_ds = _build_dataset(train_split, class_to_idx, config, train=True)
    val_ds = _build_dataset(val_split, class_to_idx, config, train=False)
    print(f"Samples: train={len(train_ds)} val={len(val_ds)}")

    num_workers = config["training"].get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )

    model = DualEncoderResNet(
        backbone=config["model"]["backbone"],
        pretrained=config["model"].get("pretrained", True),
        num_classes=num_classes,
        hidden_dim=config["model"].get("hidden_dim", 1024),
        dropout=config["model"].get("dropout", 0.1),
    )
    _load_local_checkpoint(model, config["model"].get("local_checkpoint"))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config["training"].get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    if args.dry_run:
        batch = next(iter(train_loader))
        print(f"local_image: {batch['local_image'].shape}")
        print(f"context_image: {batch['context_image'].shape}")
        print(f"target: {batch['target'].shape}")
        model.train()
        optimizer.zero_grad()
        logits = model(batch["local_image"].to(device), batch["context_image"].to(device))
        loss = criterion(logits, batch["target"].to(device))
        loss.backward()
        optimizer.step()
        print(f"dry_run_loss={loss.item():.4f}")
        return

    runs_dir = Path(config["logging"]["runs_dir"]) / config["experiment_name"]
    run_dir = runs_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(run_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    best_val_acc = -1.0
    metrics = []
    for epoch in range(1, config["training"]["num_epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time_sec": elapsed,
        }
        metrics.append(row)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        if epoch % config["logging"].get("log_interval", 1) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{config['training']['num_epochs']} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"({elapsed:.1f}s)"
            )
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
        json.dump(metrics, f, indent=2)
    writer.close()
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
