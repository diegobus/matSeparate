#!/usr/bin/env python3
"""Train a global-context HGNN on Matador-C1."""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "gnn_classifier"))

from datasets.matador import MatadorC1GlobalDataset
from gnn_classifier.hgnn import GlobalContextHGNN
from gnn_classifier.loss import greedy_loss
from scripts.train_c1_hgnn import (
    _canonicalize_graph,
    _compute_hierarchy_levels,
    _ensure_2d_logits,
    _get_leaf_indices,
    _patch_bidirectional_global_edges,
    _resolve_device,
    exact_path_match,
    hierarchy_level_accuracy,
    leaf_top1_accuracy,
    node_bce_accuracy,
    path_f1,
)
from taxonomy.tree import get_taxonomy


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_transform(image_size, mean, std):
    return T.Compose([
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
        T.Normalize(mean=mean, std=std),
    ])


def _load_split_rows(split_csv: Path):
    with open(split_csv, newline="") as f:
        return list(csv.DictReader(f))


class _GlobalHGNNSubset:
    def __init__(self, base_ds, split_rows):
        self.base_ds = base_ds
        self.rows = split_rows
        self.sample_id_to_idx = {
            base_ds.samples[i]["sample_id"]: i for i in range(len(base_ds))
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        base_idx = self.sample_id_to_idx[self.rows[idx]["sample_id"]]
        sample = self.base_ds[base_idx]
        return {
            "local_image": sample["local_image"],
            "context_image": sample["context_image"],
            "target_multihot": sample["target_multihot"],
            "leaf_idx": sample["target_indices"][-1].item(),
        }


def _collate_fn(batch):
    return {
        "local_image": torch.stack([b["local_image"] for b in batch]),
        "context_image": torch.stack([b["context_image"] for b in batch]),
        "target_multihot": torch.stack([b["target_multihot"] for b in batch]),
        "leaf_idx": torch.tensor([b["leaf_idx"] for b in batch], dtype=torch.long),
    }


def _init_global_prototypes(model, idx_to_node, config):
    mode = config["prototypes"]["init"]
    if mode == "model_default":
        print("Prototype init: model default")
        return
    if mode == "random":
        print("Prototype init: random")
        nn.init.normal_(
            model.prototypes.weight,
            mean=0,
            std=config["prototypes"].get("random_std", 0.02),
        )
        return

    artifact_path = Path(config["prototypes"]["path"])
    if not artifact_path.exists():
        raise FileNotFoundError(f"Prototype artifact not found: {artifact_path}")

    artifact = torch.load(artifact_path, map_location="cpu")
    if artifact["idx_to_node"] != idx_to_node:
        raise ValueError("Prototype artifact node ordering differs from node_index.json")

    expected_shape = (model.num_nodes, model.prototypes.weight.shape[1])
    if artifact["prototypes"].shape != expected_shape:
        raise ValueError(
            f"Artifact prototypes shape {artifact['prototypes'].shape} != {expected_shape}"
        )

    with torch.no_grad():
        model.prototypes.weight.copy_(artifact["prototypes"])

    if config["prototypes"].get("sync_projection", True):
        proj_sd = artifact.get("projection_state_dict")
        if proj_sd is None:
            if not config["prototypes"].get("allow_projection_mismatch", False):
                raise ValueError("Prototype artifact missing projection_state_dict")
            print("WARNING: projection_state_dict missing; skipping projection sync")
        else:
            model.cnn.classifier.load_state_dict(proj_sd)
            model.context_cnn.classifier.load_state_dict(proj_sd)
            with torch.no_grad():
                model.projection.weight.copy_(torch.eye(model.projection.weight.shape[0]))
                if model.projection.bias is not None:
                    model.projection.bias.zero_()
            print("Prototype init: loaded cnn_average and synced both encoders")


def _run_epoch(
    model,
    loader,
    optimizer,
    leaf_indices,
    hierarchy_levels,
    device,
    loss_mode,
    parent_child_pairs,
    epoch,
    num_epochs,
    log_interval,
):
    model.train()
    return _evaluate_or_train(
        model,
        loader,
        leaf_indices,
        hierarchy_levels,
        device,
        loss_mode,
        parent_child_pairs,
        optimizer=optimizer,
        phase="train",
        epoch=epoch,
        num_epochs=num_epochs,
        log_interval=log_interval,
    )


@torch.no_grad()
def _validate(model, loader, leaf_indices, hierarchy_levels, device, loss_mode, parent_child_pairs):
    model.eval()
    return _evaluate_or_train(
        model,
        loader,
        leaf_indices,
        hierarchy_levels,
        device,
        loss_mode,
        parent_child_pairs,
        optimizer=None,
    )


def _evaluate_or_train(
    model,
    loader,
    leaf_indices,
    hierarchy_levels,
    device,
    loss_mode,
    parent_child_pairs,
    optimizer=None,
    phase="val",
    epoch=None,
    num_epochs=None,
    log_interval=0,
):
    totals = {
        "loss": 0.0,
        "leaf_acc": 0.0,
        "node_acc": 0.0,
        "exact_match": 0.0,
        "path_f1": 0.0,
        "hier_acc": 0.0,
    }
    count = 0
    t0 = time.time()
    if epoch is not None:
        print(
            f"Epoch {epoch:02d}/{num_epochs} {phase}: starting {len(loader)} batches",
            flush=True,
        )

    for batch_idx, batch in enumerate(loader, start=1):
        local_images = batch["local_image"].to(device)
        context_images = batch["context_image"].to(device)
        targets = batch["target_multihot"].to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        logits = _ensure_2d_logits(
            model(local_images, context_images),
            local_images.size(0),
        )
        loss = greedy_loss(
            logits,
            targets,
            hierarchy_levels,
            mode=loss_mode,
            parent_child_pairs=parent_child_pairs,
        )

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        bs = local_images.size(0)
        totals["loss"] += loss.item() * bs
        totals["leaf_acc"] += leaf_top1_accuracy(logits, targets, leaf_indices) * bs
        totals["node_acc"] += node_bce_accuracy(logits, targets) * bs
        totals["exact_match"] += exact_path_match(logits, targets) * bs
        totals["path_f1"] += path_f1(logits, targets) * bs
        totals["hier_acc"] += hierarchy_level_accuracy(logits, targets, hierarchy_levels) * bs
        count += bs

        if log_interval and batch_idx % log_interval == 0:
            elapsed = time.time() - t0
            sec_per_batch = elapsed / batch_idx
            remaining = sec_per_batch * (len(loader) - batch_idx)
            print(
                f"Epoch {epoch:02d}/{num_epochs} {phase} "
                f"batch {batch_idx:04d}/{len(loader)} "
                f"samples={count} loss={totals['loss'] / count:.4f} "
                f"leaf={totals['leaf_acc'] / count:.4f} "
                f"hier={totals['hier_acc'] / count:.4f} "
                f"{sec_per_batch:.2f}s/batch eta={remaining / 60:.1f}m",
                flush=True,
            )

    return {key: value / count for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train global-context HGNN on Matador-C1.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/c1_global_hgnn.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    _set_seed(config["training"]["seed"])
    device = _resolve_device(args.device or config["training"].get("device", "auto"))
    print(f"Device: {device}")

    with open(config["data"]["node_index"]) as f:
        node_index = json.load(f)
    node_to_idx = node_index["node_to_idx"]
    idx_to_node = node_index["idx_to_node"]

    graph = get_taxonomy(config["data"]["taxonomy_json"])
    graph = _canonicalize_graph(graph, idx_to_node)
    hierarchy_levels = _compute_hierarchy_levels(graph, node_to_idx)
    leaf_indices = _get_leaf_indices(graph, node_to_idx).to(device)
    parent_child_pairs = torch.tensor(
        [[node_to_idx[u], node_to_idx[v]] for u, v in graph.edges()],
        dtype=torch.long,
        device=device,
    )

    model = GlobalContextHGNN(
        graph=graph,
        path_predict=False,
        dropout_prob=config["model"]["dropout"],
        head_type=config["model"].get("head_type", "fixed_global_pool"),
        cnn_kwargs={
            "backbone": config["model"]["cnn_backbone"],
            "pretrained": config["model"]["cnn_pretrained"],
            "output_dim": config["model"]["cnn_output_dim"],
        },
        context_cnn_kwargs={
            "backbone": config["model"].get("context_cnn_backbone", config["model"]["cnn_backbone"]),
            "pretrained": config["model"].get("context_cnn_pretrained", config["model"]["cnn_pretrained"]),
            "output_dim": config["model"].get("context_cnn_output_dim", config["model"]["cnn_output_dim"]),
        },
        gnn_kwargs={
            "input_dim": config["model"]["gnn_input_dim"],
            "hidden_dim": config["model"]["gnn_hidden_dim"],
            "output_dim": config["model"]["gnn_output_dim"],
            "num_layers": config["model"]["gnn_layers"],
            "num_heads": config["model"]["gnn_heads"],
            "skip_connection": config["model"].get("skip_connection", True),
            "dropout": config["model"].get("dropout", 0.0),
        },
    ).to(device)

    if config.get("graph", {}).get("bidirectional_global_edges", False):
        _patch_bidirectional_global_edges(model)
    _init_global_prototypes(model, idx_to_node, config)

    mean = config.get("data", {}).get("mean", [0.485, 0.456, 0.406])
    std = config.get("data", {}).get("std", [0.229, 0.224, 0.225])
    local_transform = _build_transform(config["training"]["local_image_size"], mean, std)
    context_transform = _build_transform(config["training"]["context_image_size"], mean, std)

    base_ds = MatadorC1GlobalDataset(
        manifest_csv=config["data"]["manifest_csv"],
        taxonomy_json=config["data"]["taxonomy_json"],
        appearance_tar=config["data"].get("appearance_tar") or None,
        extracted_root=config["data"].get("appearance_extracted_root") or None,
        context_tar=config["data"].get("context_tar") or None,
        context_extracted_root=config["data"].get("context_extracted_root") or None,
        node_index_json=config["data"]["node_index"],
        transform=local_transform,
        context_transform=context_transform,
    )

    train_ds = _GlobalHGNNSubset(base_ds, _load_split_rows(Path(config["data"]["train_split"])))
    val_ds = _GlobalHGNNSubset(base_ds, _load_split_rows(Path(config["data"]["val_split"])))
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
        drop_last=True,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        collate_fn=_collate_fn,
    )

    loss_mode = config["training"].get("loss_mode", "combined")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    if args.dry_run:
        batch = next(iter(train_loader))
        logits = _ensure_2d_logits(
            model(batch["local_image"].to(device), batch["context_image"].to(device)),
            batch["local_image"].size(0),
        )
        loss = greedy_loss(
            logits,
            batch["target_multihot"].to(device),
            hierarchy_levels,
            mode=loss_mode,
            parent_child_pairs=parent_child_pairs,
        )
        loss.backward()
        optimizer.step()
        print(f"Dry run OK: logits={tuple(logits.shape)} loss={loss.item():.4f}")
        return

    runs_dir = Path(config["logging"]["runs_dir"]) / config["experiment_name"]
    run_dir = runs_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(run_dir / "node_index.json", "w") as f:
        json.dump({"node_to_idx": node_to_idx, "idx_to_node": idx_to_node}, f, indent=2)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    print(f"Run directory: {run_dir}")

    best_val_loss = float("inf")
    metrics_log = []
    num_epochs = config["training"]["num_epochs"]
    batch_log_interval = config.get("logging", {}).get("batch_log_interval", 50)
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_m = _run_epoch(
            model,
            train_loader,
            optimizer,
            leaf_indices,
            hierarchy_levels,
            device,
            loss_mode,
            parent_child_pairs,
            epoch,
            num_epochs,
            batch_log_interval,
        )
        val_m = _validate(model, val_loader, leaf_indices, hierarchy_levels, device, loss_mode, parent_child_pairs)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
            "time_sec": elapsed,
        }
        metrics_log.append(row)

        for key, value in train_m.items():
            writer.add_scalar(f"{key}/train", value, epoch)
        for key, value in val_m.items():
            writer.add_scalar(f"{key}/val", value, epoch)

        print(
            f"Epoch {epoch:02d}/{num_epochs} "
            f"train_loss={train_m['loss']:.4f} leaf={train_m['leaf_acc']:.4f} "
            f"hier={train_m['hier_acc']:.4f} f1={train_m['path_f1']:.4f} | "
            f"val_loss={val_m['loss']:.4f} leaf={val_m['leaf_acc']:.4f} "
            f"hier={val_m['hier_acc']:.4f} f1={val_m['path_f1']:.4f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_m["loss"],
                "val_acc": val_m["leaf_acc"],
                "node_to_idx": node_to_idx,
                "idx_to_node": idx_to_node,
                "hierarchy_levels": [level.tolist() for level in hierarchy_levels],
            }, run_dir / "checkpoint_best.pt")

    writer.close()
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
