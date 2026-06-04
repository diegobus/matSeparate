#!/usr/bin/env python3
"""Plot training curves and write a compact summary for patch classifier runs."""

import argparse
import csv
import json
from pathlib import Path


def _load_metrics(run_dir: Path):
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        metrics = json.load(f)
    return metrics if metrics else None


def _metric_keys(metrics):
    first = metrics[0]
    if "val_leaf_acc" in first:
        return {
            "train_acc": "train_leaf_acc",
            "val_acc": "val_leaf_acc",
            "train_loss": "train_loss",
            "val_loss": "val_loss",
            "extra": ["val_hier_acc", "val_path_f1", "val_exact_match"],
        }
    return {
        "train_acc": "train_acc",
        "val_acc": "val_acc",
        "train_loss": "train_loss",
        "val_loss": "val_loss",
        "extra": [],
    }


def _best_row(run_dir: Path, metrics):
    keys = _metric_keys(metrics)
    best = max(metrics, key=lambda row: row.get(keys["val_acc"], float("-inf")))
    out = {
        "run_dir": str(run_dir),
        "run_name": run_dir.parent.name,
        "run_id": run_dir.name,
        "best_epoch": best["epoch"],
        "best_val_acc": best.get(keys["val_acc"]),
        "best_val_loss": best.get(keys["val_loss"]),
        "train_acc_at_best": best.get(keys["train_acc"]),
        "train_loss_at_best": best.get(keys["train_loss"]),
    }
    for key in keys["extra"]:
        out[key] = best.get(key)
    return out


def _plot(run_dir: Path, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = _metric_keys(metrics)
    epochs = [row["epoch"] for row in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    axes[0].plot(epochs, [row[keys["train_loss"]] for row in metrics], label="train")
    axes[0].plot(epochs, [row[keys["val_loss"]] for row in metrics], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row[keys["train_acc"]] for row in metrics], label="train")
    axes[1].plot(epochs, [row[keys["val_acc"]] for row in metrics], label="val")
    axes[1].set_title("Leaf accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(f"{run_dir.parent.name}/{run_dir.name}")
    fig.tight_layout()
    out_path = run_dir / "training_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="Run dirs or roots containing run dirs.")
    parser.add_argument("--summary-csv", type=Path, default=Path("runs/patch_classifier_grid_summary.csv"))
    args = parser.parse_args()

    run_dirs = []
    for path in args.paths:
        if (path / "metrics.json").exists():
            run_dirs.append(path)
        else:
            run_dirs.extend(sorted(p for p in path.glob("*/*") if (p / "metrics.json").exists()))
            run_dirs.extend(sorted(p for p in path.glob("*") if (p / "metrics.json").exists()))

    rows = []
    for run_dir in sorted(set(run_dirs)):
        metrics = _load_metrics(run_dir)
        if not metrics:
            continue
        _plot(run_dir, metrics)
        rows.append(_best_row(run_dir, metrics))

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Plotted {len(rows)} runs")
    print(f"Summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
