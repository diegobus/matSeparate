#!/usr/bin/env python3
"""Build report-ready figures and narrative for the patch-classifier grid."""

import argparse
import csv
import json
import math
from pathlib import Path


def _kind(run_name: str) -> str:
    if "long_hgnn_nodewise" in run_name:
        return "HGNN node-wise"
    if "long_hgnn" in run_name:
        return "HGNN fixed head"
    if "hgnn_nodewise" in run_name:
        return "HGNN node-wise"
    if "hgnn" in run_name:
        return "HGNN fixed head"
    return "ResNet50 MLP"


def _short_name(run_name: str) -> str:
    name = run_name
    name = name.replace("long_hgnn_nodewise_", "Long node-wise HGNN ")
    name = name.replace("long_hgnn_fixed_", "Long fixed HGNN ")
    name = name.replace("grid_resnet50_", "ResNet ")
    name = name.replace("grid_hgnn_nodewise_", "Node-wise HGNN ")
    name = name.replace("grid_hgnn_", "Fixed HGNN ")
    name = name.replace("_cnn_average", "")
    name = name.replace("_combined", " combined")
    name = name.replace("_max", " max")
    name = name.replace("_", " ")
    name = name.replace("lr3e-04", "lr=3e-4")
    name = name.replace("lr1e-04", "lr=1e-4")
    name = name.replace("lr1e-03", "lr=1e-3")
    name = name.replace("wd1e-04", "wd=1e-4")
    name = name.replace("wd5e-04", "wd=5e-4")
    name = name.replace("drop0.2", "drop=0.2")
    name = name.replace("drop0.1", "drop=0.1")
    name = name.replace("drop0.0", "drop=0")
    name = name.replace("ls0.05", "ls=0.05")
    name = name.replace("ls0.0", "ls=0")
    name = name.replace("30ep", "30 epochs")
    return name


def _load_runs(runs_root: Path):
    rows = []
    patterns = [
        "grid_*/*/metrics.json",
        "long_hgnn_*_30ep/*/metrics.json",
    ]
    for metrics_path in sorted({p for pattern in patterns for p in runs_root.glob(pattern)}):
        run_dir = metrics_path.parent
        run_name = run_dir.parent.name
        metrics = json.loads(metrics_path.read_text())
        if not metrics:
            continue
        if "val_leaf_acc" in metrics[0]:
            val_acc_key = "val_leaf_acc"
            train_acc_key = "train_leaf_acc"
            model = _kind(run_name)
        else:
            val_acc_key = "val_acc"
            train_acc_key = "train_acc"
            model = _kind(run_name)
        best = max(metrics, key=lambda r: r.get(val_acc_key, float("-inf")))
        best_loss = min(metrics, key=lambda r: r.get("val_loss", float("inf")))
        rows.append({
            "run_name": run_name,
            "run_dir": run_dir,
            "model": model,
            "metrics": metrics,
            "val_acc_key": val_acc_key,
            "train_acc_key": train_acc_key,
            "best_epoch": best["epoch"],
            "best_val_acc": best[val_acc_key],
            "best_val_loss_at_acc": best["val_loss"],
            "best_loss_epoch": best_loss["epoch"],
            "best_loss": best_loss["val_loss"],
            "val_acc_at_best_loss": best_loss[val_acc_key],
            "train_acc_at_best": best.get(train_acc_key),
            "final": metrics[-1],
            "is_long_run": run_name.startswith("long_"),
        })
    rows.sort(key=lambda r: r["best_val_acc"], reverse=True)
    return rows


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _color(model: str) -> str:
    return {
        "ResNet50 MLP": "#2f6f9f",
        "HGNN fixed head": "#7a6a1f",
        "HGNN node-wise": "#8f4a73",
    }[model]


def _save(fig, out_base: Path):
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def _plot_ranked(rows, out_dir: Path):
    plt = _setup_matplotlib()
    ordered = list(reversed(rows))
    fig, ax = plt.subplots(figsize=(8.3, 5.4))
    labels = [_short_name(r["run_name"]) for r in ordered]
    colors = [_color(r["model"]) for r in ordered]
    values = [r["best_val_acc"] for r in ordered]
    bars = ax.barh(range(len(ordered)), values, color=colors, height=0.72)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels)
    ax.set_xlim(0.82, 0.905)
    ax.set_xlabel("Best validation leaf accuracy")
    ax.set_title("Patch-only Matador-C1 classifier runs")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(value + 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    handles = [
        plt.Line2D([0], [0], color=_color("ResNet50 MLP"), lw=6, label="ResNet50 MLP"),
        plt.Line2D([0], [0], color=_color("HGNN fixed head"), lw=6, label="HGNN fixed head"),
        plt.Line2D([0], [0], color=_color("HGNN node-wise"), lw=6, label="HGNN node-wise"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, out_dir / "fig1_grid_best_val_accuracy")
    plt.close(fig)


def _plot_top_curves(rows, out_dir: Path):
    plt = _setup_matplotlib()
    top_by_model = []
    for model in ("ResNet50 MLP", "HGNN fixed head", "HGNN node-wise"):
        candidates = [r for r in rows if r["model"] == model]
        if candidates:
            top_by_model.append(candidates[0])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharex=True)
    for r in top_by_model:
        epochs = [m["epoch"] for m in r["metrics"]]
        label = f"{r['model']} ({'30 ep' if r['is_long_run'] else 'grid'})"
        axes[0].plot(epochs, [m["val_loss"] for m in r["metrics"]], color=_color(r["model"]), lw=2, label=label)
        axes[1].plot(epochs, [m[r["val_acc_key"]] for m in r["metrics"]], color=_color(r["model"]), lw=2, label=label)
        axes[1].scatter([r["best_epoch"]], [r["best_val_acc"]], color=_color(r["model"]), s=28, zorder=3)
    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Validation leaf accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.80, 0.91)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Best run per model family, including long HGNN training")
    fig.tight_layout()
    _save(fig, out_dir / "fig2_top_model_training_curves")
    plt.close(fig)


def _plot_hgnn_hierarchy(rows, out_dir: Path):
    plt = _setup_matplotlib()
    hgnn = [r for r in rows if r["model"] == "HGNN fixed head"][0]
    nodewise = [r for r in rows if r["model"] == "HGNN node-wise"][0]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for r in (hgnn, nodewise):
        epochs = [m["epoch"] for m in r["metrics"]]
        suffix = "30 ep" if r["is_long_run"] else "grid"
        ax.plot(epochs, [m["val_leaf_acc"] for m in r["metrics"]], color=_color(r["model"]), lw=2, label=f"{r['model']} leaf ({suffix})")
        ax.plot(epochs, [m["val_hier_acc"] for m in r["metrics"]], color=_color(r["model"]), lw=2, ls="--", label=f"{r['model']} hier ({suffix})")
    ax.set_title("HGNN hierarchy metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation score")
    ax.set_ylim(0.82, 0.965)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir / "fig3_hgnn_hierarchy_metrics")
    plt.close(fig)


def _plot_family_summary(rows, out_dir: Path):
    plt = _setup_matplotlib()
    families = ["ResNet50 MLP", "HGNN fixed head", "HGNN node-wise"]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for i, family in enumerate(families):
        vals = [r["best_val_acc"] for r in rows if r["model"] == family and not r["is_long_run"]]
        long_vals = [r["best_val_acc"] for r in rows if r["model"] == family and r["is_long_run"]]
        if not vals:
            continue
        xs = [i + (j - (len(vals) - 1) / 2) * 0.055 for j in range(len(vals))]
        ax.scatter(xs, vals, color=_color(family), s=35, alpha=0.85)
        for value in long_vals:
            ax.scatter([i], [value], color=_color(family), s=95, marker="*", edgecolor="black", linewidth=0.7, zorder=3)
        mean = sum(vals) / len(vals)
        ax.hlines(mean, i - 0.22, i + 0.22, color=_color(family), lw=3)
        ax.text(i, mean + 0.004, f"mean {mean:.3f}", ha="center", fontsize=8, color=_color(family))
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families)
    ax.set_ylabel("Best validation leaf accuracy")
    ax.set_title("Model-family performance distribution")
    ax.set_ylim(0.835, 0.905)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, out_dir / "fig4_model_family_summary")
    plt.close(fig)


def _write_summary_csv(rows, out_dir: Path):
    fields = [
        "model", "run_name", "best_epoch", "best_val_acc", "best_val_loss_at_acc",
        "best_loss_epoch", "best_loss", "val_acc_at_best_loss", "train_acc_at_best",
    ]
    with open(out_dir / "patch_classifier_report_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})


def _write_narrative(rows, out_dir: Path):
    best_resnet = [r for r in rows if r["model"] == "ResNet50 MLP"][0]
    best_fixed = [r for r in rows if r["model"] == "HGNN fixed head"][0]
    best_nodewise = [r for r in rows if r["model"] == "HGNN node-wise"][0]
    long_fixed = [r for r in rows if r["model"] == "HGNN fixed head" and r["is_long_run"]][0]
    long_nodewise = [r for r in rows if r["model"] == "HGNN node-wise" and r["is_long_run"]][0]
    max_loss = [r for r in rows if "max" in r["run_name"] and "hgnn" in r["run_name"]]
    combined_fixed = [r for r in rows if r["model"] == "HGNN fixed head" and "combined" in r["run_name"]]
    max_best = max(max_loss, key=lambda r: r["best_val_acc"])
    combined_best = max(combined_fixed, key=lambda r: r["best_val_acc"])
    resnet_vals = [r["best_val_acc"] for r in rows if r["model"] == "ResNet50 MLP"]
    fixed_vals = [r["best_val_acc"] for r in rows if r["model"] == "HGNN fixed head" and not r["is_long_run"]]
    node_vals = [r["best_val_acc"] for r in rows if r["model"] == "HGNN node-wise" and not r["is_long_run"]]
    text = f"""# Patch Classifier Grid: Figure Narrative

We ran the original 15 patch-only Matador-C1 grid experiments spanning a flat ResNet50 MLP baseline, the fixed-output HGNN classifier, and the node-wise shared HGNN scorer, then extended the best HGNN settings to 30 epochs. Figure 1 ranks all runs by best validation leaf accuracy. The strongest local-appearance-only model was the ResNet50 MLP with label smoothing, reaching {best_resnet['best_val_acc']:.3f} validation leaf accuracy at epoch {best_resnet['best_epoch']}. This remains the main patch-only accuracy baseline for later global-context experiments.

The 30-epoch fixed-head HGNN reached {long_fixed['best_val_acc']:.3f} validation leaf accuracy at epoch {long_fixed['best_epoch']}, narrowing the gap to the best ResNet baseline to {(best_resnet['best_val_acc'] - long_fixed['best_val_acc']):.3f}. Its validation hierarchical accuracy reached {long_fixed['final'].get('val_hier_acc', float('nan')):.3f} and path F1 reached {long_fixed['final'].get('val_path_f1', float('nan')):.3f} by the end of training. This supports the interpretation that the HGNN is learning hierarchy-consistent structure, even though the patch-only setup still does not beat the flat classifier on leaf accuracy.

Figure 2 shows that the best ResNet run peaks early, while the extended fixed-head HGNN keeps improving across much of the longer schedule. This is useful for the report: the flat baseline is a strong discriminative classifier for local swatches, but the HGNN provides a more structured prediction mechanism that is compatible with hierarchy paths and future material insertion.

Figure 3 isolates the HGNN metrics. The long fixed-output HGNN remains stronger than the long node-wise shared scorer: {long_fixed['best_val_acc']:.3f} versus {long_nodewise['best_val_acc']:.3f} best validation leaf accuracy. The node-wise scorer remains methodologically interesting because it decouples the output layer from a fixed class head, but these results do not justify using it as the main accuracy model yet.

The loss comparison also favors the existing combined objective. The best combined fixed-head HGNN reached {combined_best['best_val_acc']:.3f}, while the best paper-style max-loss HGNN run reached {max_best['best_val_acc']:.3f}. In this implementation and data split, reverting wholesale to the max-loss formulation would weaken the patch classifier.

Figure 4 summarizes the family-level spread, with stars marking the long HGNN runs. ResNet grid runs averaged {sum(resnet_vals)/len(resnet_vals):.3f} best validation accuracy, fixed-head HGNN grid runs averaged {sum(fixed_vals)/len(fixed_vals):.3f}, and node-wise HGNN grid runs averaged {sum(node_vals)/len(node_vals):.3f}. The research narrative from these figures is not that HGNN already wins patch-only classification, but that local appearance alone appears close to saturated. The next meaningful study is therefore the global-context condition: keep the best local ResNet as the patch-only baseline, keep fixed-head HGNN as the hierarchy-aware baseline, and test whether adding full-scene context improves leaf accuracy and downstream segmentation behavior.
"""
    (out_dir / "patch_classifier_report_narrative.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/patch_classifier_report/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_runs(args.runs_root)
    if len(rows) < 15:
        raise SystemExit(f"Expected at least 15 runs, found {len(rows)}")
    _plot_ranked(rows, args.out_dir)
    _plot_top_curves(rows, args.out_dir)
    _plot_hgnn_hierarchy(rows, args.out_dir)
    _plot_family_summary(rows, args.out_dir)
    _write_summary_csv(rows, args.out_dir)
    _write_narrative(rows, args.out_dir)
    print(f"Wrote report assets to {args.out_dir}")


if __name__ == "__main__":
    main()
