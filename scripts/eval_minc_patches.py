#!/usr/bin/env python3
"""
Evaluate an HGNN material classifier on a balanced subset of the MINC-2500 test set.

MINC-2500 is a *patch classification* dataset: every image is a single-material crop
labelled with one of 23 MINC categories. There are no segmentation masks, so this script
measures patch-level material-classification accuracy -- the metric that actually
distinguishes two candidate classifiers as front-ends for the segmentation pipeline.

It supports comparing classifiers that live in *different* label spaces:

* The MINC-trained classifier predicts the 23 MINC categories directly, so a prediction is
  correct iff ``predicted_leaf == ground_truth``.
* The C1 (Matador-trained) classifier predicts 37 Matador materials. Pass a crosswalk
  (``--crosswalk taxonomy/assets/minc_to_matador.json``) mapping each MINC ground-truth
  category to the set of acceptable Matador leaves; a prediction is correct iff the
  predicted leaf is in that set. MINC categories absent from the crosswalk are reported as
  "uncovered" and excluded from the C1 accuracy (so the number isn't unfairly deflated by
  materials Matador simply doesn't model).

Per-class accuracy is always keyed by the **MINC ground-truth category**, so two runs
(C1 vs MINC) are directly comparable per material.

Examples
--------
# MINC classifier (native label space):
python scripts/eval_minc_patches.py \
    --run-dir runs/minc_hgnn_baseline/20260529_192559 \
    --images-root data/external/minc/minc-2500 \
    --n-per-class 100 --device cuda --out out/minc_eval/minc

# C1 classifier (cross-taxonomy, via crosswalk):
python scripts/eval_minc_patches.py \
    --run-dir runs/c1_hgnn_baseline/avg_init_20260529_004313 \
    --images-root data/external/minc/minc-2500 \
    --crosswalk taxonomy/assets/minc_to_matador.json \
    --n-per-class 100 --device cuda --out out/minc_eval/c1

# Side-by-side comparison figure from two finished runs:
python scripts/eval_minc_patches.py --compare \
    out/minc_eval/minc/metrics.json out/minc_eval/c1/metrics.json \
    --out out/minc_eval/comparison.png
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

DEFAULT_MANIFEST = "data/processed/minc/splits/test.csv"
DEFAULT_IMAGES_ROOT = "data/external/minc/minc-2500"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def balanced_subset(
    rows: List[Dict[str, str]],
    n_per_class: int,
    seed: int,
    label_key: str = "material_label",
) -> List[Dict[str, str]]:
    """Up to ``n_per_class`` rows per ground-truth material, shuffled deterministically."""
    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_label[r[label_key]].append(r)

    rng = random.Random(seed)
    out: List[Dict[str, str]] = []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        out.extend(items[:n_per_class])
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(args: argparse.Namespace) -> Dict:
    from scripts.infer_api import HGNNInference

    manifest = Path(args.manifest)
    images_root = Path(args.images_root)
    rows = load_manifest(manifest)
    subset = balanced_subset(rows, args.n_per_class, args.seed)
    if args.limit:
        subset = subset[: args.limit]

    # Resolve image paths up front; fail loudly if the image tree isn't where we expect.
    missing = 0
    samples = []
    for r in subset:
        p = images_root / r["image_path"]
        if not p.exists():
            missing += 1
            if missing <= 5:
                print(f"  [warn] missing image: {p}")
            continue
        samples.append((p, r["material_label"]))
    if missing:
        print(f"[warn] {missing} images not found under {images_root}")
    if not samples:
        raise SystemExit(
            f"No images found. Check --images-root (got '{images_root}'); manifest paths "
            f"look like '{rows[0]['image_path']}'."
        )

    crosswalk: Optional[Dict[str, List[str]]] = None
    if args.crosswalk:
        with open(args.crosswalk) as f:
            raw = json.load(f)
        crosswalk = {k: v for k, v in raw.items() if not k.startswith("_")}

    api = HGNNInference.from_run_dir(args.run_dir, device=args.device)
    model_leaves = set(api.leaf_names)

    def accepted_for(gt_label: str) -> Optional[List[str]]:
        """Accepted predicted leaf names for a GT label, or None if uncovered."""
        if crosswalk is None:
            return [gt_label] if gt_label in model_leaves else None
        mapped = crosswalk.get(gt_label)
        if not mapped:
            return None
        return [m for m in mapped if m in model_leaves] or None

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(x, **k):
            return x

    # Per-class tallies keyed by GT MINC category.
    cls_total: Dict[str, int] = defaultdict(int)      # covered samples per class
    cls_top1: Dict[str, int] = defaultdict(int)
    cls_top3: Dict[str, int] = defaultdict(int)
    uncovered: Dict[str, int] = defaultdict(int)      # samples skipped (no mapping)
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    bs = args.batch_size
    t0 = time.perf_counter()
    for start in tqdm(range(0, len(samples), bs), desc="eval", unit="batch"):
        chunk = samples[start : start + bs]
        images = [str(p) for p, _ in chunk]
        results = api.infer_batch(images, return_probs=True, decode_path=False)
        for (_, gt_label), res in zip(chunk, results):
            accepted = accepted_for(gt_label)
            if accepted is None:
                uncovered[gt_label] += 1
                continue
            cls_total[gt_label] += 1

            pred = res["leaf_label"]
            confusion[gt_label][pred] += 1
            if pred in accepted:
                cls_top1[gt_label] += 1

            probs = res["leaf_probs"]
            top3_idx = np.argsort(probs)[::-1][:3]
            top3 = {api.leaf_names[i] for i in top3_idx}
            if top3 & set(accepted):
                cls_top3[gt_label] += 1
    elapsed = time.perf_counter() - t0

    covered_classes = sorted(cls_total)
    n_covered = sum(cls_total.values())
    n_uncovered = sum(uncovered.values())
    overall_top1 = sum(cls_top1.values()) / n_covered if n_covered else 0.0
    overall_top3 = sum(cls_top3.values()) / n_covered if n_covered else 0.0
    per_class = {
        c: {
            "n": cls_total[c],
            "top1": cls_top1[c] / cls_total[c] if cls_total[c] else 0.0,
            "top3": cls_top3[c] / cls_total[c] if cls_total[c] else 0.0,
        }
        for c in covered_classes
    }
    mean_class_top1 = (
        float(np.mean([per_class[c]["top1"] for c in covered_classes]))
        if covered_classes else 0.0
    )

    metrics = {
        "model_name": args.name or Path(args.run_dir).name,
        "run_dir": str(args.run_dir),
        "label_space": "minc-native" if crosswalk is None else "matador-via-crosswalk",
        "crosswalk": str(args.crosswalk) if args.crosswalk else None,
        "n_per_class": args.n_per_class,
        "n_samples_total": len(samples),
        "n_samples_covered": n_covered,
        "n_samples_uncovered": n_uncovered,
        "n_classes_covered": len(covered_classes),
        "uncovered_classes": sorted(uncovered),
        "overall_top1": overall_top1,
        "overall_top3": overall_top3,
        "mean_class_top1": mean_class_top1,
        "per_class": per_class,
        "confusion": {g: dict(p) for g, p in confusion.items()},
        "elapsed_sec": elapsed,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _print_summary(metrics)
    _plot_per_class(metrics, out_dir / "per_class_accuracy.png")
    print(f"\nWrote {out_dir/'metrics.json'} and {out_dir/'per_class_accuracy.png'}")
    print(f"Eval time: {elapsed:.1f}s for {n_covered} covered patches")
    return metrics


def _print_summary(m: Dict) -> None:
    print(f"\n=== {m['model_name']}  ({m['label_space']}) ===")
    print(f"  covered patches : {m['n_samples_covered']} / {m['n_samples_total']}")
    print(f"  classes covered : {m['n_classes_covered']}")
    if m["uncovered_classes"]:
        print(f"  uncovered classes (excluded): {', '.join(m['uncovered_classes'])}")
    print(f"  top-1 accuracy  : {m['overall_top1']:.3f}")
    print(f"  top-3 accuracy  : {m['overall_top3']:.3f}")
    print(f"  mean-class top-1: {m['mean_class_top1']:.3f}")


def _plot_per_class(m: Dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(m["per_class"])
    vals = [m["per_class"][c]["top1"] for c in classes]
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.5), 4.5))
    ax.bar(classes, vals, color="#4c72b0")
    ax.axhline(m["overall_top1"], color="crimson", ls="--", lw=1,
               label=f"overall top-1 = {m['overall_top1']:.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("top-1 accuracy")
    ax.set_title(f"{m['model_name']} — per-MINC-class accuracy ({m['label_space']})")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def compare(metric_paths: List[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = []
    for p in metric_paths:
        with open(p) as f:
            models.append(json.load(f))

    all_classes = sorted({c for m in models for c in m["per_class"]})
    x = np.arange(len(all_classes))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(max(8, len(all_classes) * 0.55), 5.5))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for i, m in enumerate(models):
        vals = [m["per_class"].get(c, {}).get("top1", np.nan) for c in all_classes]
        ax.bar(x + i * width, vals, width,
               label=f"{m['model_name']} (top-1={m['overall_top1']:.3f})",
               color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(all_classes, rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("MINC-2500 patch classification — per-class accuracy")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    print("\n=== comparison ===")
    for m in models:
        print(f"  {m['model_name']:30s} top-1={m['overall_top1']:.3f}  "
              f"top-3={m['overall_top3']:.3f}  mean-class={m['mean_class_top1']:.3f}  "
              f"({m['n_samples_covered']} patches, {m['label_space']})")
    print(f"\nWrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs="+", default=None,
                        help="compare mode: paths to >=2 metrics.json files (skips eval)")
    parser.add_argument("--run-dir", type=Path, default=None, help="HGNN run directory")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--images-root", type=str, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--crosswalk", type=str, default=None,
                        help="GT-label -> accepted-leaf-names JSON (for cross-taxonomy models)")
    parser.add_argument("--n-per-class", type=int, default=100,
                        help="balanced samples per MINC category (controls runtime)")
    parser.add_argument("--limit", type=int, default=None, help="hard cap on total patches")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--name", type=str, default=None, help="label for plots/summary")
    parser.add_argument("--out", type=Path, default=Path("out/minc_eval/run"))
    args = parser.parse_args()

    if args.compare:
        if len(args.compare) < 2:
            parser.error("--compare needs >=2 metrics.json paths")
        out = args.out if str(args.out).endswith(".png") else args.out / "comparison.png"
        compare(args.compare, out)
        return

    if args.run_dir is None:
        parser.error("--run-dir is required (or use --compare)")
    evaluate(args)


if __name__ == "__main__":
    main()
