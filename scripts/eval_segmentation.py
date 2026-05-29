#!/usr/bin/env python3
"""
Evaluate predicted material segmentations against ground-truth label maps.

Prediction layout (as written by ``scripts/segment_image.py`` / ``result.save``):

    pred_dir/
      <image_key>/
        label_map.png      # single-channel class ids
        labels.json        # {"level": ..., "legend": {"0": "background", "1": "...", ...}}

Ground truth: a directory of single-channel label PNGs named ``<image_key>.png`` plus a
legend JSON mapping ``id -> name`` (``--gt-legend``). If the GT vocabulary differs from our
taxonomy (e.g. MINC's 23 categories), pass ``--crosswalk`` mapping ``pred_name -> gt_name``;
only names shared by both legends are scored, everything else is ignored.

Examples:
    python scripts/eval_segmentation.py \
        --pred-dir out/ --gt-dir data/gt_masks --gt-legend data/gt_legend.json \
        --out out/metrics.json --confusion out/confusion.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from segmentation.metrics import (  # noqa: E402
    build_confusion,
    build_shared_space,
    metrics_from_confusion,
    remap_label_map,
)


def _load_label_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.int64)


def _resize_nearest(arr: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    img = Image.fromarray(arr.astype(np.int32), mode="I")
    img = img.resize((w, h), resample=Image.NEAREST)
    return np.array(img, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Evaluate material segmentations.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="parent of per-image output dirs")
    parser.add_argument("--gt-dir", type=Path, required=True, help="dir of <key>.png GT label maps")
    parser.add_argument("--gt-legend", type=Path, required=True, help="json id->name for GT")
    parser.add_argument("--crosswalk", type=Path, default=None, help="json pred_name->gt_name")
    parser.add_argument("--out", type=Path, default=None, help="write metrics json here")
    parser.add_argument("--confusion", type=Path, default=None, help="save confusion heatmap png")
    parser.add_argument("--keep-background", action="store_true", help="score background too")
    args = parser.parse_args()

    gt_legend = json.loads(args.gt_legend.read_text())
    crosswalk = json.loads(args.crosswalk.read_text()) if args.crosswalk else None
    ignore_ids = () if args.keep_background else (0,)

    pred_subdirs = sorted(d for d in args.pred_dir.iterdir() if (d / "label_map.png").exists())
    if not pred_subdirs:
        parser.error(f"no prediction subdirs with label_map.png found under {args.pred_dir}")

    # Shared space from the first prediction legend + GT legend (assumed consistent).
    first_legend = json.loads((pred_subdirs[0] / "labels.json").read_text())["legend"]
    class_names, pred_remap, gt_remap = build_shared_space(first_legend, gt_legend, crosswalk)
    num_classes = len(class_names)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    matched, skipped = 0, []
    for sub in pred_subdirs:
        key = sub.name
        gt_path = args.gt_dir / f"{key}.png"
        if not gt_path.exists():
            skipped.append(key)
            continue
        pred = remap_label_map(_load_label_png(sub / "label_map.png"), pred_remap)
        gt = remap_label_map(_load_label_png(gt_path), gt_remap)
        if gt.shape != pred.shape:
            gt = remap_label_map(
                _resize_nearest(_load_label_png(gt_path), pred.shape), gt_remap
            )
        cm += build_confusion(pred, gt, num_classes, ignore_ids=ignore_ids)
        matched += 1

    if matched == 0:
        parser.error("no prediction/GT pairs matched by filename key")

    metrics = metrics_from_confusion(cm, class_names, ignore_ids=ignore_ids)

    print(f"Matched {matched} image(s); skipped {len(skipped)} without GT.")
    print(f"mean IoU      : {metrics.mean_iou:.4f}")
    print(f"mean class acc: {metrics.mean_acc:.4f}")
    print(f"pixel acc     : {metrics.pixel_acc:.4f}")
    print(f"classes scored: {metrics.num_classes_present}")
    print("\nper-class IoU:")
    for name, iou in sorted(metrics.per_class_iou.items(), key=lambda kv: -kv[1]):
        print(f"  {name:20s} {iou:.4f}")

    if args.out:
        payload = metrics.to_dict()
        payload["matched"] = matched
        payload["skipped"] = skipped
        payload["class_names"] = class_names
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote metrics -> {args.out}")

    if args.confusion:
        _save_confusion(cm, class_names, args.confusion, ignore_ids)
        print(f"Wrote confusion -> {args.confusion}")


def _save_confusion(cm, class_names, path, ignore_ids):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = [i for i in range(len(class_names)) if i not in ignore_ids and cm[i].sum() > 0]
    if not keep:
        return
    sub = cm[np.ix_(keep, keep)].astype(np.float64)
    row = sub.sum(axis=1, keepdims=True)
    norm = sub / np.where(row > 0, row, 1)
    names = [class_names[i] for i in keep]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.5),) * 2)
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
