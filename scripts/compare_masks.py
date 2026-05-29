#!/usr/bin/env python3
"""
Class-agnostic comparison of our segmentation mask to a reference mask (e.g. MINC GT).

Only the *partition* of pixels matters here -- the label values are ignored -- so this
answers "how similar are the masks?" regardless of whether the material classes match.

Metrics:
  - Adjusted Rand Index (1 = identical grouping, 0 = chance)
  - Variation of Information (bits; 0 = identical, lower better)
  - Segmentation Covering (both directions, [0,1], higher better)
  - Mean best-IoU (avg over GT regions of best-matching predicted region)
  - Boundary F1 (boundary agreement within a pixel tolerance)

Example:
    python scripts/compare_masks.py \
        --pred out/scene_w64/label_map.png \
        --ref  data/minc_gt/minc_1.png \
        --connected --viz out/scene_w64/compare.png --out out/scene_w64/agreement.json
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

from segmentation.metrics import mask_agreement, to_connected_components  # noqa: E402


def _load_label(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:  # color-coded mask -> collapse to unique-color ids
        flat = arr.reshape(-1, arr.shape[2])
        _, inv = np.unique(flat, axis=0, return_inverse=True)
        arr = inv.reshape(arr.shape[:2])
    return arr.astype(np.int64)


def _resize_nearest(arr: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    img = Image.fromarray(arr.astype(np.int32), mode="I").resize((w, h), Image.NEAREST)
    return np.array(img, dtype=np.int64)


def _random_palette(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pal = rng.integers(40, 255, size=(n + 1, 3), dtype=np.uint8)
    pal[0] = (0, 0, 0)
    return pal


def _colorize(label_map: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(label_map, return_inverse=True)
    pal = _random_palette(len(uniq))
    return pal[inv.reshape(label_map.shape) + 0]


def main():
    parser = argparse.ArgumentParser(description="Class-agnostic mask comparison.")
    parser.add_argument("--pred", type=Path, required=True, help="our label_map.png")
    parser.add_argument("--ref", type=Path, required=True, help="reference mask (e.g. MINC GT)")
    parser.add_argument("--connected", action="store_true",
                        help="split labels into connected components (compare regions, not classes)")
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--boundary-tol", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None, help="write agreement json")
    parser.add_argument("--viz", type=Path, default=None, help="save side-by-side png")
    args = parser.parse_args()

    pred = _load_label(args.pred)
    ref = _load_label(args.ref)
    if ref.shape != pred.shape:
        print(f"resizing ref {ref.shape} -> pred {pred.shape} (nearest)")
        ref = _resize_nearest(ref, pred.shape)

    ag = mask_agreement(
        pred, ref, connected=args.connected,
        connectivity=args.connectivity, boundary_tolerance=args.boundary_tol,
    )

    print(f"mode: {'connected-components (regions)' if args.connected else 'label partitions'}")
    print(f"GT regions: {ag.num_gt_regions}   pred regions: {ag.num_pred_regions}")
    print(f"Adjusted Rand Index      : {ag.adjusted_rand_index:.4f}   (1=identical, 0=chance)")
    print(f"Variation of Information : {ag.variation_of_information:.4f} bits (0=identical)")
    print(f"Covering (GT by pred)    : {ag.covering_gt_by_pred:.4f}")
    print(f"Covering (pred by GT)    : {ag.covering_pred_by_gt:.4f}")
    print(f"Mean best IoU            : {ag.mean_best_iou:.4f}")
    print(f"Boundary F1 (tol={args.boundary_tol}px)    : {ag.boundary_f1:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(ag.to_dict(), indent=2))
        print(f"wrote {args.out}")

    if args.viz:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pv = to_connected_components(pred, args.connectivity) if args.connected else pred
        rv = to_connected_components(ref, args.connectivity) if args.connected else ref
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
        axes[0].imshow(_colorize(pv)); axes[0].set_title("ours"); axes[0].axis("off")
        axes[1].imshow(_colorize(rv)); axes[1].set_title("reference (MINC)"); axes[1].axis("off")
        fig.tight_layout()
        args.viz.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.viz, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {args.viz}")


if __name__ == "__main__":
    main()
