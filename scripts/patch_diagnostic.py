#!/usr/bin/env python3
"""
Patch-level classifier diagnostic.

Tiles an image into a non-overlapping grid, runs the HGNN patch classifier on each tile,
and renders a montage annotated with the predicted leaf material + confidence. This
separates the *classifier* quality from the *merging* algorithm: if individual tiles are
misclassified here, the bottleneck is the classifier/domain, not the segmentation pipeline.

Example:
    python scripts/patch_diagnostic.py \
        --run-dir runs/c1_hgnn_baseline/avg_init_20260529_004313 \
        --image test_images/minc_1.jpg --window-size 64 --out out/diag.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.infer_api import HGNNInference  # noqa: E402
from segmentation.patches import GridSampler  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Per-patch classifier diagnostic montage.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=Path("out/patch_diagnostic.png"))
    args = parser.parse_args()

    api = HGNNInference.from_run_dir(args.run_dir, device=args.device)
    image = np.array(Image.open(args.image).convert("RGB"), dtype=np.uint8)

    sample = GridSampler(args.window_size).sample(image)
    gh, gw = sample.grid_shape

    import time

    t0 = time.perf_counter()
    results = api.infer_batch(
        [sample.patches[i] for i in range(len(sample.grid_coords))],
        return_probs=True, decode_path=False,
    )
    print(f"classified {len(results)} patches in {time.perf_counter() - t0:.1f}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(gh, gw, figsize=(gw * 1.6, gh * 1.7))
    axes = np.atleast_2d(axes)
    for (gi, gj), patch, res in zip(sample.grid_coords, sample.patches, results):
        ax = axes[gi, gj]
        ax.imshow(patch)
        conf = float(np.max(res["leaf_probs"]))
        ax.set_title(f"{res['leaf_label']}\n{conf:.2f}", fontsize=7)
        ax.axis("off")
    # blank any unused axes
    for gi in range(gh):
        for gj in range(gw):
            if (gi, gj) not in sample.grid_coords:
                axes[gi, gj].axis("off")

    fig.suptitle(f"patch predictions (window={args.window_size}px, grid {gh}x{gw})", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")

    # quick text summary of label distribution
    from collections import Counter
    counts = Counter(r["leaf_label"] for r in results)
    print("predicted label distribution:")
    for name, c in counts.most_common():
        print(f"  {name:16s} {c}")


if __name__ == "__main__":
    main()
