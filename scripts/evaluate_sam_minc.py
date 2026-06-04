#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from segmentation.sam_baseline import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/minc/minc-s"))
    parser.add_argument("--photos-dir", type=Path, default=None)
    parser.add_argument("--segments-dir", type=Path, default=None)
    parser.add_argument("--segments-txt", type=Path, default=None)
    parser.add_argument("--categories", type=Path, default=None)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "point", "box", "box_point"], default="auto")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alignment-strategy",
        choices=["resize_sam_to_gt", "resize_photo_to_mask"],
        default="resize_sam_to_gt",
    )
    parser.add_argument("--cache-auto-masks", action="store_true")
    parser.add_argument("--auto-points-per-side", type=int, default=None)
    parser.add_argument("--visualize-first-n", type=int, default=0)
    parser.add_argument("--compute-boundary-f", action="store_true")
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--num-negative-points", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available. Re-run with --device cpu.")

    payload = run_evaluation(args)
    metrics = payload["metrics"]["overall"]
    print(f"Mode: {payload['mode']}")
    print(f"Model: {payload['model_type']}")
    print(f"Alignment strategy: {payload['alignment_strategy']}")
    print(f"Valid photos: {payload['dataset_summary']['valid_photos']}")
    print(f"Valid segments: {payload['dataset_summary']['valid_segments']}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
