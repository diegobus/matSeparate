#!/usr/bin/env python3
"""
CLI for the material segmentation / merging pipeline.

Examples:
    # Most detailed (leaf) segmentation:
    python scripts/segment_image.py \
        --run-dir runs/c1_hgnn_baseline/avg_init_20260529_004313 \
        --image scene.jpg --out out/scene

    # Coarser: biotic vs abiotic (taxonomy depth 2):
    python scripts/segment_image.py \
        --run-dir runs/c1_hgnn_baseline/avg_init_20260529_004313 \
        --image scene.jpg --out out/scene_l2 --level 2

    # Use a config file for all knobs (run_dir, CRF params, thresholds, ...):
    python scripts/segment_image.py --config configs/segmentation.yaml --image scene.jpg

    # No checkpoint yet? Validate the plumbing + visuals with a stub classifier:
    python scripts/segment_image.py --image scene.jpg --out out/demo --stub --viz out/demo/panel.png
"""

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from segmentation.config import SegmentationConfig
from segmentation.pipeline import MaterialMerger

DEFAULT_TAXONOMY = _repo_root / "taxonomy" / "assets" / "matador-c1-taxonomy.json"


def _parse_level(value: str):
    if value is None:
        return None
    if value.lower() == "leaf":
        return "leaf"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--level must be 'leaf' or an integer depth")


def main():
    parser = argparse.ArgumentParser(description="Material segmentation pipeline.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out/segmentation"))
    parser.add_argument("--config", type=Path, default=None, help="optional YAML config")
    parser.add_argument("--run-dir", type=Path, default=None, help="HGNN run dir (overrides config)")
    parser.add_argument("--level", type=_parse_level, default=None, help="'leaf' or int depth")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | mps | auto")
    parser.add_argument("--sampler", type=str, default=None, choices=["grid", "sliding"],
                        help="patch sampler (sliding = finer)")
    parser.add_argument("--window-size", type=int, default=None, help="sliding window size (px)")
    parser.add_argument("--stride", type=int, default=None, help="sliding window stride (px)")
    parser.add_argument(
        "--min-patches",
        type=int,
        default=None,
        help="lower bound on patches per image (stride auto-adapts; sliding sampler only)",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="upper bound on patches per image (stride auto-adapts; sliding sampler only)",
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=None,
        help="max-prob below this -> background/unknown. Use 0 to label every pixel "
        "(recommended for mask-similarity comparison).",
    )
    parser.add_argument(
        "--no-taxonomy-compat",
        action="store_true",
        help="disable taxonomy-aware CRF label compatibility (use scalar Potts instead)",
    )
    parser.add_argument("--viz", type=Path, default=None, help="save a composite panel figure here")
    parser.add_argument(
        "--compare-levels",
        type=str,
        default=None,
        help="comma-separated levels to compare side by side, e.g. 'leaf,3,2'",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="use a checkpoint-free stub classifier (validates plumbing/visuals only)",
    )
    args = parser.parse_args()

    if args.config:
        config = SegmentationConfig.from_yaml(args.config)
    else:
        config = SegmentationConfig()

    if args.run_dir is not None:
        config.run_dir = str(args.run_dir)
    if args.device is not None:
        config.device = args.device
    if args.sampler is not None:
        config.sampling.type = args.sampler
    if args.window_size is not None:
        config.sampling.window_size = args.window_size
    if args.stride is not None:
        config.sampling.stride = args.stride
    if args.min_patches is not None:
        config.sampling.min_patches = args.min_patches
    if args.max_patches is not None:
        config.sampling.max_patches = args.max_patches
    if args.bg_threshold is not None:
        config.objects.bg_threshold = args.bg_threshold
    if args.no_taxonomy_compat:
        config.crf.taxonomy_aware_compat = False

    if args.stub:
        merger = _build_stub_merger(config)
    else:
        if config.run_dir is None:
            parser.error("a run_dir is required (via --run-dir or the config file), or pass --stub")
        merger = MaterialMerger.from_run_dir(
            config.run_dir, config=config, device=config.device
        )

    import time

    t0 = time.perf_counter()
    result = merger.segment(args.image, level=args.level)
    elapsed = time.perf_counter() - t0
    written = result.save(args.out)

    print(f"Done in {elapsed:.1f}s")
    print(f"Level: {result.level}")
    print(f"Classes ({len(result.frontier_names)}): {', '.join(result.frontier_names)}")
    print(f"Objects: {len(result.instances)}")
    for key, path in written.items():
        print(f"  {key}: {path}")

    if args.viz is not None:
        from segmentation.visualize import save_panel

        panel = save_panel(args.viz, result.image_uint8, result)
        print(f"  panel: {panel}")

    if args.compare_levels:
        from segmentation.visualize import save_level_comparison

        levels = [_parse_level(tok.strip()) for tok in args.compare_levels.split(",")]
        cmp_path = args.out / "levels_comparison.png"
        cmp = save_level_comparison(cmp_path, result.image_uint8, result, levels)
        print(f"  levels_comparison: {cmp}")


def _build_stub_merger(config: SegmentationConfig) -> MaterialMerger:
    """Build a MaterialMerger with a stub classifier (no checkpoint needed)."""
    from segmentation.classify import PatchClassifier, StubLeafPredictor
    from taxonomy.tree import get_taxonomy

    graph = get_taxonomy(str(DEFAULT_TAXONOMY))
    leaf_names = sorted(n for n in graph.nodes() if graph.out_degree(n) == 0)
    classifier = PatchClassifier(StubLeafPredictor(leaf_names))
    if config.crf.backend == "dense":
        config.crf.backend = "none"  # stub demo: skip native CRF dependency
    return MaterialMerger(classifier=classifier, graph=graph, config=config)


if __name__ == "__main__":
    main()
