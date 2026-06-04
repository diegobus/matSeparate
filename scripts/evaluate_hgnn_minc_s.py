#!/usr/bin/env python3
"""Evaluate HGNN material segmentation components on MINC-S binary segments.

This mirrors the SAM automatic-proposal protocol: run the segmenter once per
image, treat connected components as mask proposals, and score each MINC-S
ground-truth segment against the best-overlapping predicted component.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from segmentation.config import SegmentationConfig  # noqa: E402
from segmentation.pipeline import MaterialMerger  # noqa: E402


DEFAULT_MATADOR_TO_MINC = {
    "brick": "brick",
    "carpet": "carpet",
    "pottery": "ceramic",
    "natural_fiber": "fabric",
    "wool": "fabric",
    "satin": "fabric",
    "nylon": "fabric",
    "suede": "fabric",
    "foliage": "foliage",
    "grass": "foliage",
    "ivy": "foliage",
    "moss": "foliage",
    "bread": "food",
    "fruit": "food",
    "vegetable": "food",
    "leather": "leather",
    "generic_metal": "metal",
    "paper": "paper",
    "foam": "plastic",
    "wax": "plastic",
    "marble": "polishedstone",
    "granite": "stone",
    "limestone": "stone",
    "shale": "stone",
    "gravel": "stone",
    "sand": "stone",
    "timber": "wood",
    "tree_bark": "wood",
}


@dataclass
class SegmentSample:
    label_index: int
    label_name: str
    photo_id: str
    shape_id: str
    photo_path: Path
    mask_path: Path


@dataclass
class DatasetSummary:
    total_rows: int
    blank_rows: int
    invalid_rows: int
    missing_photos: int
    missing_masks: int
    valid_segments: int
    valid_photos: int
    skipped_unreadable_photos: int = 0
    skipped_unreadable_masks: int = 0


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_categories(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_photo_path(photos_dir: Path, photo_id: str) -> Path | None:
    photo_path = photos_dir / f"{photo_id}.jpg"
    return photo_path if photo_path.exists() else None


def parse_test_segments(
    segments_txt: Path,
    categories: list[str],
    photos_dir: Path,
    segments_dir: Path,
    max_images: int | None = None,
) -> tuple[dict[str, list[SegmentSample]], DatasetSummary, list[str]]:
    grouped: dict[str, list[SegmentSample]] = defaultdict(list)
    warnings: list[str] = []
    total_rows = 0
    blank_rows = 0
    invalid_rows = 0
    missing_photos = 0
    missing_masks = 0

    with open(segments_txt, newline="") as handle:
        for raw_line in handle:
            total_rows += 1
            line = raw_line.strip()
            if not line:
                blank_rows += 1
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                invalid_rows += 1
                warnings.append(f"Invalid row format: {line}")
                continue
            label_str, photo_id, shape_id = parts
            try:
                label_index = int(label_str)
            except ValueError:
                invalid_rows += 1
                warnings.append(f"Invalid label index: {line}")
                continue
            if label_index < 0 or label_index >= len(categories):
                invalid_rows += 1
                warnings.append(f"Label index out of range: {line}")
                continue
            photo_path = resolve_photo_path(photos_dir, photo_id)
            if photo_path is None:
                missing_photos += 1
                warnings.append(f"Missing photo for photo_id={photo_id}")
                continue
            mask_path = segments_dir / f"{photo_id}_{shape_id}.png"
            if not mask_path.exists():
                missing_masks += 1
                warnings.append(f"Missing mask for photo_id={photo_id}, shape_id={shape_id}")
                continue
            grouped[photo_id].append(
                SegmentSample(
                    label_index=label_index,
                    label_name=categories[label_index],
                    photo_id=photo_id,
                    shape_id=shape_id,
                    photo_path=photo_path,
                    mask_path=mask_path,
                )
            )

    photo_ids = sorted(grouped.keys())
    if max_images is not None:
        photo_ids = photo_ids[:max_images]
        grouped = {photo_id: grouped[photo_id] for photo_id in photo_ids}

    summary = DatasetSummary(
        total_rows=total_rows,
        blank_rows=blank_rows,
        invalid_rows=invalid_rows,
        missing_photos=missing_photos,
        missing_masks=missing_masks,
        valid_segments=sum(len(v) for v in grouped.values()),
        valid_photos=len(grouped),
    )
    return grouped, summary, warnings


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.array(image)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def resize_image(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    pil = Image.fromarray(image)
    return np.array(pil.resize((width, height), Image.Resampling.BILINEAR))


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return float((2.0 * intersection) / denom)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_crosswalk(path: Path | None, use_default: bool) -> dict[str, str]:
    if path is not None:
        return json.loads(path.read_text())
    return dict(DEFAULT_MATADOR_TO_MINC) if use_default else {}


def build_config(args: argparse.Namespace) -> SegmentationConfig:
    config = SegmentationConfig.from_yaml(args.config) if args.config else SegmentationConfig()
    config.run_dir = str(args.run_dir)
    config.device = args.device
    config.sampling.type = args.sampler
    config.sampling.batch_size = args.batch_size
    if args.window_size is not None:
        config.sampling.window_size = args.window_size
    if args.stride is not None:
        config.sampling.stride = args.stride
    if args.min_patches is not None:
        config.sampling.min_patches = args.min_patches
    if args.max_patches is not None:
        config.sampling.max_patches = args.max_patches
    config.crf.backend = args.crf_backend
    config.level.target = args.level
    config.objects.bg_threshold = args.bg_threshold
    config.objects.min_object_area = args.min_object_area
    config.output.write_color_viz = False
    config.output.write_instance_pngs = False
    return config


def component_candidates(result) -> list[dict[str, Any]]:
    candidates = []
    for index, inst in enumerate(result.instances):
        candidates.append(
            {
                "index": index,
                "mask": inst.mask.astype(bool),
                "area": int(inst.area),
                "material": inst.material,
                "category_id": int(inst.category_id),
                "score": float(inst.score),
                "bbox": list(inst.bbox),
            }
        )
    return candidates


def best_match(gt_mask: np.ndarray, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {
        "best_component_index": -1,
        "best_iou": 0.0,
        "best_dice": 0.0,
        "matched_pred_area": 0,
        "matched_label": "",
        "matched_category_id": -1,
        "matched_confidence_mean": 0.0,
        "matched_bbox": [],
    }
    for cand in candidates:
        pred = cand["mask"]
        iou = compute_iou(pred, gt_mask)
        if iou > best["best_iou"]:
            best = {
                "best_component_index": cand["index"],
                "best_iou": iou,
                "best_dice": compute_dice(pred, gt_mask),
                "matched_pred_area": cand["area"],
                "matched_label": cand["material"],
                "matched_category_id": cand["category_id"],
                "matched_confidence_mean": cand["score"],
                "matched_bbox": cand["bbox"],
            }
    return best


def aggregate(segment_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> dict[str, Any]:
    recalls = {}
    for threshold in (0.25, 0.50, 0.75):
        recalls[f"recall@{threshold:.2f}"] = (
            float(np.mean([row["best_iou"] >= threshold for row in segment_rows]))
            if segment_rows
            else 0.0
        )

    per_class = {}
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        by_class[row["label_name"]].append(row)
    for name, rows in sorted(by_class.items()):
        per_class[name] = {
            "num_segments": len(rows),
            "mean_best_iou": float(np.mean([row["best_iou"] for row in rows])),
            "recall@0.25": float(np.mean([row["best_iou"] >= 0.25 for row in rows])),
            "recall@0.50": float(np.mean([row["best_iou"] >= 0.50 for row in rows])),
            "recall@0.75": float(np.mean([row["best_iou"] >= 0.75 for row in rows])),
        }

    mapped = [row for row in segment_rows if row["gt_is_mapped"]]
    mapped_with_match = [row for row in mapped if row["best_component_index"] >= 0]
    semantic = {
        "num_mapped_segments": len(mapped),
        "num_unmapped_segments": len(segment_rows) - len(mapped),
        "mapped_segment_accuracy": (
            float(np.mean([row["matched_maps_to_gt"] for row in mapped_with_match]))
            if mapped_with_match
            else 0.0
        ),
    }

    return {
        "overall": {
            "num_images": len({row["photo_id"] for row in image_rows}),
            "num_image_contexts": len(image_rows),
            "num_segments": len(segment_rows),
            "mean_best_iou": (
                float(np.mean([row["best_iou"] for row in segment_rows]))
                if segment_rows
                else 0.0
            ),
            "mean_best_dice": (
                float(np.mean([row["best_dice"] for row in segment_rows]))
                if segment_rows
                else 0.0
            ),
            "mean_num_components_per_image": (
                float(np.mean([row["num_hgnn_components"] for row in image_rows]))
                if image_rows
                else 0.0
            ),
            "mean_inference_time_sec_per_image": (
                float(np.mean([row["inference_time_sec"] for row in image_rows]))
                if image_rows
                else 0.0
            ),
            **recalls,
        },
        "per_class": per_class,
        "semantic_mapped": semantic,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    photos_dir = args.photos_dir or args.dataset_root / "photos"
    segments_dir = args.segments_dir or args.dataset_root / "segments"
    segments_txt = args.segments_txt or args.dataset_root / "test-segments.txt"
    categories_path = args.categories or args.dataset_root / "categories.txt"

    categories = load_categories(categories_path)
    grouped, dataset_summary, warnings = parse_test_segments(
        segments_txt=segments_txt,
        categories=categories,
        photos_dir=photos_dir,
        segments_dir=segments_dir,
        max_images=args.max_images,
    )
    crosswalk = load_crosswalk(args.crosswalk, args.default_crosswalk)
    gt_mapped_labels = set(crosswalk.values())

    config = build_config(args)
    if args.predictor == "hgnn":
        merger = MaterialMerger.from_run_dir(args.run_dir, config=config, device=args.device)
    elif args.predictor == "global_resnet":
        merger = MaterialMerger.from_global_resnet_run_dir(
            args.run_dir,
            taxonomy_json=args.taxonomy_json,
            config=config,
            device=args.device,
            context_mode=args.context_mode,
            context_scale=args.context_scale,
        )
    else:
        merger = MaterialMerger.from_global_hgnn_run_dir(
            args.run_dir,
            config=config,
            device=args.device,
            context_mode=args.context_mode,
            context_scale=args.context_scale,
        )

    image_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    eval_meta = {"skipped_unreadable_photos": 0, "skipped_unreadable_masks": 0}

    photo_items = list(grouped.items())
    for image_index, (photo_id, samples) in enumerate(photo_items, start=1):
        try:
            photo = load_rgb_image(samples[0].photo_path)
        except OSError:
            eval_meta["skipped_unreadable_photos"] += 1
            continue

        masks_by_shape: dict[tuple[int, int], dict[str, np.ndarray]] = defaultdict(dict)
        valid_by_shape: dict[tuple[int, int], list[SegmentSample]] = defaultdict(list)
        for sample in samples:
            try:
                gt_mask = load_binary_mask(sample.mask_path)
            except OSError:
                eval_meta["skipped_unreadable_masks"] += 1
                continue
            masks_by_shape[gt_mask.shape][sample.shape_id] = gt_mask
            valid_by_shape[gt_mask.shape].append(sample)

        for mask_shape, shape_samples in valid_by_shape.items():
            working_photo = resize_image(photo, mask_shape)
            start = time.perf_counter()
            result = merger.segment(working_photo, level=args.level)
            inference_time = time.perf_counter() - start
            candidates = component_candidates(result)

            per_image_ious = []
            for sample in shape_samples:
                gt_mask = masks_by_shape[mask_shape][sample.shape_id]
                if gt_mask.sum() == 0:
                    continue
                best = best_match(gt_mask, candidates)
                matched_minc_label = crosswalk.get(best["matched_label"], "")
                gt_is_mapped = sample.label_name in gt_mapped_labels
                matched_maps_to_gt = bool(matched_minc_label == sample.label_name)
                per_image_ious.append(best["best_iou"])
                segment_rows.append(
                    {
                        "mode": f"{args.predictor}_components",
                        "photo_id": sample.photo_id,
                        "shape_id": sample.shape_id,
                        "label_index": sample.label_index,
                        "label_name": sample.label_name,
                        "mask_path": str(sample.mask_path),
                        "photo_path": str(sample.photo_path),
                        "gt_area": int(gt_mask.sum()),
                        **best,
                        "matched_minc_label": matched_minc_label,
                        "gt_is_mapped": gt_is_mapped,
                        "matched_maps_to_gt": matched_maps_to_gt,
                        "alignment_strategy": "resize_photo_to_mask",
                        "best_component_selection": True,
                        "oracle_component_selection": False,
                    }
                )

            image_rows.append(
                {
                    "photo_id": photo_id,
                    "image_index": image_index,
                    "mask_height": mask_shape[0],
                    "mask_width": mask_shape[1],
                    "num_gt_segments": len(shape_samples),
                    "num_hgnn_components": len(candidates),
                    "num_hgnn_classes_present": int(len(np.unique(result.label_map))),
                    "inference_time_sec": inference_time,
                    "mean_best_iou": float(np.mean(per_image_ious)) if per_image_ious else 0.0,
                    "recall@0.25": float(np.mean([v >= 0.25 for v in per_image_ious])) if per_image_ious else 0.0,
                    "recall@0.50": float(np.mean([v >= 0.50 for v in per_image_ious])) if per_image_ious else 0.0,
                    "recall@0.75": float(np.mean([v >= 0.75 for v in per_image_ious])) if per_image_ious else 0.0,
                    "alignment_strategy": "resize_photo_to_mask",
                }
            )

            if args.save_first_n and image_index <= args.save_first_n:
                save_dir = output_dir / "predictions" / f"{photo_id}_{mask_shape[0]}x{mask_shape[1]}"
                result.save(save_dir)

        if args.progress_every and image_index % args.progress_every == 0:
            print(f"Processed {image_index}/{len(photo_items)} images", flush=True)

    metrics = aggregate(segment_rows, image_rows)
    payload = {
        "mode": f"{args.predictor}_components",
        "model": str(args.run_dir),
        "predictor": args.predictor,
        "context_mode": args.context_mode if args.predictor in {"global_resnet", "global_hgnn"} else None,
        "context_scale": args.context_scale if args.predictor in {"global_resnet", "global_hgnn"} else None,
        "device": args.device,
        "alignment_strategy": "resize_photo_to_mask",
        "level": args.level,
        "sampler": args.sampler,
        "window_size": config.sampling.window_size,
        "stride": config.sampling.stride,
        "crf_backend": config.crf.backend,
        "bg_threshold": config.objects.bg_threshold,
        "min_object_area": config.objects.min_object_area,
        "dataset_summary": {**asdict(dataset_summary), **eval_meta},
        "warning_count": len(warnings),
        "warnings_preview": warnings[:50],
        "crosswalk": crosswalk,
        "metrics": metrics,
    }

    write_csv(image_rows, output_dir / "per_image_metrics.csv")
    write_csv(segment_rows, output_dir / "per_segment_metrics.csv")
    save_json(payload, output_dir / "aggregate_metrics.json")
    return payload


def parse_level(value: str):
    if value.lower() == "leaf":
        return "leaf"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--level must be 'leaf' or an integer depth") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/minc/minc-s"))
    parser.add_argument("--photos-dir", type=Path, default=None)
    parser.add_argument("--segments-dir", type=Path, default=None)
    parser.add_argument("--segments-txt", type=Path, default=None)
    parser.add_argument("--categories", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--predictor", choices=["hgnn", "global_resnet", "global_hgnn"], default="hgnn")
    parser.add_argument(
        "--taxonomy-json",
        type=Path,
        default=Path("taxonomy/assets/matador-c1-taxonomy.json"),
        help="Required for global_resnet predictor because its run config has no graph.",
    )
    parser.add_argument(
        "--context-mode",
        choices=["full_image", "scaled_window"],
        default="scaled_window",
        help="Context crop strategy for global_resnet predictor.",
    )
    parser.add_argument("--context-scale", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--sampler", choices=["grid", "sliding"], default="sliding")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--min-patches", type=int, default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--crf-backend", choices=["dense", "superpixel", "none"], default="superpixel")
    parser.add_argument("--level", type=parse_level, default="leaf")
    parser.add_argument("--bg-threshold", type=float, default=0.0)
    parser.add_argument("--min-object-area", type=int, default=64)
    parser.add_argument("--crosswalk", type=Path, default=None)
    parser.add_argument("--default-crosswalk", action="store_true")
    parser.add_argument("--save-first-n", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = evaluate(args)
    overall = payload["metrics"]["overall"]
    semantic = payload["metrics"]["semantic_mapped"]
    print(f"Mode: {payload['mode']}")
    print(f"Model: {payload['model']}")
    print(f"Images: {overall['num_images']}")
    print(f"Segments: {overall['num_segments']}")
    print(f"Mean best IoU: {overall['mean_best_iou']:.4f}")
    print(f"Recall@0.25: {overall['recall@0.25']:.4f}")
    print(f"Recall@0.50: {overall['recall@0.50']:.4f}")
    print(f"Recall@0.75: {overall['recall@0.75']:.4f}")
    print(f"Mean components/image: {overall['mean_num_components_per_image']:.2f}")
    print(f"Mean sec/image: {overall['mean_inference_time_sec_per_image']:.3f}")
    print(f"Mapped semantic accuracy: {semantic['mapped_segment_accuracy']:.4f}")


if __name__ == "__main__":
    main()
