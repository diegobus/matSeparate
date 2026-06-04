from __future__ import annotations

import csv
import json
import pickle
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries
from tqdm import tqdm


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


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_valid_photo_samples(
    samples: list[SegmentSample],
    eval_meta: dict[str, int],
) -> tuple[np.ndarray | None, dict[str, np.ndarray], list[SegmentSample]]:
    try:
        photo_image = load_rgb_image(samples[0].photo_path)
    except OSError:
        eval_meta["skipped_unreadable_photos"] += 1
        return None, {}, []

    gt_masks: dict[str, np.ndarray] = {}
    valid_samples: list[SegmentSample] = []
    for sample in samples:
        try:
            gt_masks[sample.shape_id] = load_binary_mask(sample.mask_path)
        except OSError:
            eval_meta["skipped_unreadable_masks"] += 1
            continue
        valid_samples.append(sample)
    return photo_image, gt_masks, valid_samples


def resize_image(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    pil = Image.fromarray(image)
    resized = pil.resize((width, height), Image.Resampling.BILINEAR)
    return np.array(resized)


def resize_bool_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    pil = Image.fromarray((mask.astype(np.uint8) * 255))
    resized = pil.resize((width, height), Image.Resampling.NEAREST)
    return np.array(resized) > 0


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


def compute_boundary_fscore(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    pred_boundary = find_boundaries(pred, mode="outer")
    gt_boundary = find_boundaries(gt, mode="outer")
    if not pred_boundary.any() and not gt_boundary.any():
        return 1.0
    if not pred_boundary.any() or not gt_boundary.any():
        return 0.0
    footprint = disk(max(int(tolerance), 1))
    pred_dil = binary_dilation(pred_boundary, footprint)
    gt_dil = binary_dilation(gt_boundary, footprint)
    precision_denom = pred_boundary.sum()
    recall_denom = gt_boundary.sum()
    if precision_denom == 0 or recall_denom == 0:
        return 0.0
    precision = float(np.logical_and(pred_boundary, gt_dil).sum() / precision_denom)
    recall = float(np.logical_and(gt_boundary, pred_dil).sum() / recall_denom)
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def mask_bbox_xyxy(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    return [x0, y0, x1, y1]


def centroid_point_xy(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def any_positive_point_xy(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs[0]), float(ys[0])]


def farthest_point_xy(mask: np.ndarray) -> list[float] | None:
    if mask.sum() == 0:
        return None
    distances = distance_transform_edt(mask)
    flat_index = int(np.argmax(distances))
    y, x = np.unravel_index(flat_index, distances.shape)
    if mask[y, x]:
        return [float(x), float(y)]
    point = centroid_point_xy(mask)
    if point is not None:
        x, y = int(round(point[0])), int(round(point[1]))
        x = min(max(x, 0), mask.shape[1] - 1)
        y = min(max(y, 0), mask.shape[0] - 1)
        if mask[y, x]:
            return [float(x), float(y)]
    return any_positive_point_xy(mask)


def sample_negative_points_xy(mask: np.ndarray, count: int, rng: np.random.Generator) -> list[list[float]]:
    if count <= 0:
        return []
    ring = binary_dilation(mask, disk(5)) & ~mask
    ys, xs = np.nonzero(ring)
    if len(xs) == 0:
        ys, xs = np.nonzero(~mask)
    if len(xs) == 0:
        return []
    take = min(count, len(xs))
    indices = rng.choice(len(xs), size=take, replace=False)
    return [[float(xs[i]), float(ys[i])] for i in indices]


def scale_point_xy(point_xy: list[float], src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> list[float]:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    x, y = point_xy
    return [float(x * dst_w / src_w), float(y * dst_h / src_h)]


def scale_box_xyxy(box_xyxy: list[float], src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> list[float]:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    x0, y0, x1, y1 = box_xyxy
    return [
        float(x0 * dst_w / src_w),
        float(y0 * dst_h / src_h),
        float(x1 * dst_w / src_w),
        float(y1 * dst_h / src_h),
    ]


def serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value)
    return value


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_value(value) for key, value in row.items()})


def save_json(data: dict[str, Any], path: Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    path.write_text(json.dumps(convert(data), indent=2))


def build_sam_model(model_type: str, checkpoint_path: Path, device: str):
    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    sam.eval()
    return sam


def build_auto_mask_generator(sam, points_per_side: int | None = None):
    kwargs: dict[str, Any] = {}
    if points_per_side is not None:
        kwargs["points_per_side"] = points_per_side
    return SamAutomaticMaskGenerator(sam, **kwargs)


def build_predictor(sam):
    return SamPredictor(sam)


def load_cached_auto_masks(cache_path: Path) -> list[dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    with open(cache_path, "rb") as handle:
        return pickle.load(handle)


def save_cached_auto_masks(cache_path: Path, masks: list[dict[str, Any]]) -> None:
    ensure_dir(cache_path.parent)
    with open(cache_path, "wb") as handle:
        pickle.dump(masks, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_auto_cache_path(cache_dir: Path, photo_id: str, image_shape: tuple[int, int], model_type: str) -> Path:
    height, width = image_shape
    return cache_dir / f"{photo_id}__{model_type}__{height}x{width}.pkl"


def compact_auto_masks(raw_masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for index, mask_info in enumerate(raw_masks):
        compacted.append(
            {
                "mask_index": index,
                "segmentation": mask_info["segmentation"].astype(bool),
                "predicted_iou": float(mask_info.get("predicted_iou", 0.0)),
                "stability_score": float(mask_info.get("stability_score", 0.0)),
                "area": int(mask_info.get("area", int(mask_info["segmentation"].sum()))),
                "bbox": [float(v) for v in mask_info.get("bbox", [])],
            }
        )
    return compacted


def best_match_against_candidates(
    gt_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    target_shape: tuple[int, int] | None = None,
    include_boundary_f: bool = False,
    boundary_tolerance: int = 2,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for candidate in candidates:
        pred_mask = candidate["segmentation"]
        if target_shape is not None and pred_mask.shape != target_shape:
            pred_mask_eval = resize_bool_mask(pred_mask, target_shape)
        else:
            pred_mask_eval = pred_mask.astype(bool)
        iou = compute_iou(pred_mask_eval, gt_mask)
        dice = compute_dice(pred_mask_eval, gt_mask)
        boundary_f = None
        if include_boundary_f:
            boundary_f = compute_boundary_fscore(pred_mask_eval, gt_mask, tolerance=boundary_tolerance)
        row = {
            "mask_index": int(candidate.get("mask_index", -1)),
            "iou": float(iou),
            "dice": float(dice),
            "boundary_fscore": None if boundary_f is None else float(boundary_f),
            "pred_area": int(pred_mask_eval.sum()),
            "predicted_iou": float(candidate.get("predicted_iou", 0.0)),
            "stability_score": float(candidate.get("stability_score", 0.0)),
            "bbox": candidate.get("bbox", []),
            "mask_eval": pred_mask_eval,
        }
        if best is None or row["iou"] > best["iou"]:
            best = row
    if best is None:
        empty_mask = np.zeros_like(gt_mask, dtype=bool)
        return {
            "mask_index": -1,
            "iou": 0.0,
            "dice": 0.0,
            "boundary_fscore": 0.0 if include_boundary_f else None,
            "pred_area": 0,
            "predicted_iou": 0.0,
            "stability_score": 0.0,
            "bbox": [],
            "mask_eval": empty_mask,
        }
    return best


def aggregate_auto(segment_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> dict[str, Any]:
    recalls = {}
    for threshold in (0.25, 0.50, 0.75):
        if segment_rows:
            recalls[f"recall@{threshold:.2f}"] = float(
                np.mean([row["best_iou"] >= threshold for row in segment_rows])
            )
        else:
            recalls[f"recall@{threshold:.2f}"] = 0.0
    per_class: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        grouped[row["label_name"]].append(row)
    for label_name, rows in grouped.items():
        per_class[label_name] = {
            "num_segments": len(rows),
            "mean_best_iou": float(np.mean([r["best_iou"] for r in rows])) if rows else 0.0,
            "recall@0.25": float(np.mean([r["best_iou"] >= 0.25 for r in rows])) if rows else 0.0,
            "recall@0.50": float(np.mean([r["best_iou"] >= 0.50 for r in rows])) if rows else 0.0,
            "recall@0.75": float(np.mean([r["best_iou"] >= 0.75 for r in rows])) if rows else 0.0,
        }
    return {
        "overall": {
            "num_images": len(image_rows),
            "num_segments": len(segment_rows),
            "mean_best_iou": float(np.mean([row["best_iou"] for row in segment_rows])) if segment_rows else 0.0,
            "mean_num_sam_masks_per_image": float(np.mean([row["num_sam_masks"] for row in image_rows])) if image_rows else 0.0,
            "mean_inference_time_sec_per_image": float(np.mean([row["inference_time_sec"] for row in image_rows])) if image_rows else 0.0,
            **recalls,
        },
        "per_class": per_class,
    }


def aggregate_prompted(segment_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]], include_boundary_f: bool) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        grouped[row["label_name"]].append(row)
    for label_name, rows in grouped.items():
        class_summary = {
            "num_segments": len(rows),
            "mean_iou": float(np.mean([r["iou"] for r in rows])) if rows else 0.0,
            "mean_dice": float(np.mean([r["dice"] for r in rows])) if rows else 0.0,
        }
        if include_boundary_f:
            class_summary["mean_boundary_fscore"] = float(np.mean([r["boundary_fscore"] for r in rows])) if rows else 0.0
        per_class[label_name] = class_summary
    overall = {
        "num_images": len(image_rows),
        "num_segments": len(segment_rows),
        "mean_iou": float(np.mean([row["iou"] for row in segment_rows])) if segment_rows else 0.0,
        "mean_dice": float(np.mean([row["dice"] for row in segment_rows])) if segment_rows else 0.0,
        "mean_set_image_time_sec_per_image": float(np.mean([row["set_image_time_sec"] for row in image_rows])) if image_rows else 0.0,
        "mean_prompt_inference_time_sec_per_segment": float(np.mean([row["inference_time_sec"] for row in segment_rows])) if segment_rows else 0.0,
    }
    if include_boundary_f:
        overall["mean_boundary_fscore"] = float(np.mean([row["boundary_fscore"] for row in segment_rows])) if segment_rows else 0.0
    return {
        "overall": overall,
        "per_class": per_class,
    }


def colorize_overlay(base_image: np.ndarray, masks: list[np.ndarray], alpha: float = 0.45, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    overlay = base_image.astype(np.float32).copy()
    for mask in masks:
        color = rng.integers(0, 255, size=3, endpoint=True).astype(np.float32)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_auto_visualization(
    output_path: Path,
    photo_image: np.ndarray,
    gt_masks: list[np.ndarray],
    sam_masks: list[np.ndarray],
    match_masks: list[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(photo_image)
    axes[0, 0].set_title("image")
    axes[0, 1].imshow(colorize_overlay(photo_image, gt_masks, seed=1))
    axes[0, 1].set_title("gt masks")
    axes[1, 0].imshow(colorize_overlay(photo_image, sam_masks, seed=2))
    axes[1, 0].set_title("sam masks")
    axes[1, 1].imshow(colorize_overlay(photo_image, match_masks, seed=3))
    axes[1, 1].set_title("best matches")
    for axis in axes.flat:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_prompt_visualization(
    output_path: Path,
    photo_image: np.ndarray,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(photo_image)
    axes[0].set_title("image")
    axes[1].imshow(colorize_overlay(photo_image, gt_masks, seed=4))
    axes[1].set_title("gt masks")
    axes[2].imshow(colorize_overlay(photo_image, pred_masks, seed=5))
    axes[2].set_title("best prompted masks")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def empty_eval_meta() -> dict[str, int]:
    return {
        "skipped_unreadable_photos": 0,
        "skipped_unreadable_masks": 0,
    }


def evaluate_auto_mode(
    sam,
    grouped_samples: dict[str, list[SegmentSample]],
    output_dir: Path,
    model_type: str,
    alignment_strategy: str,
    cache_auto_masks: bool,
    auto_points_per_side: int | None,
    visualize_first_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    generator = build_auto_mask_generator(sam, points_per_side=auto_points_per_side)
    cache_dir = output_dir / "cache" / "auto_masks"
    vis_dir = ensure_dir(output_dir / "visualizations") if visualize_first_n > 0 else None
    image_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    visualized = 0
    eval_meta = empty_eval_meta()

    for photo_id in tqdm(sorted(grouped_samples.keys()), desc="SAM auto"):
        samples = grouped_samples[photo_id]
        photo_image, gt_masks, valid_samples = load_valid_photo_samples(samples, eval_meta)
        if photo_image is None or not valid_samples:
            continue

        shape_to_masks: dict[tuple[int, int], list[dict[str, Any]]] = {}
        total_inference_time = 0.0
        if alignment_strategy == "resize_sam_to_gt":
            working_shape = photo_image.shape[:2]
            cache_path = make_auto_cache_path(cache_dir, photo_id, working_shape, model_type)
            masks = load_cached_auto_masks(cache_path) if cache_auto_masks else None
            if masks is None:
                start = time.perf_counter()
                with torch.no_grad():
                    raw_masks = generator.generate(photo_image)
                total_inference_time = time.perf_counter() - start
                masks = compact_auto_masks(raw_masks)
                if cache_auto_masks:
                    save_cached_auto_masks(cache_path, masks)
            shape_to_masks[working_shape] = masks
        else:
            unique_shapes = sorted({gt_masks[sample.shape_id].shape for sample in valid_samples})
            for shape in unique_shapes:
                resized_photo = resize_image(photo_image, shape)
                cache_path = make_auto_cache_path(cache_dir, photo_id, shape, model_type)
                masks = load_cached_auto_masks(cache_path) if cache_auto_masks else None
                if masks is None:
                    start = time.perf_counter()
                    with torch.no_grad():
                        raw_masks = generator.generate(resized_photo)
                    total_inference_time += time.perf_counter() - start
                    masks = compact_auto_masks(raw_masks)
                    if cache_auto_masks:
                        save_cached_auto_masks(cache_path, masks)
                shape_to_masks[shape] = masks

        per_image_best_ious: list[float] = []
        matched_vis_masks: list[np.ndarray] = []
        gt_vis_masks: list[np.ndarray] = []
        sam_vis_masks: list[np.ndarray] = []
        first_shape = next(iter(shape_to_masks.keys()))
        sam_vis_masks = [mask["segmentation"] for mask in shape_to_masks[first_shape][: min(20, len(shape_to_masks[first_shape]))]]

        for sample in valid_samples:
            gt_mask = gt_masks[sample.shape_id]
            if gt_mask.sum() == 0:
                continue
            if alignment_strategy == "resize_sam_to_gt":
                candidates = shape_to_masks[photo_image.shape[:2]]
                best = best_match_against_candidates(gt_mask, candidates, target_shape=gt_mask.shape)
            else:
                candidates = shape_to_masks[gt_mask.shape]
                best = best_match_against_candidates(gt_mask, candidates, target_shape=None)
            per_image_best_ious.append(best["iou"])
            gt_vis_masks.append(gt_mask)
            matched_vis_masks.append(best["mask_eval"])
            segment_rows.append(
                {
                    "mode": "auto",
                    "photo_id": sample.photo_id,
                    "shape_id": sample.shape_id,
                    "label_index": sample.label_index,
                    "label_name": sample.label_name,
                    "mask_path": sample.mask_path,
                    "photo_path": sample.photo_path,
                    "gt_area": int(gt_mask.sum()),
                    "best_mask_index": best["mask_index"],
                    "best_iou": best["iou"],
                    "best_dice": best["dice"],
                    "matched_pred_area": best["pred_area"],
                    "matched_predicted_iou": best["predicted_iou"],
                    "matched_stability_score": best["stability_score"],
                    "matched_bbox": best["bbox"],
                    "alignment_strategy": alignment_strategy,
                    "oracle_multimask_selection": False,
                }
            )

        image_rows.append(
            {
                "mode": "auto",
                "photo_id": photo_id,
                "photo_path": valid_samples[0].photo_path,
                "num_gt_segments": len(per_image_best_ious),
                "num_sam_masks": (
                    len(shape_to_masks[photo_image.shape[:2]])
                    if alignment_strategy == "resize_sam_to_gt"
                    else int(sum(len(v) for v in shape_to_masks.values()))
                ) if shape_to_masks else 0,
                "num_sam_mask_sets": len(shape_to_masks),
                "inference_time_sec": total_inference_time,
                "mean_best_iou": float(np.mean(per_image_best_ious)) if per_image_best_ious else 0.0,
                "recall@0.25": float(np.mean([v >= 0.25 for v in per_image_best_ious])) if per_image_best_ious else 0.0,
                "recall@0.50": float(np.mean([v >= 0.50 for v in per_image_best_ious])) if per_image_best_ious else 0.0,
                "recall@0.75": float(np.mean([v >= 0.75 for v in per_image_best_ious])) if per_image_best_ious else 0.0,
                "alignment_strategy": alignment_strategy,
            }
        )

        if vis_dir is not None and visualized < visualize_first_n:
            vis_image = photo_image
            vis_gt = gt_vis_masks[: min(6, len(gt_vis_masks))]
            vis_match = matched_vis_masks[: min(6, len(matched_vis_masks))]
            vis_sam = sam_vis_masks[: min(20, len(sam_vis_masks))]
            if gt_vis_masks and gt_vis_masks[0].shape != vis_image.shape[:2]:
                vis_shape = gt_vis_masks[0].shape
                vis_image = resize_image(photo_image, vis_shape)
                vis_gt = [resize_bool_mask(mask, vis_shape) if mask.shape != vis_shape else mask for mask in vis_gt]
                vis_match = [resize_bool_mask(mask, vis_shape) if mask.shape != vis_shape else mask for mask in vis_match]
                vis_sam = [resize_bool_mask(mask, vis_shape) if mask.shape != vis_shape else mask for mask in vis_sam]
            save_auto_visualization(vis_dir / f"{photo_id}_auto.png", vis_image, vis_gt, vis_sam, vis_match)
            visualized += 1

    aggregate = aggregate_auto(segment_rows, image_rows)
    return image_rows, segment_rows, aggregate, eval_meta


def evaluate_prompt_mode(
    sam,
    grouped_samples: dict[str, list[SegmentSample]],
    output_dir: Path,
    mode: str,
    alignment_strategy: str,
    visualize_first_n: int,
    include_boundary_f: bool,
    boundary_tolerance: int,
    num_negative_points: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    predictor = build_predictor(sam)
    vis_dir = ensure_dir(output_dir / "visualizations") if visualize_first_n > 0 else None
    rng = np.random.default_rng(seed)
    image_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    visualized = 0
    eval_meta = empty_eval_meta()

    for photo_id in tqdm(sorted(grouped_samples.keys()), desc=f"SAM {mode}"):
        samples = grouped_samples[photo_id]
        photo_image, gt_masks, valid_samples = load_valid_photo_samples(samples, eval_meta)
        if photo_image is None or not valid_samples:
            continue

        set_image_time_total = 0.0
        segment_count = 0
        per_image_ious: list[float] = []
        per_image_dices: list[float] = []
        per_image_boundary: list[float] = []
        pred_vis_masks: list[np.ndarray] = []
        gt_vis_masks: list[np.ndarray] = []

        if alignment_strategy == "resize_sam_to_gt":
            start = time.perf_counter()
            predictor.set_image(photo_image)
            set_image_time_total = time.perf_counter() - start
            shape_contexts = {photo_image.shape[:2]: photo_image}
        else:
            shape_contexts = {}
            for shape in sorted({gt_masks[sample.shape_id].shape for sample in valid_samples}):
                resized_photo = resize_image(photo_image, shape)
                shape_contexts[shape] = resized_photo

        grouped_by_shape: dict[tuple[int, int], list[SegmentSample]] = defaultdict(list)
        for sample in valid_samples:
            grouped_by_shape[gt_masks[sample.shape_id].shape].append(sample)

        if alignment_strategy == "resize_sam_to_gt":
            shape_iters = [(photo_image.shape[:2], valid_samples)]
        else:
            shape_iters = list(grouped_by_shape.items())

        for active_shape, shape_samples in shape_iters:
            if alignment_strategy == "resize_photo_to_mask":
                start = time.perf_counter()
                predictor.set_image(shape_contexts[active_shape])
                set_image_time_total += time.perf_counter() - start
            for sample in shape_samples:
                gt_mask = gt_masks[sample.shape_id]
                if gt_mask.sum() == 0:
                    continue
                point_gt = farthest_point_xy(gt_mask) if mode in {"point", "box_point"} else None
                negative_points_gt = sample_negative_points_xy(gt_mask, num_negative_points, rng) if mode == "box_point" else []
                box_gt = mask_bbox_xyxy(gt_mask) if mode in {"box", "box_point"} else None
                if mode in {"point", "box_point"} and point_gt is None:
                    continue
                if mode in {"box", "box_point"} and box_gt is None:
                    continue

                if alignment_strategy == "resize_sam_to_gt":
                    prompt_shape = photo_image.shape[:2]
                    point_prompt = scale_point_xy(point_gt, gt_mask.shape, prompt_shape) if point_gt is not None else None
                    neg_prompt = [scale_point_xy(p, gt_mask.shape, prompt_shape) for p in negative_points_gt]
                    box_prompt = scale_box_xyxy(box_gt, gt_mask.shape, prompt_shape) if box_gt is not None else None
                else:
                    point_prompt = point_gt
                    neg_prompt = negative_points_gt
                    box_prompt = box_gt

                point_coords = None
                point_labels = None
                if point_prompt is not None:
                    coords = [point_prompt] + neg_prompt
                    labels = [1] + [0] * len(neg_prompt)
                    point_coords = np.array(coords, dtype=np.float32)
                    point_labels = np.array(labels, dtype=np.int32)
                box_array = np.array(box_prompt, dtype=np.float32) if box_prompt is not None else None

                start = time.perf_counter()
                with torch.no_grad():
                    masks, scores, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box_array,
                        multimask_output=True,
                    )
                inference_time = time.perf_counter() - start
                candidates = [
                    {
                        "mask_index": index,
                        "segmentation": masks[index].astype(bool),
                        "predicted_iou": float(scores[index]),
                        "stability_score": 0.0,
                        "bbox": box_prompt or [],
                    }
                    for index in range(masks.shape[0])
                ]
                target_shape = gt_mask.shape if alignment_strategy == "resize_sam_to_gt" else None
                best = best_match_against_candidates(
                    gt_mask,
                    candidates,
                    target_shape=target_shape,
                    include_boundary_f=include_boundary_f,
                    boundary_tolerance=boundary_tolerance,
                )
                segment_count += 1
                per_image_ious.append(best["iou"])
                per_image_dices.append(best["dice"])
                if include_boundary_f and best["boundary_fscore"] is not None:
                    per_image_boundary.append(best["boundary_fscore"])
                gt_vis_masks.append(gt_mask)
                pred_vis_masks.append(best["mask_eval"])
                row = {
                    "mode": mode,
                    "photo_id": sample.photo_id,
                    "shape_id": sample.shape_id,
                    "label_index": sample.label_index,
                    "label_name": sample.label_name,
                    "mask_path": sample.mask_path,
                    "photo_path": sample.photo_path,
                    "gt_area": int(gt_mask.sum()),
                    "pred_area": best["pred_area"],
                    "best_mask_index": best["mask_index"],
                    "iou": best["iou"],
                    "dice": best["dice"],
                    "boundary_fscore": best["boundary_fscore"],
                    "sam_scores": [float(v) for v in scores.tolist()],
                    "prompt_point_xy": point_gt or [],
                    "prompt_negative_points_xy": negative_points_gt,
                    "prompt_box_xyxy": box_gt or [],
                    "inference_time_sec": inference_time,
                    "alignment_strategy": alignment_strategy,
                    "oracle_multimask_selection": True,
                }
                segment_rows.append(row)

        image_row = {
            "mode": mode,
            "photo_id": photo_id,
            "photo_path": valid_samples[0].photo_path,
            "num_gt_segments": segment_count,
            "set_image_time_sec": set_image_time_total,
            "mean_iou": float(np.mean(per_image_ious)) if per_image_ious else 0.0,
            "mean_dice": float(np.mean(per_image_dices)) if per_image_dices else 0.0,
            "alignment_strategy": alignment_strategy,
        }
        if include_boundary_f:
            image_row["mean_boundary_fscore"] = float(np.mean(per_image_boundary)) if per_image_boundary else 0.0
        image_rows.append(image_row)

        if vis_dir is not None and visualized < visualize_first_n:
            vis_image = photo_image
            vis_gt = gt_vis_masks[: min(6, len(gt_vis_masks))]
            vis_pred = pred_vis_masks[: min(6, len(pred_vis_masks))]
            if gt_vis_masks and gt_vis_masks[0].shape != vis_image.shape[:2]:
                vis_shape = gt_vis_masks[0].shape
                vis_image = resize_image(photo_image, vis_shape)
                vis_gt = [resize_bool_mask(mask, vis_shape) if mask.shape != vis_shape else mask for mask in vis_gt]
                vis_pred = [resize_bool_mask(mask, vis_shape) if mask.shape != vis_shape else mask for mask in vis_pred]
            save_prompt_visualization(vis_dir / f"{photo_id}_{mode}.png", vis_image, vis_gt, vis_pred)
            visualized += 1

    aggregate = aggregate_prompted(segment_rows, image_rows, include_boundary_f)
    return image_rows, segment_rows, aggregate, eval_meta


def run_evaluation(args) -> dict[str, Any]:
    output_dir = ensure_dir(Path(args.output_dir))
    photos_dir = Path(args.photos_dir) if args.photos_dir else Path(args.dataset_root) / "photos"
    segments_dir = Path(args.segments_dir) if args.segments_dir else Path(args.dataset_root) / "segments"
    categories_path = Path(args.categories) if args.categories else Path(args.dataset_root) / "categories.txt"
    segments_txt = Path(args.segments_txt) if args.segments_txt else Path(args.dataset_root) / "test-segments.txt"

    set_random_seed(args.seed)

    categories = load_categories(categories_path)
    grouped_samples, dataset_summary, warnings = parse_test_segments(
        segments_txt=segments_txt,
        categories=categories,
        photos_dir=photos_dir,
        segments_dir=segments_dir,
        max_images=args.max_images,
    )

    config_payload = vars(args).copy()
    config_payload["photos_dir"] = str(photos_dir)
    config_payload["segments_dir"] = str(segments_dir)
    config_payload["categories"] = str(categories_path)
    config_payload["segments_txt"] = str(segments_txt)
    save_json(config_payload, output_dir / "config.json")

    sam = build_sam_model(args.model_type, Path(args.sam_checkpoint), args.device)

    if args.mode == "auto":
        image_rows, segment_rows, aggregate, eval_meta = evaluate_auto_mode(
            sam=sam,
            grouped_samples=grouped_samples,
            output_dir=output_dir,
            model_type=args.model_type,
            alignment_strategy=args.alignment_strategy,
            cache_auto_masks=args.cache_auto_masks,
            auto_points_per_side=args.auto_points_per_side,
            visualize_first_n=args.visualize_first_n,
        )
    else:
        image_rows, segment_rows, aggregate, eval_meta = evaluate_prompt_mode(
            sam=sam,
            grouped_samples=grouped_samples,
            output_dir=output_dir,
            mode=args.mode,
            alignment_strategy=args.alignment_strategy,
            visualize_first_n=args.visualize_first_n,
            include_boundary_f=args.compute_boundary_f,
            boundary_tolerance=args.boundary_tolerance,
            num_negative_points=args.num_negative_points,
            seed=args.seed,
        )

    payload = {
        "mode": args.mode,
        "model_type": args.model_type,
        "sam_checkpoint": str(args.sam_checkpoint),
        "device": args.device,
        "alignment_strategy": args.alignment_strategy,
        "dataset_summary": {
            **asdict(dataset_summary),
            **eval_meta,
        },
        "warning_count": len(warnings),
        "warnings_preview": warnings[:50],
        "metrics": aggregate,
    }

    write_csv(image_rows, output_dir / "per_image_metrics.csv")
    write_csv(segment_rows, output_dir / "per_segment_metrics.csv")
    save_json(payload, output_dir / "aggregate_metrics.json")
    return payload
