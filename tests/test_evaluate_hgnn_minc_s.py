"""Tests for the HGNN MINC-S evaluator helpers."""

import numpy as np

from scripts.evaluate_hgnn_minc_s import aggregate, compute_dice, compute_iou


def test_overlap_metrics():
    pred = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    gt = np.array([[1, 0, 0], [0, 1, 1]], dtype=bool)
    assert abs(compute_iou(pred, gt) - 0.5) < 1e-9
    assert abs(compute_dice(pred, gt) - (4 / 6)) < 1e-9


def test_aggregate_sam_style_fields():
    rows = [
        {
            "photo_id": "a",
            "label_name": "wood",
            "best_iou": 0.8,
            "best_dice": 0.9,
            "gt_is_mapped": True,
            "best_component_index": 0,
            "matched_maps_to_gt": True,
        },
        {
            "photo_id": "b",
            "label_name": "metal",
            "best_iou": 0.2,
            "best_dice": 0.3,
            "gt_is_mapped": False,
            "best_component_index": 1,
            "matched_maps_to_gt": False,
        },
    ]
    image_rows = [
        {"photo_id": "a", "num_hgnn_components": 3, "inference_time_sec": 1.0},
        {"photo_id": "b", "num_hgnn_components": 5, "inference_time_sec": 3.0},
    ]
    metrics = aggregate(rows, image_rows)
    overall = metrics["overall"]
    assert overall["num_images"] == 2
    assert overall["num_segments"] == 2
    assert overall["mean_best_iou"] == 0.5
    assert overall["recall@0.50"] == 0.5
    assert overall["mean_num_components_per_image"] == 4.0
    assert overall["mean_inference_time_sec_per_image"] == 2.0
    assert metrics["semantic_mapped"]["mapped_segment_accuracy"] == 1.0
