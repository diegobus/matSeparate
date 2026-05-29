"""Tests for segmentation.metrics."""

import numpy as np

from segmentation.metrics import (
    adjusted_rand_index,
    build_confusion,
    build_shared_space,
    mask_agreement,
    metrics_from_confusion,
    remap_label_map,
    to_connected_components,
    variation_of_information,
)


def test_perfect_prediction_iou_one():
    gt = np.array([[1, 1, 2], [2, 2, 1]], dtype=np.int64)
    pred = gt.copy()
    class_names = ["background", "wood", "metal"]
    cm = build_confusion(pred, gt, num_classes=3)
    m = metrics_from_confusion(cm, class_names)
    assert m.mean_iou == 1.0
    assert m.mean_acc == 1.0
    assert m.pixel_acc == 1.0
    assert m.num_classes_present == 2


def test_half_wrong():
    gt = np.array([[1, 1], [2, 2]], dtype=np.int64)
    pred = np.array([[1, 2], [2, 2]], dtype=np.int64)
    class_names = ["background", "wood", "metal"]
    cm = build_confusion(pred, gt, num_classes=3)
    m = metrics_from_confusion(cm, class_names)
    # wood: tp=1, fn=1 -> iou 0.5 ; metal: tp=2, fp=1 -> iou 2/3
    assert abs(m.per_class_iou["wood"] - 0.5) < 1e-9
    assert abs(m.per_class_iou["metal"] - (2 / 3)) < 1e-9


def test_background_ignored_by_default():
    gt = np.array([[0, 0], [1, 1]], dtype=np.int64)
    pred = np.array([[5, 5], [1, 1]], dtype=np.int64)  # bg pixels predicted wrong
    class_names = ["background", "wood"]
    cm = build_confusion(pred, gt, num_classes=2)  # pred id 5 clipped into range
    m = metrics_from_confusion(cm, class_names)
    # only 'wood' scored, and it is perfect where GT is non-background
    assert m.per_class_iou["wood"] == 1.0
    assert m.num_classes_present == 1


def test_shared_space_and_crosswalk():
    pred_legend = {"0": "background", "1": "timber", "2": "marble"}
    gt_legend = {"0": "background", "1": "wood", "2": "stone"}
    crosswalk = {"timber": "wood", "marble": "stone"}
    names, pred_remap, gt_remap = build_shared_space(pred_legend, gt_legend, crosswalk)
    assert names[0] == "background"
    assert set(names[1:]) == {"wood", "stone"}
    # pred id 1 (timber->wood) and gt id 1 (wood) map to the same shared id
    assert pred_remap[1] == gt_remap[1]
    assert pred_remap[2] == gt_remap[2]


def test_remap_label_map():
    lm = np.array([[1, 2], [2, 0]], dtype=np.int64)
    out = remap_label_map(lm, {0: 0, 1: 3, 2: 7})
    assert out.tolist() == [[3, 7], [7, 0]]


# --- class-agnostic mask similarity --- #

def test_identical_partition_is_perfect():
    a = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 1, 1]], dtype=np.int64)
    assert adjusted_rand_index(a, a) == 1.0
    assert variation_of_information(a, a) < 1e-9
    ag = mask_agreement(a, a)
    assert ag.mean_best_iou == 1.0
    assert ag.covering_gt_by_pred == 1.0
    assert ag.boundary_f1 == 1.0


def test_label_values_irrelevant():
    # same grouping, different label values -> still identical partition
    a = np.array([[0, 0, 5, 5], [0, 0, 5, 5]], dtype=np.int64)
    b = np.array([[9, 9, 3, 3], [9, 9, 3, 3]], dtype=np.int64)
    assert adjusted_rand_index(a, b) == 1.0
    ag = mask_agreement(a, b)
    assert ag.mean_best_iou == 1.0


def test_split_region_lowers_agreement():
    whole = np.zeros((4, 4), dtype=np.int64)
    whole[:, 2:] = 1
    scrambled = np.arange(16).reshape(4, 4)  # every pixel its own region
    ari_same = adjusted_rand_index(whole, whole)
    ari_diff = adjusted_rand_index(whole, scrambled)
    assert ari_diff < ari_same


def test_connected_components_split():
    # one label, two spatially separate blobs -> 2 regions when connected=True
    lm = np.zeros((3, 7), dtype=np.int64)
    lm[:, 0:2] = 5
    lm[:, 5:7] = 5
    cc = to_connected_components(lm, connectivity=8)
    # background (0) is one region, plus the two 5-blobs => 3 region ids
    assert len(np.unique(cc)) == 3
