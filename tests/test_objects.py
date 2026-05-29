"""Tests for segmentation.objects: label map + connected-component instances."""

import numpy as np

from segmentation.objects import (
    BACKGROUND_ID,
    build_label_map,
    extract_instances,
)


def _two_class_probs(h, w):
    """Left half -> class 0, right half -> class 1, both high confidence."""
    probs = np.zeros((h, w, 2), dtype=np.float32)
    probs[:, : w // 2, 0] = 0.9
    probs[:, : w // 2, 1] = 0.1
    probs[:, w // 2 :, 0] = 0.1
    probs[:, w // 2 :, 1] = 0.9
    return probs


def test_label_map_ids_and_background():
    probs = _two_class_probs(4, 8)
    label_map, conf = build_label_map(probs, bg_threshold=0.5)
    assert label_map.shape == (4, 8)
    assert set(np.unique(label_map)) == {1, 2}  # materials shifted to start at 1
    assert conf.max() <= 1.0

    # raise threshold above every confidence -> everything background
    label_map2, _ = build_label_map(probs, bg_threshold=0.95)
    assert np.all(label_map2 == BACKGROUND_ID)


def test_adjacent_same_material_is_one_object():
    probs = _two_class_probs(6, 10)
    label_map, conf = build_label_map(probs, bg_threshold=0.5)
    insts = extract_instances(
        label_map, conf, ["mat_a", "mat_b"], min_object_area=1
    )
    # exactly one object per class (each class is a single connected block)
    materials = sorted(i.material for i in insts)
    assert materials == ["mat_a", "mat_b"]


def test_disjoint_same_material_two_objects():
    # class 1 appears as two separated vertical strips with a background gap
    h, w = 6, 9
    probs = np.zeros((h, w, 1), dtype=np.float32)
    probs[:, 0:3, 0] = 0.9
    probs[:, 6:9, 0] = 0.9
    # middle stays low -> background
    label_map, conf = build_label_map(probs, bg_threshold=0.5)
    insts = extract_instances(label_map, conf, ["wood"], connectivity=8, min_object_area=1)
    assert len(insts) == 2
    assert all(i.material == "wood" for i in insts)


def test_min_area_filter():
    h, w = 6, 9
    probs = np.zeros((h, w, 1), dtype=np.float32)
    probs[0, 0, 0] = 0.9  # single-pixel speck
    probs[:, 5:9, 0] = 0.9  # larger block
    label_map, conf = build_label_map(probs, bg_threshold=0.5)
    insts = extract_instances(label_map, conf, ["wood"], min_object_area=5)
    assert len(insts) == 1
    assert insts[0].area >= 5


def test_instance_fields():
    probs = _two_class_probs(4, 8)
    label_map, conf = build_label_map(probs, bg_threshold=0.5)
    insts = extract_instances(label_map, conf, ["a", "b"], min_object_area=1)
    inst = insts[0]
    assert inst.mask.shape == (4, 8)
    assert len(inst.bbox) == 4
    assert 0.0 <= inst.score <= 1.0
