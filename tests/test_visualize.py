"""Smoke tests for segmentation.visualize."""

from pathlib import Path

import numpy as np

from segmentation.classify import PatchClassifier, StubLeafPredictor
from segmentation.config import SegmentationConfig
from segmentation.pipeline import MaterialMerger
from segmentation.visualize import (
    instance_overlay,
    save_level_comparison,
    save_panel,
    semantic_overlay,
)
from taxonomy.tree import get_taxonomy

REPO = Path(__file__).resolve().parent.parent
C1_TAXONOMY = REPO / "taxonomy" / "assets" / "matador-c1-taxonomy.json"


def _make_result():
    graph = get_taxonomy(str(C1_TAXONOMY))
    leaf_names = sorted(n for n in graph.nodes() if graph.out_degree(n) == 0)
    config = SegmentationConfig()
    config.crf.backend = "none"
    config.sampling.patch_size = 16
    config.objects.min_object_area = 4
    config.objects.bg_threshold = 0.2
    merger = MaterialMerger(PatchClassifier(StubLeafPredictor(leaf_names)), graph, config)
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[:24, :24] = 30
    img[24:, 24:] = 220
    return img, merger.segment(img, level="leaf")


def test_overlays_shapes():
    img, result = _make_result()
    sem = semantic_overlay(img, result.label_map, len(result.frontier_names))
    inst = instance_overlay(img, result.instances)
    assert sem.shape == img.shape
    assert inst.shape == img.shape
    assert sem.dtype == np.uint8


def test_save_panel(tmp_path):
    img, result = _make_result()
    out = tmp_path / "panel.png"
    path = save_panel(out, img, result)
    assert Path(path).exists()


def test_save_panel_with_gt(tmp_path):
    img, result = _make_result()
    gt = np.zeros(result.label_map.shape, dtype=np.int64)
    gt[:24] = 1
    out = tmp_path / "panel_gt.png"
    path = save_panel(out, img, result, gt_label_map=gt, gt_frontier_names=["a"])
    assert Path(path).exists()


def test_save_level_comparison(tmp_path):
    img, result = _make_result()  # result.level == "leaf"
    out = tmp_path / "compare.png"
    path = save_level_comparison(out, img, result, levels=["leaf", 3, 2])
    assert Path(path).exists()
    # original result is unchanged / still leaf level after re-cuts
    assert result.level == "leaf"
