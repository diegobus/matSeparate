"""End-to-end smoke test for MaterialMerger using a stub classifier (no checkpoint)."""

from pathlib import Path

import numpy as np
import pytest

from segmentation.classify import PatchClassifier
from segmentation.config import SegmentationConfig
from segmentation.pipeline import MaterialMerger
from taxonomy.tree import get_taxonomy

REPO = Path(__file__).resolve().parent.parent
C1_TAXONOMY = REPO / "taxonomy" / "assets" / "matador-c1-taxonomy.json"


class StubPredictor:
    """Maps each patch to a near-one-hot leaf distribution by mean brightness."""

    def __init__(self, leaf_names):
        self.leaf_names = list(leaf_names)
        # pick a few well-separated leaves to activate
        self._targets = [
            self.leaf_names[0],
            self.leaf_names[len(self.leaf_names) // 3],
            self.leaf_names[2 * len(self.leaf_names) // 3],
            self.leaf_names[-1],
        ]
        self._target_idx = [self.leaf_names.index(t) for t in self._targets]

    def predict_leaf_probs(self, patches: np.ndarray) -> np.ndarray:
        n = patches.shape[0]
        L = len(self.leaf_names)
        out = np.full((n, L), 1e-3, dtype=np.float32)
        for i in range(n):
            v = float(patches[i].mean()) / 255.0
            bucket = min(int(v * len(self._target_idx)), len(self._target_idx) - 1)
            out[i, self._target_idx[bucket]] = 5.0
        out /= out.sum(axis=1, keepdims=True)
        return out


@pytest.fixture(scope="module")
def graph():
    return get_taxonomy(str(C1_TAXONOMY))


@pytest.fixture(scope="module")
def leaf_names(graph):
    return sorted(n for n in graph.nodes() if graph.out_degree(n) == 0)


@pytest.fixture
def merger(graph, leaf_names):
    config = SegmentationConfig()
    config.crf.backend = "none"
    config.sampling.patch_size = 16
    config.objects.min_object_area = 4
    config.objects.bg_threshold = 0.2
    classifier = PatchClassifier(StubPredictor(leaf_names))
    return MaterialMerger(classifier=classifier, graph=graph, config=config)


def _quadrant_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32, :32] = 20
    img[:32, 32:] = 100
    img[32:, :32] = 170
    img[32:, 32:] = 240
    return img


def test_segment_leaf_level(merger):
    result = merger.segment(_quadrant_image(), level="leaf")
    assert result.label_map.shape == (64, 64)
    assert result.level == "leaf"
    assert len(result.instances) >= 1
    # label ids never exceed number of classes
    assert result.label_map.max() <= len(result.frontier_names)


def test_recut_is_coarser(merger):
    result = merger.segment(_quadrant_image(), level="leaf")
    coarse = result.recut(2)
    assert set(coarse.frontier_names) <= {"abiotic", "biotic"}
    assert coarse.refined_leaf_probs is result.refined_leaf_probs  # cached, no recompute


def test_save_roundtrip(merger, tmp_path):
    result = merger.segment(_quadrant_image(), level="leaf")
    written = result.save(tmp_path)
    assert (tmp_path / "label_map.png").exists()
    assert (tmp_path / "instances.json").exists()
    assert "labels" in written
