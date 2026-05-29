"""Tests for segmentation.taxonomy_cut: leaf default + depth coarsening."""

from pathlib import Path

import numpy as np
import pytest

from segmentation.taxonomy_cut import apply_frontier, build_frontier
from taxonomy.tree import get_taxonomy

REPO = Path(__file__).resolve().parent.parent
C1_TAXONOMY = REPO / "taxonomy" / "assets" / "matador-c1-taxonomy.json"


@pytest.fixture(scope="module")
def graph():
    return get_taxonomy(str(C1_TAXONOMY))


@pytest.fixture(scope="module")
def leaf_names(graph):
    return sorted(n for n in graph.nodes() if graph.out_degree(n) == 0)


def test_leaf_level_is_identity(graph, leaf_names):
    cut = build_frontier(graph, leaf_names, level="leaf")
    assert cut.frontier_names == list(leaf_names)
    assert np.allclose(cut.aggregation, np.eye(len(leaf_names)))


def test_aggregation_is_row_stochastic_onehot(graph, leaf_names):
    for depth in (1, 2, 3, 4):
        cut = build_frontier(graph, leaf_names, level=depth)
        # each leaf maps to exactly one frontier class
        assert np.all(cut.aggregation.sum(axis=1) == 1)
        assert set(np.unique(cut.aggregation)) <= {0.0, 1.0}
        assert cut.aggregation.shape == (len(leaf_names), len(cut.frontier_names))


def test_depth2_is_biotic_abiotic(graph, leaf_names):
    cut = build_frontier(graph, leaf_names, level=2)
    assert set(cut.frontier_names) == {"abiotic", "biotic"}


def test_depth3_classes(graph, leaf_names):
    cut = build_frontier(graph, leaf_names, level=3)
    expected = {"metal", "rock", "ceramic", "polymer", "natural", "derivative"}
    assert set(cut.frontier_names) == expected


def test_generic_metal_kept_at_depth5(graph, leaf_names):
    # generic_metal is the only leaf at depth 4; at depth 5 it keeps its own label.
    cut = build_frontier(graph, leaf_names, level=5, shallow_leaf="keep")
    assert "generic_metal" in cut.frontier_names


def test_apply_frontier_preserves_pixel_mass(graph, leaf_names):
    rng = np.random.default_rng(0)
    L = len(leaf_names)
    p = rng.random((5, 7, L)).astype(np.float32)
    p /= p.sum(axis=-1, keepdims=True)
    cut = build_frontier(graph, leaf_names, level=3)
    f = apply_frontier(p, cut)
    assert f.shape == (5, 7, len(cut.frontier_names))
    # mass is preserved per pixel (each leaf routed to exactly one class)
    assert np.allclose(f.sum(axis=-1), p.sum(axis=-1), atol=1e-5)


def test_invalid_level_raises(graph, leaf_names):
    with pytest.raises(ValueError):
        build_frontier(graph, leaf_names, level="deepest")
