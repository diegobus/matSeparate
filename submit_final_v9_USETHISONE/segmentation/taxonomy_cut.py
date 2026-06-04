"""
Taxonomy level cut: aggregate leaf-material probabilities to a chosen level.

Default level ``"leaf"`` keeps the full set of leaf materials (most detailed). Passing an
integer depth ``d`` coarsens the segmentation: each leaf maps to its ancestor at depth
``d``. A leaf shallower than ``d`` keeps its own label (``shallow_leaf="keep"``), which
preserves detail; for the Matador-C1 tree this only matters for ``generic_metal`` (depth 4)
at the never-coarsening case ``d = 5``.

The aggregation is a single ``(L -> K)`` 0/1 matrix ``A`` so that
``F = P_leaf @ A`` and any level is one cheap matmul away from the cached leaf map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import networkx as nx
import numpy as np


@dataclass
class FrontierCut:
    """A resolved taxonomy cut: the frontier class names and the leaf->frontier matrix."""

    frontier_names: List[str]  # length K, the class names at this level
    aggregation: np.ndarray  # (L, K) 0/1, row-stochastic (each leaf -> exactly one class)
    leaf_names: List[str]  # length L, the leaf ordering this matrix expects
    level: Union[str, int]


def _node_depths(graph: nx.DiGraph, root: str = "root") -> dict:
    return nx.single_source_shortest_path_length(graph, root)


def build_frontier(
    graph: nx.DiGraph,
    leaf_names: List[str],
    level: Union[str, int] = "leaf",
    shallow_leaf: str = "keep",
    root: str = "root",
) -> FrontierCut:
    """Resolve a level into frontier class names + a leaf->frontier aggregation matrix."""
    num_leaves = len(leaf_names)

    if level == "leaf":
        agg = np.eye(num_leaves, dtype=np.float32)
        return FrontierCut(list(leaf_names), agg, list(leaf_names), "leaf")

    if not isinstance(level, int):
        raise ValueError(f"level must be 'leaf' or an int depth, got {level!r}")
    if level < 0:
        raise ValueError("integer level (depth) must be >= 0")

    depths = _node_depths(graph, root)

    # Map each leaf to its representative node at the requested depth.
    leaf_to_rep: List[str] = []
    for leaf in leaf_names:
        if leaf not in graph:
            raise ValueError(f"leaf '{leaf}' not found in taxonomy graph")
        path = nx.shortest_path(graph, root, leaf)  # [root, ..., leaf]
        leaf_depth = len(path) - 1
        if leaf_depth >= level:
            rep = path[level]
        else:
            # Leaf is shallower than the requested depth.
            if shallow_leaf == "keep":
                rep = leaf
            elif shallow_leaf == "parent":
                rep = path[-2] if len(path) >= 2 else leaf
            else:
                raise ValueError(f"unknown shallow_leaf mode: {shallow_leaf}")
        leaf_to_rep.append(rep)

    frontier_names = sorted(set(leaf_to_rep))
    rep_to_col = {name: k for k, name in enumerate(frontier_names)}

    agg = np.zeros((num_leaves, len(frontier_names)), dtype=np.float32)
    for i, rep in enumerate(leaf_to_rep):
        agg[i, rep_to_col[rep]] = 1.0

    return FrontierCut(frontier_names, agg, list(leaf_names), level)


def apply_frontier(p_leaf_dense: np.ndarray, cut: FrontierCut) -> np.ndarray:
    """Aggregate a dense leaf-prob map ``(H, W, L)`` to ``(H, W, K)`` at the cut level."""
    if p_leaf_dense.shape[-1] != cut.aggregation.shape[0]:
        raise ValueError(
            f"leaf dim {p_leaf_dense.shape[-1]} != aggregation rows "
            f"{cut.aggregation.shape[0]}"
        )
    h, w, _ = p_leaf_dense.shape
    flat = p_leaf_dense.reshape(-1, p_leaf_dense.shape[-1])
    frontier = flat @ cut.aggregation  # (H*W, K)
    return frontier.reshape(h, w, cut.aggregation.shape[1]).astype(np.float32)
