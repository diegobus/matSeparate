"""
Dense CRF refinement of the per-pixel probability map.

Primary backend uses ``pydensecrf`` (Krähenbühl & Koltun fully-connected CRF) with a
Gaussian smoothness term and an edge-aware bilateral term -- the step that snaps the blocky
grid unaries to real image boundaries. Optionally the label-compatibility matrix is built
from taxonomy distance so sibling-material confusions are penalized less than distant ones.

Because ``pydensecrf`` is an unmaintained native dependency that can be hard to install,
this module degrades gracefully:

    dense  --(missing pydensecrf)-->  superpixel  --(missing skimage)-->  none

All backends take and return a dense probability map ``(H, W, L)``.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import networkx as nx
import numpy as np


def _have_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def build_taxonomy_compat(
    graph: nx.DiGraph,
    leaf_names: List[str],
    scale: float = 1.0,
) -> np.ndarray:
    """Build an ``(L, L)`` label-compatibility matrix from taxonomy distance.

    Diagonal is 0 (no penalty for agreeing). Off-diagonal entries grow with the
    (undirected) shortest-path distance between the two leaf nodes in the taxonomy,
    normalized to ``[0, 1]`` and multiplied by ``scale``. Confusing siblings is therefore
    cheaper than confusing distant materials.
    """
    undirected = graph.to_undirected()
    n = len(leaf_names)
    dist = np.zeros((n, n), dtype=np.float32)
    lengths = dict(nx.all_pairs_shortest_path_length(undirected))
    for i, a in enumerate(leaf_names):
        for j, b in enumerate(leaf_names):
            if i == j:
                continue
            dist[i, j] = lengths.get(a, {}).get(b, 0)
    max_d = dist.max()
    if max_d > 0:
        dist = dist / max_d
    return (dist * float(scale)).astype(np.float32)


def _refine_dense(
    image_uint8: np.ndarray,
    p_dense: np.ndarray,
    config,
    leaf_names: Optional[List[str]],
    graph: Optional[nx.DiGraph],
) -> np.ndarray:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w, num_labels = p_dense.shape
    probs = np.clip(p_dense, 1e-8, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)

    unary = unary_from_softmax(
        np.ascontiguousarray(probs.transpose(2, 0, 1))  # (L, H, W)
    )

    d = dcrf.DenseCRF2D(w, h, num_labels)
    d.setUnaryEnergy(np.ascontiguousarray(unary))

    # Compatibility: scalar Potts by default, taxonomy-aware matrix if requested.
    compat_g = config.gaussian_compat
    compat_b = config.bilateral_compat
    if config.taxonomy_aware_compat and graph is not None and leaf_names is not None:
        try:
            compat_b = build_taxonomy_compat(
                graph, leaf_names, scale=config.bilateral_compat
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"taxonomy-aware compat failed ({exc}); using Potts")

    d.addPairwiseGaussian(sxy=config.gaussian_sxy, compat=float(compat_g))
    d.addPairwiseBilateral(
        sxy=config.bilateral_sxy,
        srgb=config.bilateral_srgb,
        rgbim=np.ascontiguousarray(image_uint8),
        compat=compat_b if isinstance(compat_b, np.ndarray) else float(compat_b),
    )

    q = d.inference(config.n_iterations)
    refined = np.array(q).reshape(num_labels, h, w).transpose(1, 2, 0)
    return refined.astype(np.float32)


def _refine_superpixel(
    image_uint8: np.ndarray,
    p_dense: np.ndarray,
    config,
) -> np.ndarray:
    from skimage.segmentation import felzenszwalb, slic

    if config.superpixel_method == "felzenszwalb":
        segments = felzenszwalb(image_uint8, scale=100, sigma=0.5, min_size=50)
    else:
        segments = slic(
            image_uint8,
            n_segments=config.superpixel_n_segments,
            start_label=0,
            channel_axis=-1,
        )

    refined = np.empty_like(p_dense)
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        mean_prob = p_dense[mask].mean(axis=0)
        refined[mask] = mean_prob
    denom = refined.sum(axis=-1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return (refined / denom).astype(np.float32)


def refine(
    image_uint8: np.ndarray,
    p_dense: np.ndarray,
    config,
    leaf_names: Optional[List[str]] = None,
    graph: Optional[nx.DiGraph] = None,
) -> np.ndarray:
    """Refine a dense probability map. Returns a ``(H, W, L)`` float32 map.

    Args:
        image_uint8: (H, W, 3) uint8 RGB image (the bilateral term reads its colors).
        p_dense: (H, W, L) dense leaf-probability map.
        config: a ``CRFConfig``.
        leaf_names / graph: needed only for taxonomy-aware compatibility.
    """
    backend = config.backend

    if backend == "none":
        return p_dense.astype(np.float32)

    if backend == "dense":
        if _have_module("pydensecrf"):
            return _refine_dense(image_uint8, p_dense, config, leaf_names, graph)
        warnings.warn(
            "pydensecrf not installed; falling back to superpixel CRF backend. "
            "Install with: pip install git+https://github.com/lucasb-eyer/pydensecrf.git"
        )
        backend = "superpixel"

    if backend == "superpixel":
        if _have_module("skimage"):
            return _refine_superpixel(image_uint8, p_dense, config)
        warnings.warn("scikit-image not installed; skipping CRF refinement (backend=none).")
        return p_dense.astype(np.float32)

    raise ValueError(f"unknown CRF backend: {config.backend}")
