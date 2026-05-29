"""
Patch classification: run the leaf-material classifier over sampled patches and
assemble a coarse ``(gh, gw, num_leaves)`` probability grid.

The classifier is abstracted behind a tiny ``LeafProbPredictor`` protocol so the pipeline
can run with the real HGNN (``HGNNLeafPredictor``) or with a stub in tests. We always keep
the **leaf softmax** distribution (a proper simplex), not the sigmoid node probs; coarser
taxonomy levels are derived later by summing descendant leaves (see ``taxonomy_cut``).
"""

from __future__ import annotations

from typing import Iterable, List, Protocol

import numpy as np

from segmentation.patches import SampleResult


def _progress(iterable: Iterable, total: int, desc: str, enabled: bool):
    """Wrap an iterable in a tqdm bar if available/enabled, else return it unchanged."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="batch")


class LeafProbPredictor(Protocol):
    """Anything that can turn a batch of HWC patches into leaf probabilities."""

    leaf_names: List[str]

    def predict_leaf_probs(self, patches: np.ndarray) -> np.ndarray:
        """patches: (N, P, P, 3) -> (N, num_leaves) softmax over leaves."""
        ...


class HGNNLeafPredictor:
    """Adapter wrapping ``scripts.infer_api.HGNNInference`` as a ``LeafProbPredictor``."""

    def __init__(self, api, batch_size: int = 32, show_progress: bool = True):
        self.api = api
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.leaf_names: List[str] = list(api.leaf_names)

    def predict_leaf_probs(self, patches: np.ndarray) -> np.ndarray:
        n = patches.shape[0]
        out = np.empty((n, len(self.leaf_names)), dtype=np.float32)
        starts = list(range(0, n, self.batch_size))
        for start in _progress(
            starts, total=len(starts), desc="classifying patches", enabled=self.show_progress
        ):
            chunk = [patches[i] for i in range(start, min(start + self.batch_size, n))]
            results = self.api.infer_batch(
                chunk, return_probs=True, decode_path=False
            )
            for j, res in enumerate(results):
                out[start + j] = np.asarray(res["leaf_probs"], dtype=np.float32)
        return out


class StubLeafPredictor:
    """A checkpoint-free predictor for testing plumbing / visualization.

    Maps each patch to a near-one-hot leaf distribution based on its mean brightness, so
    different image regions get different (arbitrary) materials. Predictions are
    meaningless materials -- this only validates the pipeline and visuals, not quality.
    """

    def __init__(self, leaf_names: List[str], num_buckets: int = 6):
        self.leaf_names = list(leaf_names)
        n = len(self.leaf_names)
        step = max(1, n // max(1, num_buckets))
        self._target_idx = list(range(0, n, step))[:num_buckets] or [0]

    def predict_leaf_probs(self, patches: np.ndarray) -> np.ndarray:
        n = patches.shape[0]
        out = np.full((n, len(self.leaf_names)), 1e-3, dtype=np.float32)
        for i in range(n):
            v = float(patches[i].mean()) / 255.0
            b = min(int(v * len(self._target_idx)), len(self._target_idx) - 1)
            out[i, self._target_idx[b]] = 5.0
        out /= out.sum(axis=1, keepdims=True)
        return out


class PatchClassifier:
    """Assembles a coarse leaf-probability grid from sampled patches."""

    def __init__(self, predictor: LeafProbPredictor):
        self.predictor = predictor

    @property
    def leaf_names(self) -> List[str]:
        return list(self.predictor.leaf_names)

    @classmethod
    def from_hgnn(
        cls, api, batch_size: int = 32, show_progress: bool = True
    ) -> "PatchClassifier":
        return cls(HGNNLeafPredictor(api, batch_size=batch_size, show_progress=show_progress))

    def classify(self, sample: SampleResult) -> np.ndarray:
        """Return P_grid with shape ``(gh, gw, num_leaves)`` (float32 simplex)."""
        gh, gw = sample.grid_shape
        num_leaves = len(self.leaf_names)
        probs = self.predictor.predict_leaf_probs(sample.patches)  # (N, L)
        if probs.shape[0] != len(sample.grid_coords):
            raise ValueError(
                f"predictor returned {probs.shape[0]} rows for "
                f"{len(sample.grid_coords)} patches"
            )
        p_grid = np.zeros((gh, gw, num_leaves), dtype=np.float32)
        for (gi, gj), row in zip(sample.grid_coords, probs):
            p_grid[gi, gj] = row
        return p_grid
