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


class ContextLeafProbPredictor(LeafProbPredictor, Protocol):
    """Predictor that can use a local patch plus a larger image context."""

    def predict_leaf_probs_with_context(
        self,
        patches: np.ndarray,
        contexts: np.ndarray,
    ) -> np.ndarray:
        """patches and contexts are both HWC uint8 batches."""
        ...


def _crop_with_padding(
    image: np.ndarray,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    pad_mode: str = "reflect",
) -> np.ndarray:
    h, w = image.shape[:2]
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, y1 - h)
    pad_right = max(0, x1 - w)
    if pad_top or pad_left or pad_bottom or pad_right:
        mode = pad_mode
        if mode == "reflect" and (
            pad_top >= h or pad_bottom >= h or pad_left >= w or pad_right >= w
        ):
            mode = "edge"
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=mode,
        )
        y0 += pad_top
        y1 += pad_top
        x0 += pad_left
        x1 += pad_left
    return image[y0:y1, x0:x1, :]


def build_context_batch(
    sample: SampleResult,
    image_context: np.ndarray,
    mode: str,
    scale: float = 4.0,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Build one context image per local patch without using mask annotations."""
    if mode == "full_image":
        return np.repeat(image_context[None, ...], sample.patches.shape[0], axis=0)
    if mode != "scaled_window":
        raise ValueError(f"Unknown context mode: {mode}")
    if sample.bounds is None:
        raise ValueError("scaled_window context requires sampler bounds")
    contexts = []
    for y0, x0, y1, x1 in sample.bounds:
        cy = 0.5 * (y0 + y1)
        cx = 0.5 * (x0 + x1)
        side = max(y1 - y0, x1 - x0) * scale
        half = 0.5 * side
        contexts.append(
            _crop_with_padding(
                image_context,
                int(round(cy - half)),
                int(round(cx - half)),
                int(round(cy + half)),
                int(round(cx + half)),
                pad_mode=pad_mode,
            )
        )
    return np.stack(contexts, axis=0).astype(np.uint8, copy=False)


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


class GlobalResNetLeafPredictor:
    """Adapter wrapping ``GlobalResNetInference`` for local/global segmentation patches."""

    def __init__(self, api, batch_size: int = 32, show_progress: bool = True):
        self.api = api
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.leaf_names: List[str] = list(api.leaf_names)

    def predict_leaf_probs(self, patches: np.ndarray) -> np.ndarray:
        contexts = np.repeat(patches[:, None, ...], 1, axis=1)[:, 0]
        return self.predict_leaf_probs_with_context(patches, contexts)

    def predict_leaf_probs_with_context(
        self,
        patches: np.ndarray,
        contexts: np.ndarray,
    ) -> np.ndarray:
        n = patches.shape[0]
        out = np.empty((n, len(self.leaf_names)), dtype=np.float32)
        starts = list(range(0, n, self.batch_size))
        for start in _progress(
            starts, total=len(starts), desc="classifying patches", enabled=self.show_progress
        ):
            end = min(start + self.batch_size, n)
            results = self.api.infer_batch(
                [patches[i] for i in range(start, end)],
                [contexts[i] for i in range(start, end)],
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

    def __init__(
        self,
        predictor: LeafProbPredictor,
        context_mode: str = "none",
        context_scale: float = 4.0,
    ):
        self.predictor = predictor
        self.context_mode = context_mode
        self.context_scale = context_scale

    @property
    def leaf_names(self) -> List[str]:
        return list(self.predictor.leaf_names)

    @classmethod
    def from_hgnn(
        cls, api, batch_size: int = 32, show_progress: bool = True
    ) -> "PatchClassifier":
        return cls(HGNNLeafPredictor(api, batch_size=batch_size, show_progress=show_progress))

    @classmethod
    def from_global_resnet(
        cls,
        api,
        batch_size: int = 32,
        show_progress: bool = True,
        context_mode: str = "scaled_window",
        context_scale: float = 4.0,
    ) -> "PatchClassifier":
        return cls(
            GlobalResNetLeafPredictor(api, batch_size=batch_size, show_progress=show_progress),
            context_mode=context_mode,
            context_scale=context_scale,
        )

    @classmethod
    def from_global_hgnn(
        cls,
        api,
        batch_size: int = 32,
        show_progress: bool = True,
        context_mode: str = "scaled_window",
        context_scale: float = 4.0,
    ) -> "PatchClassifier":
        return cls(
            GlobalResNetLeafPredictor(api, batch_size=batch_size, show_progress=show_progress),
            context_mode=context_mode,
            context_scale=context_scale,
        )

    def classify(self, sample: SampleResult, image_context: np.ndarray | None = None) -> np.ndarray:
        """Return P_grid with shape ``(gh, gw, num_leaves)`` (float32 simplex)."""
        gh, gw = sample.grid_shape
        num_leaves = len(self.leaf_names)
        if (
            self.context_mode != "none"
            and image_context is not None
            and hasattr(self.predictor, "predict_leaf_probs_with_context")
        ):
            contexts = build_context_batch(
                sample,
                image_context=image_context,
                mode=self.context_mode,
                scale=self.context_scale,
            )
            probs = self.predictor.predict_leaf_probs_with_context(sample.patches, contexts)
        else:
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
