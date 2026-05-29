"""
Patch sampling: break an image into a grid of patches for the patch classifier.

The default ``GridSampler`` tiles the image into a **non-overlapping** grid of
``patch_size`` squares, reflect-padding the bottom/right remainder so every tile is
full-size and the grid is rectangular. Each patch carries its grid index ``(gi, gj)`` and
pixel bounds so the classifier output can be assembled into a coarse ``(gh, gw, L)`` grid.

The ``PatchSampler`` interface is intentionally small so an overlapping / multi-scale
``SlidingWindowSampler`` can be dropped in later (see MERGE.md) without touching the rest
of the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SampleResult:
    """Output of a ``PatchSampler``."""

    patches: np.ndarray  # (N, patch_size, patch_size, 3)
    grid_coords: List[Tuple[int, int]]  # (gi, gj) per patch, len == N
    grid_shape: Tuple[int, int]  # (gh, gw)
    orig_shape: Tuple[int, int]  # (H, W) of the input image
    padded_shape: Tuple[int, int]  # (Hp, Wp) after padding


class PatchSampler:
    """Base interface for patch samplers."""

    def sample(self, image: np.ndarray) -> SampleResult:  # pragma: no cover - interface
        raise NotImplementedError


class GridSampler(PatchSampler):
    """Non-overlapping grid of ``patch_size`` tiles with reflect padding."""

    def __init__(self, patch_size: int = 224, pad_mode: str = "reflect"):
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.pad_mode = pad_mode

    def sample(self, image: np.ndarray) -> SampleResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

        h, w = image.shape[:2]
        p = self.patch_size
        gh = math.ceil(h / p)
        gw = math.ceil(w / p)
        hp, wp = gh * p, gw * p

        pad_h = hp - h
        pad_w = wp - w
        if pad_h or pad_w:
            # reflect needs the pad to be < dimension; fall back to edge for tiny images
            mode = self.pad_mode
            if mode == "reflect" and (pad_h >= h or pad_w >= w):
                mode = "edge"
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        else:
            padded = image

        patches = []
        grid_coords = []
        for gi in range(gh):
            for gj in range(gw):
                y0, x0 = gi * p, gj * p
                tile = padded[y0 : y0 + p, x0 : x0 + p, :]
                patches.append(tile)
                grid_coords.append((gi, gj))

        patches_arr = np.stack(patches, axis=0) if patches else np.empty((0, p, p, 3))
        return SampleResult(
            patches=patches_arr,
            grid_coords=grid_coords,
            grid_shape=(gh, gw),
            orig_shape=(h, w),
            padded_shape=(hp, wp),
        )


def _positions(length: int, window: int, stride: int) -> List[int]:
    """Start coordinates so windows of ``window`` (stride ``stride``) tile ``length``.

    The last window is snapped to the edge so the full extent is always covered.
    """
    if length <= window:
        return [0]
    pos = list(range(0, length - window + 1, stride))
    if pos[-1] != length - window:
        pos.append(length - window)
    return pos


class SlidingWindowSampler(PatchSampler):
    """Overlapping windows of ``window_size`` at spacing ``stride``.

    Each window is later resized to the classifier's input size, so a *small* window
    yields both a denser prediction grid (finer segmentation) and a closer-up material
    view (closer to the patch-classifier training distribution). The window predictions
    are laid out on a regular ``(n_rows, n_cols)`` grid (one cell per window position) and
    bilinearly upsampled downstream -- with overlap this grid is much denser than the
    non-overlapping ``GridSampler``.
    """

    def __init__(self, window_size: int = 96, stride: int = 48, pad_mode: str = "reflect"):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.stride = stride if stride and stride > 0 else max(1, window_size // 2)
        self.pad_mode = pad_mode

    def sample(self, image: np.ndarray) -> SampleResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

        h, w = image.shape[:2]
        ws = self.window_size

        # pad so a full window always fits (at least window_size in each dim)
        hp, wp = max(h, ws), max(w, ws)
        pad_h, pad_w = hp - h, wp - w
        if pad_h or pad_w:
            mode = self.pad_mode
            if mode == "reflect" and (pad_h >= h or pad_w >= w):
                mode = "edge"
            padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        else:
            padded = image

        ys = _positions(hp, ws, self.stride)
        xs = _positions(wp, ws, self.stride)

        patches, grid_coords = [], []
        for gi, y0 in enumerate(ys):
            for gj, x0 in enumerate(xs):
                patches.append(padded[y0 : y0 + ws, x0 : x0 + ws, :])
                grid_coords.append((gi, gj))

        patches_arr = np.stack(patches, axis=0) if patches else np.empty((0, ws, ws, 3))
        return SampleResult(
            patches=patches_arr,
            grid_coords=grid_coords,
            grid_shape=(len(ys), len(xs)),
            orig_shape=(h, w),
            padded_shape=(hp, wp),
        )


def build_sampler(sampling_config) -> PatchSampler:
    """Factory: build a sampler from a ``SamplingConfig``."""
    if sampling_config.type == "grid":
        return GridSampler(
            patch_size=sampling_config.patch_size,
            pad_mode=sampling_config.pad_mode,
        )
    if sampling_config.type == "sliding":
        return SlidingWindowSampler(
            window_size=sampling_config.window_size,
            stride=sampling_config.stride,
            pad_mode=sampling_config.pad_mode,
        )
    raise NotImplementedError(f"Unknown sampler type '{sampling_config.type}'")
