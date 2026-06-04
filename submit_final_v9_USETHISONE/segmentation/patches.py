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
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SampleResult:
    """Output of a ``PatchSampler``."""

    patches: np.ndarray  # (N, patch_size, patch_size, 3)
    grid_coords: List[Tuple[int, int]]  # (gi, gj) per patch, len == N
    grid_shape: Tuple[int, int]  # (gh, gw)
    orig_shape: Tuple[int, int]  # (H, W) of the input image
    padded_shape: Tuple[int, int]  # (Hp, Wp) after padding
    window_size: int = -1  # effective window (px) actually used (sliding only)
    stride: int = -1  # effective stride (px) actually used (sliding only)

    @property
    def num_patches(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]


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
    stride = max(1, stride)
    pos = list(range(0, length - window + 1, stride))
    if pos[-1] != length - window:
        pos.append(length - window)
    return pos


def _count(length: int, window: int, stride: int) -> int:
    """Number of window positions ``_positions`` would produce (without building them)."""
    if length <= window:
        return 1
    stride = max(1, stride)
    n = len(range(0, length - window + 1, stride))
    if (n - 1) * stride != length - window:
        n += 1
    return n


_MIN_WINDOW = 16  # never shrink a window below this when chasing min_patches


def _resolve_window_stride(
    h: int,
    w: int,
    window: int,
    stride: int,
    min_patches: Optional[int],
    max_patches: Optional[int],
) -> Tuple[int, int]:
    """Pick an effective ``(window, stride)`` so the patch count lands in range.

    Strategy (only the stride moves in the common case):
      * too many patches  -> increase stride (coarser) until ``<= max_patches``
      * too few patches    -> decrease stride (finer) down to 1; if that still isn't
        enough (image barely larger than the window), shrink the window as a last resort.
    ``max_patches`` is treated as a hard cap: if min/max can't both be met, the cap wins.
    When neither bound is set, the configured ``(window, stride)`` is returned unchanged.
    """
    if min_patches is None and max_patches is None:
        return window, stride if (stride and stride > 0) else max(1, window // 2)

    if min_patches is not None and max_patches is not None and min_patches > max_patches:
        raise ValueError(f"min_patches ({min_patches}) > max_patches ({max_patches})")

    window = max(1, min(window, h, w))
    stride = stride if (stride and stride > 0) else max(1, window // 2)
    s_cap = max(h, w)  # any larger stride yields a single position per dim

    def total(ws: int, s: int) -> int:
        return _count(h, ws, s) * _count(w, ws, s)

    if max_patches is not None:
        while total(window, stride) > max_patches and stride < s_cap:
            stride += 1

    if min_patches is not None:
        while total(window, stride) < min_patches and stride > 1:
            stride -= 1
        # still short at stride 1: shrink the window so more positions fit
        while total(window, stride) < min_patches and window > _MIN_WINDOW:
            window -= 1
        # shrinking the window can overshoot the cap; re-coarsen if needed
        if max_patches is not None:
            while total(window, stride) > max_patches and stride < s_cap:
                stride += 1

    return window, stride


class SlidingWindowSampler(PatchSampler):
    """Overlapping windows of ``window_size`` at spacing ``stride``.

    Each window is later resized to the classifier's input size, so a *small* window
    yields both a denser prediction grid (finer segmentation) and a closer-up material
    view (closer to the patch-classifier training distribution). The window predictions
    are laid out on a regular ``(n_rows, n_cols)`` grid (one cell per window position) and
    bilinearly upsampled downstream -- with overlap this grid is much denser than the
    non-overlapping ``GridSampler``.
    """

    def __init__(
        self,
        window_size: int = 96,
        stride: int = 48,
        pad_mode: str = "reflect",
        min_patches: Optional[int] = None,
        max_patches: Optional[int] = None,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.stride = stride if stride and stride > 0 else max(1, window_size // 2)
        self.pad_mode = pad_mode
        self.min_patches = min_patches
        self.max_patches = max_patches

    def _pad(self, image: np.ndarray, hp: int, wp: int) -> np.ndarray:
        h, w = image.shape[:2]
        pad_h, pad_w = hp - h, wp - w
        if not (pad_h or pad_w):
            return image
        mode = self.pad_mode
        if mode == "reflect" and (pad_h >= h or pad_w >= w):
            mode = "edge"
        return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)

    def _grid(self, h: int, w: int):
        """Resolve the (window, stride) and the window start coords for this image."""
        ws, stride = _resolve_window_stride(
            h, w, self.window_size, self.stride, self.min_patches, self.max_patches
        )
        hp, wp = max(h, ws), max(w, ws)
        ys = _positions(hp, ws, stride)
        xs = _positions(wp, ws, stride)
        return ws, stride, hp, wp, ys, xs

    def sample(self, image: np.ndarray) -> SampleResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

        h, w = image.shape[:2]
        ws, stride, hp, wp, ys, xs = self._grid(h, w)
        padded = self._pad(image, hp, wp)

        patches, grid_coords = [], []
        for gi, y0 in enumerate(ys):
            for gj, x0 in enumerate(xs):
                patches.append(padded[y0 : y0 + ws, x0 : x0 + ws, :])
                grid_coords.append((gi, gj))

        patches_arr = np.stack(patches, axis=0) if patches else np.empty((0, ws, ws, 3))
        if self.min_patches is not None or self.max_patches is not None:
            print(
                f"[sampler] {len(ys)}x{len(xs)} = {len(ys) * len(xs)} patches "
                f"(window={ws}px, stride={stride}px) for {h}x{w} image"
            )
        return SampleResult(
            patches=patches_arr,
            grid_coords=grid_coords,
            grid_shape=(len(ys), len(xs)),
            orig_shape=(h, w),
            padded_shape=(hp, wp),
            window_size=ws,
            stride=stride,
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
            min_patches=getattr(sampling_config, "min_patches", None),
            max_patches=getattr(sampling_config, "max_patches", None),
        )
    raise NotImplementedError(f"Unknown sampler type '{sampling_config.type}'")
