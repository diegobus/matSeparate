"""
Upsample the coarse per-patch probability grid to a dense per-pixel probability map.

Bilinear interpolation is applied **in probability space** (per leaf channel), then the
result is re-projected onto the simplex (bilinear mixing can break normalization slightly).
This mirrors MINC's "upsample the probability map" step before the CRF.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def upsample_probs(
    p_grid: np.ndarray,
    target_hw: Tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool = False,
    renormalize: bool = True,
) -> np.ndarray:
    """Upsample ``(gh, gw, L)`` -> ``(H, W, L)`` per-channel.

    Args:
        p_grid: coarse grid of leaf probabilities, shape (gh, gw, L).
        target_hw: (H, W) of the output dense map (typically the original image size).
        mode: interpolation mode ("bilinear" | "bicubic" | "nearest").
        align_corners: passed to ``F.interpolate`` for bilinear/bicubic.
        renormalize: re-project each pixel onto the probability simplex.
    """
    if p_grid.ndim != 3:
        raise ValueError(f"p_grid must be (gh, gw, L), got {p_grid.shape}")
    h, w = target_hw
    gh, gw, num_leaves = p_grid.shape

    tensor = torch.from_numpy(np.ascontiguousarray(p_grid, dtype=np.float32))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, L, gh, gw)

    interp_kwargs = {"size": (h, w), "mode": mode}
    if mode in ("bilinear", "bicubic"):
        interp_kwargs["align_corners"] = align_corners
    dense = F.interpolate(tensor, **interp_kwargs)  # (1, L, H, W)

    dense = dense.squeeze(0).permute(1, 2, 0).contiguous().numpy()  # (H, W, L)
    dense = np.clip(dense, 0.0, None)

    if renormalize:
        denom = dense.sum(axis=-1, keepdims=True)
        denom = np.where(denom <= 0, 1.0, denom)
        dense = dense / denom

    return dense.astype(np.float32)
