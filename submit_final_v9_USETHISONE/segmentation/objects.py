"""
From a frontier probability map to (a) a semantic label map and (b) per-object instance
masks.

An *object* is a spatially-connected component of a single material at the chosen level
(so a wooden bowl touching a wooden table is one object). Low-confidence pixels (max class
probability below ``bg_threshold``) are assigned to the reserved ``background/unknown``
class, id ``0``. Material classes get ids ``1..K``.

Connected components use ``scipy.ndimage`` (always available); ``skimage`` is not required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import ndimage

BACKGROUND_ID = 0
BACKGROUND_NAME = "background"


@dataclass
class Instance:
    """A single connected-component material object."""

    id: int
    material: str
    category_id: int  # frontier class id (1..K)
    bbox: Tuple[int, int, int, int]  # COCO xywh
    area: int
    score: float
    mask: np.ndarray = field(repr=False)  # (H, W) bool


def build_label_map(
    frontier_probs: np.ndarray,
    bg_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Argmax a ``(H, W, K)`` frontier map into a ``(H, W)`` label map.

    Returns ``(label_map, confidence)`` where ``label_map`` has ``1..K`` for materials and
    ``0`` for background/unknown, and ``confidence`` is the per-pixel max class probability.

    With the default ``bg_threshold == 0.0`` no pixel is ever sent to background: every
    pixel simply takes its highest-probability label. A positive threshold reinstates the
    background/unknown class for pixels whose top probability falls below it.
    """
    if frontier_probs.ndim != 3:
        raise ValueError(f"expected (H, W, K), got {frontier_probs.shape}")
    conf = frontier_probs.max(axis=-1)
    arg = frontier_probs.argmax(axis=-1).astype(np.int32)
    label_map = arg + 1  # shift so material ids start at 1
    if bg_threshold > 0.0:
        label_map[conf < bg_threshold] = BACKGROUND_ID
    return label_map.astype(np.int32), conf.astype(np.float32)


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    raise ValueError("connectivity must be 4 or 8")


def extract_instances(
    label_map: np.ndarray,
    confidence: np.ndarray,
    frontier_names: List[str],
    connectivity: int = 8,
    min_object_area: int = 64,
    morph_close: bool = False,
    morph_close_radius: int = 2,
) -> List[Instance]:
    """Connected-component objects from a label map, one per material region."""
    structure = _structure(connectivity)
    instances: List[Instance] = []
    next_id = 1

    num_classes = len(frontier_names)
    for class_id in range(1, num_classes + 1):
        class_mask = label_map == class_id
        if not class_mask.any():
            continue

        if morph_close and morph_close_radius > 0:
            r = morph_close_radius
            selem = np.ones((2 * r + 1, 2 * r + 1), dtype=bool)
            class_mask = ndimage.binary_closing(class_mask, structure=selem)

        labeled, n_comp = ndimage.label(class_mask, structure=structure)
        if n_comp == 0:
            continue

        objects = ndimage.find_objects(labeled)
        for comp_idx, sl in enumerate(objects, start=1):
            if sl is None:
                continue
            comp_mask = labeled == comp_idx
            area = int(comp_mask.sum())
            if area < min_object_area:
                continue

            ys, xs = sl[0], sl[1]
            y0, x0 = ys.start, xs.start
            h = ys.stop - ys.start
            w = xs.stop - xs.start
            score = float(confidence[comp_mask].mean())

            instances.append(
                Instance(
                    id=next_id,
                    material=frontier_names[class_id - 1],
                    category_id=class_id,
                    bbox=(int(x0), int(y0), int(w), int(h)),
                    area=area,
                    score=score,
                    mask=comp_mask,
                )
            )
            next_id += 1

    return instances
