"""
Serialization of segmentation results.

Two coordinated output families (see MERGE.md §5[7]):

* **MINC-style semantic segmentation** -- ``label_map.png`` (single-channel class ids,
  ``0`` = background) plus ``labels.json`` (id -> material legend + metadata). Optional
  ``label_map_color.png`` for visualization.
* **COCO-style instances** -- ``instances.json`` (per-object bbox/area/score + RLE mask),
  comparable to how SAM outputs are stored. Optional per-object PNGs.

The COCO RLE encoder is self-contained (uncompressed, column-major counts) so
``pycocotools`` is not required; if present it can decode these directly.
"""

from __future__ import annotations

import colorsys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from segmentation.objects import BACKGROUND_ID, BACKGROUND_NAME, Instance


def encode_rle(mask: np.ndarray) -> Dict:
    """COCO-style uncompressed RLE (column-major). Returns ``{size, counts}``."""
    mask = np.asarray(mask, dtype=np.uint8)
    h, w = mask.shape
    flat = mask.flatten(order="F")
    if flat.size == 0:
        return {"size": [h, w], "counts": [0]}
    change = np.where(np.diff(flat) != 0)[0] + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).astype(int)
    counts = runs.tolist()
    if flat[0] == 1:  # COCO counts must start with a run of 0s
        counts = [0] + counts
    return {"size": [int(h), int(w)], "counts": counts}


def decode_rle(rle: Dict) -> np.ndarray:
    """Inverse of ``encode_rle`` (handy for tests/visualization)."""
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for run in counts:
        if val == 1:
            flat[pos : pos + run] = 1
        pos += run
        val ^= 1
    return flat.reshape((h, w), order="F")


def build_palette(num_classes: int) -> np.ndarray:
    """Deterministic ``(num_classes + 1, 3)`` uint8 palette; index 0 = black background."""
    palette = np.zeros((num_classes + 1, 3), dtype=np.uint8)
    for k in range(1, num_classes + 1):
        hue = ((k - 1) * 0.61803398875) % 1.0  # golden-ratio hue spacing
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
        palette[k] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def colorize_label_map(label_map: np.ndarray, num_classes: int) -> np.ndarray:
    palette = build_palette(num_classes)
    return palette[np.clip(label_map, 0, num_classes)]


def build_legend(frontier_names: List[str]) -> Dict[str, str]:
    legend = {str(BACKGROUND_ID): BACKGROUND_NAME}
    for k, name in enumerate(frontier_names, start=1):
        legend[str(k)] = name
    return legend


def instances_to_coco(
    instances: List[Instance],
    image_size: Tuple[int, int],
    level,
    image_id: int = 0,
) -> Dict:
    h, w = image_size
    annotations = []
    for inst in instances:
        annotations.append(
            {
                "id": inst.id,
                "image_id": image_id,
                "category_id": inst.category_id,
                "material": inst.material,
                "bbox": list(inst.bbox),
                "area": int(inst.area),
                "score": float(inst.score),
                "segmentation": encode_rle(inst.mask),
            }
        )
    return {
        "image": {"id": image_id, "height": int(h), "width": int(w)},
        "level": level,
        "annotations": annotations,
    }


def save_results(
    out_dir,
    label_map: np.ndarray,
    frontier_names: List[str],
    instances: List[Instance],
    level,
    image_size: Tuple[int, int],
    meta: Optional[Dict] = None,
    write_color_viz: bool = True,
    write_instance_pngs: bool = False,
) -> Dict[str, str]:
    """Write all artifacts to ``out_dir``. Returns a dict of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(frontier_names)
    written: Dict[str, str] = {}

    if num_classes > 255:
        raise ValueError("label map PNG supports up to 255 material classes")

    # MINC-style semantic label map (single channel).
    label_path = out_dir / "label_map.png"
    Image.fromarray(label_map.astype(np.uint8), mode="L").save(label_path)
    written["label_map"] = str(label_path)

    # Legend + metadata.
    legend = build_legend(frontier_names)
    labels_payload = {"level": level, "legend": legend}
    if meta:
        labels_payload["meta"] = meta
    labels_path = out_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_payload, f, indent=2)
    written["labels"] = str(labels_path)

    # COCO-style instances.
    coco = instances_to_coco(instances, image_size, level)
    inst_path = out_dir / "instances.json"
    with open(inst_path, "w") as f:
        json.dump(coco, f, indent=2)
    written["instances"] = str(inst_path)

    # Optional color visualization.
    if write_color_viz:
        color = colorize_label_map(label_map, num_classes)
        color_path = out_dir / "label_map_color.png"
        Image.fromarray(color, mode="RGB").save(color_path)
        written["label_map_color"] = str(color_path)

    # Optional per-object binary masks.
    if write_instance_pngs and instances:
        inst_dir = out_dir / "instances"
        inst_dir.mkdir(exist_ok=True)
        for inst in instances:
            p = inst_dir / f"{inst.id:04d}_{inst.material}.png"
            Image.fromarray((inst.mask.astype(np.uint8) * 255), mode="L").save(p)
        written["instance_pngs"] = str(inst_dir)

    return written
