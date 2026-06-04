"""
Configuration for the material segmentation / merging pipeline.

A single ``SegmentationConfig`` (composed of small nested dataclasses) holds every
knob in the pipeline. Construct it directly, from a dict, or from a YAML file
(``configs/segmentation.yaml``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class SamplingConfig:
    """How the image is broken into patches for the classifier."""

    type: str = "grid"  # "grid" (coarse, fast) | "sliding" (overlapping, much finer)
    patch_size: int = 224  # grid tile size (and classifier input size)
    pad_mode: str = "reflect"  # numpy pad mode used to make H,W a multiple of patch_size
    batch_size: int = 32  # patches per forward pass
    # sliding-window only:
    window_size: int = 96  # crop size in image pixels (resized to patch_size for model)
    stride: Optional[int] = 48  # window spacing; smaller -> finer/denser/slower
    # Optional patch-count bounds (sliding only). When set, the stride is adapted per image
    # so the total patch count stays in [min_patches, max_patches] regardless of image size
    # (prevents huge images from exploding patch counts, and tiny ones from under-sampling).
    min_patches: Optional[int] = None
    max_patches: Optional[int] = None


@dataclass
class UpsampleConfig:
    """Coarse grid -> dense per-pixel probability map."""

    mode: str = "bilinear"
    align_corners: bool = False
    renormalize: bool = True  # re-project onto the probability simplex after interpolation


@dataclass
class CRFConfig:
    """Dense CRF refinement of the dense probability map."""

    backend: str = "dense"  # "dense" (pydensecrf) | "superpixel" | "none"
    n_iterations: int = 7
    gaussian_sxy: float = 3.0
    bilateral_sxy: float = 60.0
    bilateral_srgb: float = 13.0
    gaussian_compat: float = 3.0
    bilateral_compat: float = 10.0
    taxonomy_aware_compat: bool = True
    # superpixel fallback only:
    superpixel_method: str = "slic"  # "slic" | "felzenszwalb"
    superpixel_n_segments: int = 400


@dataclass
class LevelConfig:
    """Which level of the taxonomy to segment at."""

    # "leaf" (default, most detailed) OR an integer depth (coarser).
    target: Union[str, int] = "leaf"
    shallow_leaf: str = "keep"  # leaves shallower than a requested depth keep their own label


@dataclass
class ObjectsConfig:
    """Label-map thresholding and connected-component object extraction."""

    # max-prob below this -> background/unknown (class 0).
    # Default 0.0 = never background: every pixel takes its highest-probability label.
    # Raise it (e.g. 0.5) to send low-confidence pixels to the background class instead.
    bg_threshold: float = 0.0
    connectivity: int = 8  # 4 or 8
    min_object_area: int = 64  # drop components smaller than this many pixels
    morph_close: bool = False  # bridge small gaps before connected components (off by default)
    morph_close_radius: int = 2


@dataclass
class OutputConfig:
    """What artifacts to write to disk."""

    write_color_viz: bool = True
    write_instance_pngs: bool = False
    minc_crosswalk: Optional[str] = None  # optional path to a Matador->MINC mapping json


@dataclass
class SegmentationConfig:
    """Top-level configuration for ``MaterialMerger``."""

    run_dir: Optional[str] = None  # HGNN training run directory (config/ckpt/node_index)
    # "cpu" by default: torch_geometric's scatter ops are broken on Apple MPS.
    device: str = "cpu"

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    upsample: UpsampleConfig = field(default_factory=UpsampleConfig)
    crf: CRFConfig = field(default_factory=CRFConfig)
    level: LevelConfig = field(default_factory=LevelConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentationConfig":
        data = dict(data or {})
        nested = {
            "sampling": SamplingConfig,
            "upsample": UpsampleConfig,
            "crf": CRFConfig,
            "level": LevelConfig,
            "objects": ObjectsConfig,
            "output": OutputConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in nested:
                sub_cls = nested[key]
                valid = {f.name for f in dataclasses.fields(sub_cls)}
                sub_kwargs = {k: v for k, v in (value or {}).items() if k in valid}
                kwargs[key] = sub_cls(**sub_kwargs)
            else:
                kwargs[key] = value
        valid_top = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in kwargs.items() if k in valid_top}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SegmentationConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
