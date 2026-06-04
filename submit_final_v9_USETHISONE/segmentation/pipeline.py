"""
``MaterialMerger`` -- end-to-end orchestration of the material segmentation pipeline.

    image -> grid patches -> HGNN leaf probs -> bilinear upsample -> dense CRF
          -> taxonomy level cut -> label map + connected-component objects -> export

The CRF-refined **leaf** probability map is cached on the result, so re-cutting to a
coarser level (``recut``) is a single cheap matmul -- no re-classification or CRF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
from PIL import Image

from segmentation import crf
from segmentation.classify import PatchClassifier
from segmentation.config import SegmentationConfig
from segmentation.formats import save_results
from segmentation.objects import Instance, build_label_map, extract_instances
from segmentation.patches import build_sampler
from segmentation.taxonomy_cut import apply_frontier, build_frontier
from segmentation.upsample import upsample_probs


def load_image_uint8(image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Coerce arbitrary image input to an ``(H, W, 3)`` uint8 RGB array."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)
    raise TypeError(f"Unsupported image type: {type(image)}")


@dataclass
class SegmentationResult:
    """Outputs of one segmentation, plus cached state enabling cheap re-cuts."""

    label_map: np.ndarray
    frontier_names: List[str]
    instances: List[Instance]
    level: Union[str, int]
    image_size: tuple
    confidence: np.ndarray = field(repr=False)
    refined_leaf_probs: np.ndarray = field(repr=False)
    image_uint8: np.ndarray = field(repr=False)
    meta: Dict = field(default_factory=dict)
    _merger: Optional["MaterialMerger"] = field(default=None, repr=False)

    def recut(self, level: Union[str, int]) -> "SegmentationResult":
        if self._merger is None:
            raise RuntimeError("result is not attached to a MaterialMerger")
        return self._merger.recut(self, level)

    def save(self, out_dir: Union[str, Path]) -> Dict[str, str]:
        cfg = self._merger.config if self._merger else None
        write_color = cfg.output.write_color_viz if cfg else True
        write_pngs = cfg.output.write_instance_pngs if cfg else False
        return save_results(
            out_dir=out_dir,
            label_map=self.label_map,
            frontier_names=self.frontier_names,
            instances=self.instances,
            level=self.level,
            image_size=self.image_size,
            meta=self.meta,
            write_color_viz=write_color,
            write_instance_pngs=write_pngs,
        )


class MaterialMerger:
    """Coordinates sampler, classifier, CRF, taxonomy cut, and object extraction."""

    def __init__(
        self,
        classifier: PatchClassifier,
        graph: nx.DiGraph,
        config: Optional[SegmentationConfig] = None,
    ):
        self.classifier = classifier
        self.graph = graph
        self.config = config or SegmentationConfig()
        self.sampler = build_sampler(self.config.sampling)
        self.leaf_names = classifier.leaf_names

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Union[str, Path],
        config: Optional[SegmentationConfig] = None,
        device: str = "auto",
    ) -> "MaterialMerger":
        import sys

        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from scripts.infer_api import HGNNInference
        from taxonomy.tree import get_taxonomy

        config = config or SegmentationConfig()
        api = HGNNInference.from_run_dir(run_dir, device=device)
        classifier = PatchClassifier.from_hgnn(api, batch_size=config.sampling.batch_size)

        graph = cls._load_graph(run_dir, repo_root, get_taxonomy)
        return cls(classifier=classifier, graph=graph, config=config)

    @staticmethod
    def _load_graph(run_dir, repo_root, get_taxonomy) -> nx.DiGraph:
        import yaml

        run_dir = Path(run_dir)
        with open(run_dir / "config.yaml") as f:
            run_config = yaml.safe_load(f)
        tax_path = run_config.get("data", {}).get("taxonomy_json")
        candidates = []
        if tax_path:
            candidates += [Path(tax_path), repo_root / tax_path]
        for cand in candidates:
            if cand and Path(cand).exists():
                return get_taxonomy(str(cand))
        return get_taxonomy()  # packaged default

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #

    def segment(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        level: Optional[Union[str, int]] = None,
    ) -> SegmentationResult:
        if level is None:
            level = self.config.level.target

        image_uint8 = load_image_uint8(image)
        h, w = image_uint8.shape[:2]

        p_grid = self.classifier.classify(self.sampler.sample(image_uint8))

        p_dense = upsample_probs(
            p_grid,
            target_hw=(h, w),
            mode=self.config.upsample.mode,
            align_corners=self.config.upsample.align_corners,
            renormalize=self.config.upsample.renormalize,
        )
        refined = crf.refine(
            image_uint8,
            p_dense,
            self.config.crf,
            leaf_names=self.leaf_names,
            graph=self.graph,
        )

        return self._finish(refined, image_uint8, level)

    def recut(
        self, result: SegmentationResult, level: Union[str, int]
    ) -> SegmentationResult:
        """Re-derive a (typically coarser) level from cached refined leaf probs."""
        return self._finish(result.refined_leaf_probs, result.image_uint8, level)

    def _finish(
        self,
        refined_leaf_probs: np.ndarray,
        image_uint8: np.ndarray,
        level: Union[str, int],
    ) -> SegmentationResult:
        cut = build_frontier(
            self.graph,
            self.leaf_names,
            level=level,
            shallow_leaf=self.config.level.shallow_leaf,
        )
        frontier = apply_frontier(refined_leaf_probs, cut)
        label_map, confidence = build_label_map(
            frontier, bg_threshold=self.config.objects.bg_threshold
        )
        instances = extract_instances(
            label_map,
            confidence,
            cut.frontier_names,
            connectivity=self.config.objects.connectivity,
            min_object_area=self.config.objects.min_object_area,
            morph_close=self.config.objects.morph_close,
            morph_close_radius=self.config.objects.morph_close_radius,
        )
        h, w = image_uint8.shape[:2]
        meta = {
            "level": level,
            "bg_threshold": self.config.objects.bg_threshold,
            "num_classes": len(cut.frontier_names),
            "num_objects": len(instances),
            "crf_backend": self.config.crf.backend,
        }
        return SegmentationResult(
            label_map=label_map,
            frontier_names=cut.frontier_names,
            instances=instances,
            level=level,
            image_size=(h, w),
            confidence=confidence,
            refined_leaf_probs=refined_leaf_probs,
            image_uint8=image_uint8,
            meta=meta,
            _merger=self,
        )
