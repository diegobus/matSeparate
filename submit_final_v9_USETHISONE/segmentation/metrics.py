"""
Semantic segmentation metrics for the material merging pipeline.

Compares predicted material **label maps** (MINC-style, integer class ids + a legend) to
ground-truth label maps, reporting per-class IoU, mean IoU, mean class accuracy, and
pixel accuracy -- the metrics MINC reports for full-scene material classification.

Because our taxonomy is not MINC's 23 categories, prediction and GT label maps usually
live in different id spaces. ``align_to_shared_space`` re-maps both into a common
**name-based** id space (with an optional name crosswalk), assigning any name present on
only one side to the ignored ``background`` id (0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

BACKGROUND_ID = 0
BACKGROUND_NAME = "background"


@dataclass
class SegMetrics:
    per_class_iou: Dict[str, float]
    per_class_acc: Dict[str, float]
    mean_iou: float
    mean_acc: float
    pixel_acc: float
    num_classes_present: int

    def to_dict(self) -> Dict:
        return {
            "mean_iou": self.mean_iou,
            "mean_acc": self.mean_acc,
            "pixel_acc": self.pixel_acc,
            "num_classes_present": self.num_classes_present,
            "per_class_iou": self.per_class_iou,
            "per_class_acc": self.per_class_acc,
        }


def build_confusion(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_ids: Tuple[int, ...] = (BACKGROUND_ID,),
) -> np.ndarray:
    """Confusion matrix (rows = GT class, cols = predicted class).

    Pixels whose GT id is in ``ignore_ids`` are excluded. ``num_classes`` is the size of
    the (shared) id space *including* the background id at 0.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    valid = np.ones(gt.shape, dtype=bool)
    for ig in ignore_ids:
        valid &= gt != ig
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    # clip stray predicted ids into range so bincount stays bounded
    p = np.clip(p, 0, num_classes - 1)
    g = np.clip(g, 0, num_classes - 1)
    idx = g * num_classes + p
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def metrics_from_confusion(
    cm: np.ndarray,
    class_names: List[str],
    ignore_ids: Tuple[int, ...] = (BACKGROUND_ID,),
) -> SegMetrics:
    """Derive IoU / accuracy metrics from a confusion matrix.

    Classes are scored only if they are *present in the GT* (their row has support) and
    not in ``ignore_ids``.
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    gt_totals = cm.sum(axis=1)
    pred_totals = cm.sum(axis=0)
    denom = gt_totals + pred_totals - tp

    present = gt_totals > 0
    for ig in ignore_ids:
        if 0 <= ig < len(present):
            present[ig] = False

    per_class_iou: Dict[str, float] = {}
    per_class_acc: Dict[str, float] = {}
    ious, accs = [], []
    for c in range(len(class_names)):
        if not present[c]:
            continue
        iou = tp[c] / denom[c] if denom[c] > 0 else 0.0
        acc = tp[c] / gt_totals[c] if gt_totals[c] > 0 else 0.0
        per_class_iou[class_names[c]] = float(iou)
        per_class_acc[class_names[c]] = float(acc)
        ious.append(iou)
        accs.append(acc)

    scored_mask = present.copy()
    pixel_correct = tp[scored_mask].sum()
    pixel_total = gt_totals[scored_mask].sum()
    pixel_acc = float(pixel_correct / pixel_total) if pixel_total > 0 else 0.0

    return SegMetrics(
        per_class_iou=per_class_iou,
        per_class_acc=per_class_acc,
        mean_iou=float(np.mean(ious)) if ious else 0.0,
        mean_acc=float(np.mean(accs)) if accs else 0.0,
        pixel_acc=pixel_acc,
        num_classes_present=len(ious),
    )


def build_shared_space(
    pred_legend: Dict[str, str],
    gt_legend: Dict[str, str],
    crosswalk: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[int, int], Dict[int, int]]:
    """Build a shared name-based id space for two legends.

    Args:
        pred_legend / gt_legend: ``{id_str: name}`` legends (id 0 == background).
        crosswalk: optional mapping ``{pred_name: gt_name}`` to align differing
            vocabularies (e.g. Matador -> MINC). Names not mapped pass through unchanged.

    Returns:
        ``(class_names, pred_id_remap, gt_id_remap)`` where ``class_names[0] == background``
        and the remaps send original ids -> shared ids (unknown names -> 0).
    """
    crosswalk = crosswalk or {}

    def pred_name(name: str) -> str:
        return crosswalk.get(name, name)

    pred_names = {pred_name(n) for k, n in pred_legend.items() if int(k) != BACKGROUND_ID}
    gt_names = {n for k, n in gt_legend.items() if int(k) != BACKGROUND_ID}
    shared = sorted(pred_names & gt_names)

    class_names = [BACKGROUND_NAME] + shared
    name_to_shared = {name: i for i, name in enumerate(class_names)}

    pred_remap = {
        int(k): name_to_shared.get(pred_name(n), BACKGROUND_ID)
        for k, n in pred_legend.items()
    }
    gt_remap = {int(k): name_to_shared.get(n, BACKGROUND_ID) for k, n in gt_legend.items()}
    return class_names, pred_remap, gt_remap


def remap_label_map(label_map: np.ndarray, remap: Dict[int, int]) -> np.ndarray:
    out = np.zeros_like(label_map, dtype=np.int64)
    for src, dst in remap.items():
        out[label_map == src] = dst
    return out


# --------------------------------------------------------------------------- #
# Class-agnostic mask / partition similarity
#
# These compare two segmentations purely as partitions of the pixels -- the
# label *values* are irrelevant, only which pixels are grouped together. Use
# this to compare our masks to MINC's masks (or SAM's) when the classes do not
# need to match.
# --------------------------------------------------------------------------- #


@dataclass
class MaskAgreement:
    adjusted_rand_index: float  # 1 = identical partition, 0 = chance
    variation_of_information: float  # bits; 0 = identical, lower is better
    covering_gt_by_pred: float  # how well our regions cover each GT region [0,1]
    covering_pred_by_gt: float  # symmetric direction [0,1]
    mean_best_iou: float  # avg over GT regions of best-matching pred IoU [0,1]
    boundary_f1: float  # boundary precision/recall F1 at a pixel tolerance [0,1]
    num_gt_regions: int
    num_pred_regions: int

    def to_dict(self) -> Dict:
        return {k: (float(v) if isinstance(v, float) else int(v)) for k, v in vars(self).items()}


def _relabel_consecutive(label_map: np.ndarray):
    flat = label_map.reshape(-1)
    uniq, inv = np.unique(flat, return_inverse=True)
    return inv.reshape(label_map.shape), len(uniq)


def to_connected_components(label_map: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Split a (semantic) label map into per-region instance ids via connected components."""
    from scipy import ndimage

    structure = (
        ndimage.generate_binary_structure(2, 2)
        if connectivity == 8
        else ndimage.generate_binary_structure(2, 1)
    )
    out = np.zeros(label_map.shape, dtype=np.int64)
    next_id = 1
    for c in np.unique(label_map):
        comp, n = ndimage.label(label_map == c, structure=structure)
        for k in range(1, n + 1):
            out[comp == k] = next_id
            next_id += 1
    return out


def _contingency(a: np.ndarray, b: np.ndarray):
    a, ka = _relabel_consecutive(a)
    b, kb = _relabel_consecutive(b)
    idx = a.reshape(-1).astype(np.int64) * kb + b.reshape(-1).astype(np.int64)
    cont = np.bincount(idx, minlength=ka * kb).reshape(ka, kb).astype(np.float64)
    return cont


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    cont = _contingency(a, b)
    n = cont.sum()
    if n <= 1:
        return 1.0

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_comb = comb2(cont).sum()
    sum_a = comb2(cont.sum(axis=1)).sum()
    sum_b = comb2(cont.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(n)
    max_index = 0.5 * (sum_a + sum_b)
    if max_index == expected:
        return 1.0
    return float((sum_comb - expected) / (max_index - expected))


def variation_of_information(a: np.ndarray, b: np.ndarray) -> float:
    cont = _contingency(a, b)
    n = cont.sum()
    if n == 0:
        return 0.0
    pij = cont / n
    a_marg = pij.sum(axis=1)
    b_marg = pij.sum(axis=0)

    def ent(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    h_a, h_b = ent(a_marg), ent(b_marg)
    nz = pij > 0
    mi = float((pij[nz] * np.log2(pij[nz] / (a_marg[:, None] * b_marg[None, :])[nz])).sum())
    return max(0.0, h_a + h_b - 2.0 * mi)


def _region_ious(gt: np.ndarray, pred: np.ndarray):
    cont = _contingency(gt, pred)  # (n_gt, n_pred) intersection counts
    gt_sizes = cont.sum(axis=1, keepdims=True)
    pred_sizes = cont.sum(axis=0, keepdims=True)
    union = gt_sizes + pred_sizes - cont
    iou = np.where(union > 0, cont / union, 0.0)
    return iou, cont, gt_sizes.ravel(), pred_sizes.ravel()


def segmentation_covering(gt: np.ndarray, pred: np.ndarray) -> float:
    """Covering of GT by pred: size-weighted mean of each GT region's best IoU."""
    iou, _, gt_sizes, _ = _region_ious(gt, pred)
    n = gt_sizes.sum()
    if n == 0:
        return 0.0
    best = iou.max(axis=1)
    return float((gt_sizes * best).sum() / n)


def mean_best_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    iou, _, _, _ = _region_ious(gt, pred)
    if iou.shape[0] == 0:
        return 0.0
    return float(iou.max(axis=1).mean())


def _boundaries(label_map: np.ndarray) -> np.ndarray:
    b = np.zeros(label_map.shape, dtype=bool)
    b[:, :-1] |= label_map[:, :-1] != label_map[:, 1:]
    b[:-1, :] |= label_map[:-1, :] != label_map[1:, :]
    return b


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance: int = 2) -> float:
    """Boundary F1: precision/recall of boundary pixels within ``tolerance`` px."""
    from scipy import ndimage

    gb = _boundaries(gt)
    pb = _boundaries(pred)
    if not gb.any() and not pb.any():
        return 1.0
    if not gb.any() or not pb.any():
        return 0.0
    # distance from every pixel to the nearest GT / pred boundary
    dist_to_gt = ndimage.distance_transform_edt(~gb)
    dist_to_pred = ndimage.distance_transform_edt(~pb)
    precision = (dist_to_gt[pb] <= tolerance).mean()  # pred boundary near a GT boundary
    recall = (dist_to_pred[gb] <= tolerance).mean()  # GT boundary near a pred boundary
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def mask_agreement(
    pred_label_map: np.ndarray,
    gt_label_map: np.ndarray,
    connected: bool = False,
    connectivity: int = 8,
    boundary_tolerance: int = 2,
) -> MaskAgreement:
    """Class-agnostic comparison of two segmentations as pixel partitions.

    Args:
        pred_label_map / gt_label_map: integer label maps (label values ignored).
        connected: if True, first split each label into connected components so the
            comparison is over spatial *regions* (instance-like) rather than label classes.
    """
    if pred_label_map.shape != gt_label_map.shape:
        raise ValueError(
            f"shape mismatch: pred {pred_label_map.shape} vs gt {gt_label_map.shape}"
        )
    pred = to_connected_components(pred_label_map, connectivity) if connected else pred_label_map
    gt = to_connected_components(gt_label_map, connectivity) if connected else gt_label_map

    return MaskAgreement(
        adjusted_rand_index=adjusted_rand_index(gt, pred),
        variation_of_information=variation_of_information(gt, pred),
        covering_gt_by_pred=segmentation_covering(gt, pred),
        covering_pred_by_gt=segmentation_covering(pred, gt),
        mean_best_iou=mean_best_iou(gt, pred),
        boundary_f1=boundary_f1(gt, pred, tolerance=boundary_tolerance),
        num_gt_regions=int(_relabel_consecutive(gt)[1]),
        num_pred_regions=int(_relabel_consecutive(pred)[1]),
    )
