from itertools import chain
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def hierarchical_softmax_loss(
    logits,
    targets,
    hierarchy_levels,
    agg: str = "avg",
    reduction: str = "mean",
    enforce_participation: bool = True,
):
    # logits: (batch_size, num_classes)
    # targets: (batch_size, num_classes) - one-hot or multi-label
    # hierarchy_levels: List of class indices for each level

    if agg == "avg":
        coeff = lambda l: 1 / len(hierarchy_levels)
    elif agg == "sum":
        coeff = lambda l: 1
    elif agg == "level":
        coeff = lambda l: (l + 1)
    elif agg == "level_norm":
        coeff = lambda l: (l + 1) / len(hierarchy_levels)
    elif agg == "level_size":
        coeff = lambda l: len(hierarchy_levels[l]) / len(
            list(chain.from_iterable(hierarchy_levels))
        )
    else:
        raise ValueError(f"Invalid agg function: {agg}.")

    if reduction == "none":
        loss = torch.zeros(logits.size(0), device=logits.device)
    else:
        loss = 0.0
    for l, level in enumerate(hierarchy_levels):
        level_logits = logits[:, level]  # Select logits for the current level
        level_targets = targets[:, level]

        participation = None
        if enforce_participation:
            participation = level_targets.sum(dim=1) > 0
            level_logits = level_logits[participation]
            level_targets = level_targets[participation]

        ce = F.cross_entropy(
            level_logits, level_targets.argmax(dim=1), reduction=reduction
        )

        if reduction == "none":
            if enforce_participation:
                loss[participation] += coeff(l) * ce
            else:
                loss += coeff(l) * ce
        else:
            loss += coeff(l) * ce
    return loss


def greedy_loss(
    node_logits: torch.Tensor,
    labels_multihot: torch.Tensor,
    hierarchy_levels: List[np.ndarray],
    mode: str = "combined",
    parent_child_pairs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Hierarchical classification loss with multiple strategies.

    Args:
        mode:
            - "max": winner-take-all over path and hierarchy loss per sample.
            - "combined": weighted sum of path and hierarchy loss.
            - "parent_child_consistency": combined + hinge penalty when child
              logit exceeds parent logit (requires parent_child_pairs).
        parent_child_pairs: (num_edges, 2) tensor of [parent_idx, child_idx]
            pairs from the taxonomy graph. Required for parent_child_consistency.
    """
    # Common components
    path_loss = F.binary_cross_entropy_with_logits(
        node_logits,
        labels_multihot,
        reduction="none",
    ).mean(dim=1)

    hierarchy_loss = hierarchical_softmax_loss(
        node_logits,
        labels_multihot,
        hierarchy_levels,
        agg="level_size",
        reduction="none",
    )

    if mode == "max":
        loss = (
            torch.cat(
                (path_loss[..., None], hierarchy_loss[..., None]),
                dim=1,
            )
            .max(dim=1)
            .values.mean(dim=0)
        )
    elif mode == "combined":
        loss = 0.6 * path_loss.mean() + 0.4 * hierarchy_loss.mean()
    elif mode == "parent_child_consistency":
        if parent_child_pairs is None or parent_child_pairs.numel() == 0:
            raise ValueError(
                "parent_child_pairs is required for parent_child_consistency mode"
            )
        parent_child_pairs = parent_child_pairs.to(node_logits.device)
        parent_indices = parent_child_pairs[:, 0]
        child_indices = parent_child_pairs[:, 1]

        parent_logits = node_logits[:, parent_indices]  # [B, E]
        child_logits = node_logits[:, child_indices]    # [B, E]

        # Penalize when child logit exceeds parent logit
        consistency_loss = torch.relu(child_logits - parent_logits).mean()

        loss = (
            0.5 * path_loss.mean()
            + 0.3 * hierarchy_loss.mean()
            + 0.2 * consistency_loss
        )
    else:
        raise ValueError(
            f"Unknown loss mode: {mode}. "
            "Choose from 'max', 'combined', 'parent_child_consistency'."
        )

    return loss
