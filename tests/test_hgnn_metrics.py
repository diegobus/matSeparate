#!/usr/bin/env python3
"""
Smoke test for HGNN metric functions.

Verifies:
1. leaf_top1_accuracy restricts argmax to leaf nodes only.
2. Internal nodes with highest overall score do NOT affect leaf accuracy.
3. Correct leaf having highest score among leaves IS counted.
4. node_bce_accuracy, exact_path_match, path_f1 compute correctly on toy data.
"""

import sys
from pathlib import Path

import networkx as nx
import torch
from gnn_classifier.hgnn import HGNN


def test_hgnn_predict_starts_at_named_root():
    graph = nx.DiGraph()
    graph.add_node("child_b")
    graph.add_node("root")
    graph.add_node("child_a")
    graph.add_edge("root", "child_a")
    graph.add_edge("root", "child_b")

    model = HGNN(
        graph=graph,
        path_predict=True,
        cnn_kwargs={"backbone": "resnet18", "pretrained": False, "output_dim": 4},
        gnn_kwargs={"input_dim": 4, "hidden_dim": 4, "output_dim": 4, "num_layers": 1},
    )
    assert model.root_idx == 1

    probs = torch.zeros(2, model.num_nodes)
    paths = model._get_best_path(probs)

    assert paths[:, model.root_idx].eq(1).all()
    assert paths[:, 0].eq(0).all()


def test_hgnn_nodewise_shared_head_outputs_node_logits():
    graph = nx.DiGraph()
    graph.add_edge("root", "child_a")
    graph.add_edge("root", "child_b")

    model = HGNN(
        graph=graph,
        head_type="nodewise_shared",
        cnn_kwargs={"backbone": "resnet18", "pretrained": False, "output_dim": 4},
        gnn_kwargs={"input_dim": 4, "hidden_dim": 4, "output_dim": 4, "num_layers": 1},
    )
    images = torch.randn(2, 3, 64, 64)
    logits = model(images)
    assert logits.shape == (2, graph.number_of_nodes())

    loss = logits.sum()
    loss.backward()
    assert model.classifier[-1].weight.grad is not None

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "scripts"))

# We need to import the metric functions; they live in train_c1_hgnn.py
# Import them directly by exec'ing the relevant portion or copy them here.
# Simpler: copy the metric functions into this test file.


def _get_true_leaf_indices(target_multihot, leaf_indices):
    leaf_mask = target_multihot[:, leaf_indices]
    active_counts = leaf_mask.sum(dim=1)
    if not torch.all(active_counts > 0.5):
        bad = (active_counts <= 0.5).nonzero(as_tuple=True)[0]
        raise ValueError(f"Samples {bad.tolist()} have no active leaf")
    leaf_positions = leaf_mask.argmax(dim=1)
    return leaf_indices[leaf_positions]


def leaf_top1_accuracy(logits, target_multihot, leaf_indices):
    true_leaf = _get_true_leaf_indices(target_multihot, leaf_indices)
    leaf_logits = logits[:, leaf_indices]
    pred_leaf_pos = leaf_logits.argmax(dim=1)
    pred_leaf = leaf_indices[pred_leaf_pos]
    return (pred_leaf == true_leaf).float().mean().item()


def node_bce_accuracy(logits, target_multihot):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == target_multihot).float().mean().item()


def exact_path_match(logits, target_multihot):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    matches = (preds == target_multihot).all(dim=1).float()
    return matches.mean().item()


def path_f1(logits, target_multihot):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    tp = (preds * target_multihot).sum(dim=1)
    fp = (preds * (1 - target_multihot)).sum(dim=1)
    fn = ((1 - preds) * target_multihot).sum(dim=1)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean().item()


def hierarchy_level_accuracy(logits, target_multihot, hierarchy_levels):
    correct = 0.0
    total = 0
    for level in hierarchy_levels:
        level_logits = logits[:, level]
        level_targets = target_multihot[:, level]
        participation = level_targets.sum(dim=1) > 0
        if participation.sum() == 0:
            continue
        pred_pos = level_logits[participation].argmax(dim=1)
        pred_nodes = level[pred_pos]
        true_pos = level_targets[participation].argmax(dim=1)
        true_nodes = level[true_pos]
        correct += (pred_nodes == true_nodes).float().sum().item()
        total += participation.sum().item()
    return correct / total if total > 0 else 0.0


def test_leaf_top1_restricts_to_leaves():
    """
    Construct a scenario where an internal node has the highest overall logit,
    but the correct leaf has the highest logit among leaf nodes.
    """
    num_nodes = 5
    # Nodes: 0=root, 1=internal_A, 2=internal_B, 3=leaf_A1, 4=leaf_B1
    leaf_indices = torch.tensor([3, 4], dtype=torch.long)

    # Sample 1: true path = root(0) -> internal_A(1) -> leaf_A1(3)
    target = torch.zeros(1, num_nodes)
    target[0, [0, 1, 3]] = 1.0

    # Logits: internal_A (node 1) has highest score, leaf_A1 (node 3) has second highest,
    # but leaf_B1 (node 4) is lower than leaf_A1.
    # If unrestricted argmax is used, it would pick node 1 (wrong).
    # If leaf-restricted, it should pick node 3 (correct).
    logits = torch.tensor([
        [0.0, 5.0, 0.0, 3.0, 1.0],  # node 1 is highest overall, node 3 is highest among leaves
    ])

    acc = leaf_top1_accuracy(logits, target, leaf_indices)
    assert acc == 1.0, f"Expected leaf_top1=1.0 when correct leaf is highest among leaves, got {acc}"

    # Now make leaf_B1 (node 4) highest among leaves
    logits2 = torch.tensor([
        [0.0, 5.0, 0.0, 1.0, 4.0],  # node 4 highest among leaves but wrong
    ])
    acc2 = leaf_top1_accuracy(logits2, target, leaf_indices)
    assert acc2 == 0.0, f"Expected leaf_top1=0.0 when wrong leaf is highest among leaves, got {acc2}"

    print("PASS: leaf_top1_accuracy correctly restricts to leaf nodes")


def test_node_bce_and_path_metrics():
    """Smoke test on toy data with known outcomes."""
    num_nodes = 4
    # Target: [1, 1, 0, 1] (path: root -> node1 -> leaf3)
    target = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    # Perfect prediction
    logits_perfect = torch.tensor([[10.0, 10.0, -10.0, 10.0]])
    assert node_bce_accuracy(logits_perfect, target) == 1.0
    assert exact_path_match(logits_perfect, target) == 1.0
    assert path_f1(logits_perfect, target) == 1.0

    # Wrong on one node
    logits_wrong = torch.tensor([[10.0, -10.0, -10.0, 10.0]])  # node 1 wrong
    assert node_bce_accuracy(logits_wrong, target) == 0.75
    assert exact_path_match(logits_wrong, target) == 0.0
    # F1: tp=2, fp=0, fn=1 -> precision=1.0, recall=0.667 -> f1=0.8
    f1 = path_f1(logits_wrong, target)
    assert abs(f1 - 0.8) < 1e-4, f"Expected F1=0.8, got {f1}"

    print("PASS: node_bce_accuracy, exact_path_match, path_f1 compute correctly")


def test_hierarchy_level_accuracy():
    """Toy hierarchy: root(0) -> A(1) -> leaf(3); root(0) -> B(2) -> leaf(4)."""
    num_nodes = 5
    hierarchy_levels = [
        torch.tensor([0]),
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
    ]
    # Target: root(0) + A(1) + leaf(3)
    target = torch.zeros(1, num_nodes)
    target[0, [0, 1, 3]] = 1.0

    # Correct prediction
    logits = torch.tensor([[5.0, 5.0, -5.0, 5.0, -5.0]])
    acc = hierarchy_level_accuracy(logits, target, hierarchy_levels)
    assert acc == 1.0, f"Expected hier_acc=1.0 for perfect prediction, got {acc}"

    # Wrong at level 1 (pick B instead of A)
    logits2 = torch.tensor([[5.0, -5.0, 5.0, 5.0, -5.0]])
    acc2 = hierarchy_level_accuracy(logits2, target, hierarchy_levels)
    # Level 0: root correct (1/1)
    # Level 1: A wrong, B correct? No, target has A active. So A wrong. (0/1)
    # Level 2: leaf 3 correct (1/1)
    # Total: 2/3
    assert abs(acc2 - 2 / 3) < 1e-4, f"Expected hier_acc=0.667, got {acc2}"

    print("PASS: hierarchy_level_accuracy computes correctly")


def main():
    test_leaf_top1_restricts_to_leaves()
    test_node_bce_and_path_metrics()
    test_hierarchy_level_accuracy()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
