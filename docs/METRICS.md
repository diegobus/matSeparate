# HGNN Evaluation Metrics Reference

This document defines every metric reported by `train_c1_hgnn.py` and `eval_c1_hgnn.py`, how each is computed, and what it tells you about model performance.

All metrics are computed from a single forward pass that outputs logits for all 58 taxonomy nodes. The target for each sample is a **multi-hot hierarchy path** (e.g., `root → solid → abiotic → metal → generic_metal`), not a single class label.

---

## Metric Definitions

### 1. Loss (training/validation)

**What it is:** The `greedy_loss` combines two objectives:
- **BCE loss** on the full 58-node multi-hot path (sigmoid cross-entropy)
- **Hierarchical softmax loss** on each taxonomy level, weighted by level size

**What it tells you:** The primary training objective. Lower is better. If loss is decreasing but leaf accuracy is flat, the model is learning internal node structure before leaf precision.

---

### 2. Leaf Accuracy (`leaf_acc`)

**What it is:** Top-1 accuracy restricted to the **37 C1 leaf nodes only**.

**How it's computed:**
1. For each sample, identify the true leaf node from the target multi-hot path (the active node with `out_degree == 0`).
2. Extract logits for only the 37 leaf nodes.
3. Predict the leaf with the highest logit among leaves.
4. Compare to the true leaf.

**Why it matters:** This is the primary classification metric. A sample's path prediction can be mostly correct (all internal nodes right) but wrong on the leaf — and that leaf is the actual C1 label we care about. By restricting argmax to leaves, we avoid the old bug where an internal node (e.g., "solid") could have the highest overall logit and be counted as "correct."

**Typical values:**
- Random guess among 37 leaves: ~2.7%
- Untrained model (dry-run): ~0%
- After 2 epochs with fixes: ~70–75%

---

### 3. Node Accuracy (`node_acc`)

**What it is:** Per-node thresholded accuracy across all 58 taxonomy nodes.

**How it's computed:**
1. Apply sigmoid to logits → probabilities.
2. Threshold at 0.5 to get binary predictions per node.
3. Compute fraction of correct predictions across all nodes and all samples.

**What it tells you:** How well the model predicts each individual node's presence/absence in the path. High node accuracy (>90%) is expected because most paths share common ancestors (e.g., "root" and "solid" appear in almost every sample). Low node accuracy indicates the model hasn't learned even coarse structure.

**Typical values:**
- Untrained: ~45–50%
- After 2 epochs: ~90–95%

---

### 4. Exact Path Match (`exact_match`)

**What it is:** Fraction of samples where **every single node** in the thresholded prediction matches the target exactly.

**How it's computed:**
1. Threshold sigmoid(logits) > 0.5 for all 58 nodes.
2. Check if the predicted binary vector exactly equals the target multi-hot vector.
3. Count samples with perfect matches.

**What it tells you:** The strictest metric. A single wrong node (e.g., predicting "foliage" instead of "concrete" at level 3) makes the entire sample incorrect. This is useful for understanding if the model is learning consistent, complete paths or just getting the right leaf while hallucinating wrong internal nodes. In practice, exact match is low early in training because even one false positive on an unrelated branch ruins it.

**Typical values:**
- Untrained: ~0%
- After 2 epochs: ~5–10%

---

### 5. Path F1 (`path_f1`)

**What it is:** Macro-averaged F1 score for the multi-label path prediction.

**How it's computed (per sample):**
- **TP** (true positives): nodes correctly predicted as active
- **FP** (false positives): nodes predicted active but not in target
- **FN** (false negatives): nodes in target but predicted inactive
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 · Precision · Recall / (Precision + Recall)
- Averaged across all samples

**What it tells you:** A balanced measure of how complete and precise the predicted path is. Unlike exact match, a few false positives/negatives don't zero out the score. Good for tracking early training progress before exact match becomes meaningful. Path F1 is sensitive to both missing nodes (low recall) and over-predicting nodes (low precision).

**Typical values:**
- Untrained: ~15–20%
- After 2 epochs: ~65–75%

---

### 6. Hierarchy Accuracy (`hier_acc`)

**What it is:** Per-level argmax accuracy averaged over all hierarchy levels.

**How it's computed:**
1. For each hierarchy level (e.g., level 3: {metal, rock, ceramic, ...}), look at only those nodes' logits.
2. For each sample that participates in that level (has an active node there), predict the node with the highest logit.
3. Compare to the true active node at that level.
4. Average correctness across all levels and samples.

**What it tells you:** Whether the model correctly identifies the right branch at each level of the taxonomy. This decouples leaf accuracy from coarse-grained decisions. For example, if the model correctly picks "abiotic" at level 2 but then picks "rock" instead of "metal" at level 3, hierarchy accuracy at level 2 is 1.0 and level 3 is 0.0. Useful for diagnosing where in the taxonomy the model is confused.

**Typical values:**
- Untrained: ~45–55%
- After 2 epochs: ~85–90%

---

## Summary Table

| Metric | What it measures | Strictness | Useful for |
|--------|-----------------|------------|------------|
| **loss** | Training objective (BCE + hierarchical) | N/A | Primary optimization target |
| **leaf_acc** | Correct C1 leaf label | High | **Primary classification metric** |
| **node_acc** | Correct per-node predictions | Medium (many easy nodes) | Checking if model learns any structure |
| **exact_match** | Perfect full-path prediction | Very high | Understanding path consistency |
| **path_f1** | Balanced precision/recall of path | Medium | Early training progress tracking |
| **hier_acc** | Correct branch at each taxonomy level | Medium | Diagnosing where errors occur |

---

## Interpreting Metric Combinations

### Healthy training
```
loss ↓, leaf_acc ↑, node_acc ↑, path_f1 ↑, hier_acc ↑
```
All metrics improving together = model is learning.

### Leaf accuracy lags behind node accuracy
```
node_acc = 0.90, leaf_acc = 0.30
```
The model knows the coarse structure but can't distinguish leaves. Consider:
- Prototypes not synced with query projection
- Not enough training epochs
- GNN not propagating enough information to leaf nodes

### Exact match near zero but path_f1 decent
```
exact_match = 0.02, path_f1 = 0.60
```
Model gets most nodes right but makes 1–2 errors per sample. Usually means it confuses closely related sibling leaves (e.g., "granite" vs "limestone"). This is normal early in training.

### Hierarchy accuracy high but leaf accuracy low
```
hier_acc = 0.85, leaf_acc = 0.20
```
Model navigates the taxonomy well but fails at the final leaf decision. The leaf-level prototypes may need better initialization, or the GNN output isn't discriminative enough for the 36 leaf classes.

### All metrics flat after initial drop
```
loss plateau, leaf_acc stuck at 0.15
```
Model is not learning. Check:
- Learning rate too low/high
- Prototype artifact missing or mismatched
- Data not loading (run smoke test)
- CNN backbone frozen accidentally
