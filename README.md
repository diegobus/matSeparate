# matSeparate: Material Segmentation with SAM + HGNN

Material segmentation pipeline for MINC-S scene photos. SAM generates region proposals; an HGNN classifier labels each region with one of 23 MINC material categories. A reconciliation step fuses proposals into a pixel-level map.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Results](#results)
   - [SAM Structural Ceiling](#sam-structural-ceiling)
   - [Patch Classifier Comparison](#patch-classifier-comparison)
   - [GT Segment Evaluation](#gt-segment-evaluation)
   - [End-to-End Pipeline](#end-to-end-pipeline)
   - [Hierarchy-Guided Mask Merging](#hierarchy-guided-mask-merging)
5. [Ablation Studies](#ablation-studies)
   - [Ablation 1: Graph vs Parameters](#ablation-1-graph-vs-parameters)
   - [Ablation 2: Graph Topology](#ablation-2-graph-topology)
   - [Ablation 3: Cross-Parent Error Reduction](#ablation-3-cross-parent-error-reduction)
   - [Ablation 4: Context Removal](#ablation-4-context-removal)
   - [Ablation 5: Statistical Significance](#ablation-5-statistical-significance)
6. [Setup & Usage](#setup--usage)
7. [Appendix](#appendix)

---

## Pipeline Overview

```
Scene photo
    │
    ▼
SAM (vit_b, auto mode)
~64 masks/image
    │
    ▼  For each mask:
    │   1. Crop to bounding box
    │   2. Fill non-mask pixels → ImageNet mean
    │
    ▼
HGNN Classifier
ResNet50 backbone + 2-layer GAT
over material taxonomy graph
    │
    ▼
Reconciliation
Per-pixel: label from highest-confidence
covering mask. Uncovered / low-confidence → "other"
    │
    ▼
Pixel-level material map
```

**Key design choices:**
- Non-mask pixels are filled with ImageNet mean at inference, matching how SAM crops differ from clean training patches.
- Low-confidence pixels (max-softmax < 0.3) and SAM-uncovered pixels default to `other` (class 11). SAM is object/region-driven and unlikely to propose genuinely ambiguous material regions — defaulting them to `other` is correct rather than propagating wrong labels via nearest-neighbor fill.

---

## Datasets

| Dataset | Description |
|---|---|
| **MINC-2500** | 57,500 labeled material patches, 23 classes, 5-fold CV. Used for training. |
| **MINC-S** | 1,654 scene photos with 7,061 GT segment masks across 23 classes. Used for evaluation only. |

Data paths (after download):
```
data/external/minc/minc-2500/   ← MINC-2500 patches
data/external/minc/minc-s/      ← MINC-S photos + segment annotations
```

---

## Models

| Model | Architecture | Loss | Key property |
|---|---|---|---|
| **Flat ResNet50** | timm ResNet50 → 23-class head | CE | Baseline |
| **Flat + HierLoss** | ResNet50 → 38-node head | `greedy_loss` | Tests whether the hierarchical loss alone helps |
| **MaskDropAugment** | ResNet50 → 23-class head | CE | Trained with elliptical masking augmentation to close domain gap |
| **HGNN** | ResNet50 + 2-layer GAT over taxonomy graph → 38 logits | `greedy_loss` | Main model |

**`greedy_loss`:** At each level of the taxonomy, CE is applied over the children of the predicted parent node. This encodes the hierarchy directly into the gradient signal.

**HGNN backbone is frozen** during training. Only the GNN head (prototype embeddings + GAT + pooling) is trained. This isolates the contribution of the graph structure rather than additional visual feature tuning.

---

## Results

### SAM Structural Ceiling

SAM vit_b auto mode run on 194 MINC-S photos:

| Metric | Value |
|---|---|
| Recall @ IoU ≥ 0.5 | **70.4%** (751 / 1,067 GT segments matched) |
| Average masks / image | ~64 |
| Miss rate (structural) | ~30% |

The 30% miss rate is structural: SAM is object-centric. Large material regions (carpet, wall, sky) that span the full scene are rarely isolated as single proposals. Merging cannot close this gap — the scale mismatch is the root cause.

---

### Patch Classifier Comparison

Evaluated on **751 MINC-S GT segments where SAM produced a matching mask (IoU ≥ 0.5)**.  
Each model receives the SAM crop with non-mask pixels filled to ImageNet mean.

| Model | Accuracy | CHD ↓ | Hier@d2 ↑ |
|---|---|---|---|
| Flat ResNet50 | 57.92% | 2.116 | 0.615 |
| Flat + HierLoss | 58.06% | 2.115 | 0.622 |
| MaskDropAugment | 59.79% | 1.980 | 0.639 |
| **HGNN** | **65.91%** | **1.686** | **0.684** |

**Metrics:**
- **CHD** (Confusional Hierarchy Distance): mean tree distance between predicted and GT class. Lower is better.
- **Hier@d2**: fraction of predictions within tree distance ≤ 2 of GT. Higher is better.

**Key findings:**
- HierLoss alone adds negligible value (+0.14pp) over flat CE. The hierarchical gradient signal requires the graph structure to be effective.
- MaskDropAugment helps (+1.87pp) but doesn't match HGNN (+8pp). Elliptical masks are a weak approximation of real segment shapes, and disrupting context during training conflicts with the GNN's prototype-matching mechanism.
- HGNN achieves the largest CHD reduction (−0.43), indicating fewer semantically severe mistakes.

---

### GT Segment Evaluation

Direct classification on **all 6,917 MINC-S GT segment masks** (no SAM retrieval step).  
Two crop modes: **masked** (non-segment pixels → ImageNet mean) vs **bbox** (full bounding box, no masking).

#### Masked crop

| Model | Accuracy | CHD ↓ | Hier@d2 ↑ |
|---|---|---|---|
| Flat ResNet50 | 56.82% | 2.108 | 0.608 |
| Flat + HierLoss | 55.54% | 2.157 | 0.601 |
| **HGNN** | **61.40%** | **1.855** | **0.655** |

#### Bbox crop (no masking)

| Model | Accuracy | CHD ↓ | Hier@d2 ↑ |
|---|---|---|---|
| Flat ResNet50 | 63.63% | 1.706 | 0.680 |
| Flat + HierLoss | 63.65% | 1.717 | 0.675 |
| **HGNN** | **64.03%** | **1.682** | **0.681** |

**Key finding:** Masking hurts all models (masked vs bbox: ~6–7pp accuracy gap). HGNN's advantage over flat is **larger on masked crops (+4.6pp)** than bbox (+0.4pp), suggesting the GNN's taxonomy-aware prototype matching compensates for removed context.

---

### End-to-End Pipeline

Full SAM + HGNN reconciliation on 20 MINC-S photos:

| Metric | Value |
|---|---|
| Avg pixel coverage (≥ 1 SAM mask) | 70.3% |
| Uncovered pixels → `other` | ~30% |
| High-confidence predictions | foliage, glass, leather, skin, hair |

---

### Hierarchy-Guided Mask Merging

Adjacent SAM masks sharing a predicted taxonomy parent are merged and re-classified, on the hypothesis that SAM over-splits material regions.

Evaluated on 194 MINC-S photos (8,818 merge events):

| Metric | Before merge | After merge | Δ |
|---|---|---|---|
| Recall @ IoU | 70.38% | 71.13% | +0.75pp |
| Classifier Accuracy | 61.52% | 61.79% | +0.27pp |
| End-to-end (R × A) | 43.30% | 43.96% | +0.66pp |

**Finding:** Gains are modest. The 30% SAM miss rate comes from scale mismatch (SAM never generates the proposal at all), not region splitting, so merging cannot address the main failure mode.

---

## Ablation Studies

All ablations evaluated on **751 MINC-S GT segments matched by SAM (IoU ≥ 0.5), using the GT segment mask for cropping** (consistent across models).

---

### Ablation 1: Graph vs Parameters

*Is the HGNN gain from graph structure, or just extra parameters?*

All variants use **CE loss on leaf logits** (loss is not a confound). Non-CNN parameters are matched where possible.

| Model | Non-CNN params | MINC-2500 val | GT mask crop acc | CHD | Hier@d2 |
|---|---|---|---|---|---|
| MLP head (ResNet50 + 4-layer MLP) | 430K | 64.35% | 46.87% | 2.654 | 0.523 |
| HGNN-CE (true taxonomy, CE loss)  | 616K | 70.54% | 53.00% | 2.441 | 0.558 |

**Conclusion:** +6.1pp from graph structure alone, not parameters. The MLP also generalises much worse under distribution shift (MINC-2500 val → real SAM crops: −17pp), suggesting the graph's prototype-matching provides a more robust inductive bias.

*Note: The main HGNN uses `greedy_loss` instead of CE. The additional +8.5pp over HGNN-CE confirms that the hierarchical loss is the largest single contributor to performance.*

---

### Ablation 2: Graph Topology

*Does the specific material hierarchy matter, or just the GNN mechanism?*

Same HGNN architecture across all three; only the graph topology varies. All trained with CE loss.

| Model | Graph topology | MINC-2500 val | GT mask crop acc |
|---|---|---|---|
| HGNN-CE | True taxonomy tree | 70.54% | 53.00% |
| Random tree | Random spanning tree (same 38 nodes) | 70.26% | 53.79% |
| Full graph | Fully connected (all-to-all edges) | 69.98% | 53.13% |

**Conclusion:** All three topologies perform within **~0.8pp** of each other. The specific semantic structure of the material taxonomy provides essentially no advantage. What matters is the *mechanism* — prototype nodes + message passing — not the hierarchy itself.

---

### Ablation 3: Cross-Parent Error Reduction

*Does HGNN reduce semantically severe mistakes?*

Confusion matrices grouped by parent taxonomy node (masonry, vitreous, textile, etc.).

| Model | Within-parent acc | Cross-parent error rate |
|---|---|---|
| Flat ResNet50 | 58.32% | 41.68% |
| **HGNN** | **64.85%** | **35.15%** |

**HGNN reduces cross-parent errors by −6.5pp.** Largest improvements per category group:

| Group | Flat acc | HGNN acc | Gain |
|---|---|---|---|
| synthetic | 68% | 80% | +12pp |
| animal_derived | 54% | 66% | +12pp |
| wood_derived | 44% | 54% | +10pp |
| vitreous | ~20% | ~20% | ≈0 |

Worst group for both models: **vitreous** (glass/mirror, ~20% within-parent accuracy). Both models confuse glass/mirror with synthetic materials (plastic/painted) — a natural failure without a dedicated reflective-surface grouping in the taxonomy.

---

### Ablation 4: Context Removal

*Does HGNN's advantage grow as scene context is removed?*

Models evaluated at five mask-drop levels: 0% = full bounding box, 100% = all non-segment pixels set to ImageNet mean.

| Context removed | Flat acc | HGNN acc | HGNN gain |
|---|---|---|---|
| 0% (full bbox) | 55.93% | 60.99% | +5.1pp |
| 25% | 54.06% | 59.65% | +5.6pp |
| 50% | 53.13% | 59.39% | +6.3pp |
| 75% | 53.26% | 58.85% | +5.6pp |
| **100% (full mask)** | 54.86% | **61.52%** | **+6.7pp** |

**Conclusion:** HGNN's gain grows as context is removed (+5.1pp → +6.7pp). The GNN's prototype-matching is most valuable precisely when external scene context is absent — which is the real-world SAM crop scenario.

---

### Ablation 5: Statistical Significance

Bootstrap confidence intervals over 1,000 resamples of 751 MINC-S GT segments (full masked crop).

| Metric | Mean | 95% CI |
|---|---|---|
| Flat accuracy | 54.80% | [51.4%, 58.2%] |
| HGNN accuracy | **61.40%** | [57.9%, 64.9%] |
| **Accuracy gain (HGNN − flat)** | **+6.6pp** | **[+3.7pp, +9.9pp]** |
| Flat CHD | 2.251 | [2.07, 2.44] |
| HGNN CHD | 1.892 | [1.71, 2.08] |
| CHD reduction | 0.359 | [0.20, 0.53] |

**P(HGNN > Flat) = 1.000.** The 95% CI for the accuracy gain is entirely above zero — the result is statistically robust.

---

## Setup & Usage

### Install

```bash
pip install -r requirements.txt
```

### Run everything

```bash
bash scripts/run_all.sh
```

This trains all models and runs all evaluations sequentially. Already-trained checkpoints are skipped automatically.

### Individual steps

```bash
# Train baseline classifiers
python scripts/train_classifier.py --model flat       --epochs 10
python scripts/train_classifier.py --model hierloss   --epochs 10
python scripts/train_classifier.py --model maskdrop   --epochs 10

# Train main HGNN
python scripts/train_hgnn.py --epochs 10

# Train graph ablation variants (5 epochs each for fair comparison)
python scripts/train_ablation_variants.py  # or --variant mlp_head|hgnn_ce|random_tree|full_graph

# Evaluate classifiers on MINC-S
python scripts/eval_classifiers.py --section all

# Run ablation studies
python scripts/eval_ablations.py --ablation 1 2 3 4 5

# Run SAM + HGNN pipeline on scene photos
python scripts/run_pipeline.py

# Optional: hierarchy-guided mask merging before pipeline
python scripts/run_merge.py
python scripts/run_pipeline.py --sam-dir out/sam_merged
```

### Repository structure

```
scripts/
  train_classifier.py       # flat / hierloss / maskdrop baselines
  train_hgnn.py             # main HGNN (greedy_loss)
  train_ablation_variants.py # mlp_head / hgnn_ce / random_tree / full_graph
  eval_classifiers.py       # classifier eval on MINC-S (Sections 1 & 2)
  eval_ablations.py         # all 5 ablation studies
  run_pipeline.py           # SAM + HGNN → pixel-level material maps
  run_merge.py              # hierarchy-guided SAM mask merging
  run_all.sh                # master sequential runner

gnn_classifier/
  hgnn.py                   # HGNN model (ResNet50 + GAT)
  loss.py                   # greedy_loss hierarchical objective

taxonomy/
  tree.py                   # taxonomy graph utilities
  assets/minc-taxonomy.json # 38-node MINC material hierarchy

datasets/
  minc.py                   # MINC-2500 dataset
```

---

## Appendix

### A. HierSeg: Dense Material Segmentation

A separate dense segmentation approach was explored using a ConvNeXt-tiny encoder + FPN neck trained on synthetic 2×2 composite images assembled from MINC-2500 patches, with a hierarchical boundary loss.

**Architecture:** ConvNeXt-tiny (ImageNet-22K pretrained) + 4-scale FPN (128ch) + 23-class dense head. ~28.7M parameters.

**Hierarchical boundary loss:** For adjacent pixel pairs with different GT labels: penalise low total variation (push for sharp boundaries) weighted by taxonomy tree distance. For same-label pairs: penalise high TV (encourage smoothness).

| Run | λ_hier | Best MINC-S Acc | Epochs |
|---|---|---|---|
| Baseline | 0.0 | 49.27% | 30 |
| HierSeg | 0.5 | **50.51%** | 30 |

+1.24pp from the boundary loss, within noise. The primary bottleneck is the training data: synthetic 2×2 composites have artificial boundaries and no real scene context. Dense GT labels from a scene dataset (e.g., OpenSurfaces) would be needed to properly validate this approach.

---

### B. Training Details

| Model | Epochs | Batch | LR | Backbone | Notes |
|---|---|---|---|---|---|
| Flat ResNet50 | 10 | 64 | 1e-3 | ResNet50 (ImageNet) | timm, full fine-tune |
| Flat + HierLoss | 10 | 64 | 1e-3 | ResNet50 (ImageNet) | 38-node head, greedy_loss |
| MaskDropAugment | 10 | 64 | 1e-3 | ResNet50 (ImageNet) | p=0.5 ellipse masking |
| HGNN | 10 | 32 | 1e-4 | ResNet50 (frozen) | Differential LR: backbone 0.1× |
| Ablation variants | 5 | 32 | 1e-4 | ResNet50 (frozen) | Fixed budget for topology comparison |

Optimizer: AdamW, weight decay 5e-4. Scheduler: CosineAnnealingLR. Data: MINC-2500 fold-1.
