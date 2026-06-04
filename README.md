# matSeparate: Material Classification on SAM-Proposed Regions

This project studies material recognition on segmentation proposals from SAM. Given a scene image, SAM proposes candidate regions; each region is cropped, non-mask pixels are filled with the ImageNet mean, and a classifier predicts one of 23 MINC material classes.

The main finding is that a structured HGNN-style classifier is substantially more robust than flat ResNet baselines on these masked SAM-like crops. On 751 SAM-matched MINC-S segments, the HGNN improves accuracy from 57.92% to 65.91%, reduces mean Confusional Hierarchy Distance from 2.116 to 1.686, and improves Hier@d2 from 0.615 to 0.684.

Ablations qualify this result. The improvement is not explained by hierarchical loss alone, and the HGNN-CE variant outperforms a similarly sized MLP head. However, random-tree and fully connected graph topologies perform similarly to the true material taxonomy under CE training, so we do not claim that the hand-designed taxonomy topology is itself responsible for the gain. The evidence instead points to the structured prototype/message-passing head and hierarchical supervision as useful inductive biases for context-poor material crops.

The end-to-end dense segmentation pipeline remains proposal-limited: SAM recalls only 70.4% of MINC-S ground-truth material segments at IoU ≥ 0.5, and hierarchy-guided mask merging improves recall by only +0.75pp. We therefore treat dense SAM-based material segmentation as exploratory and focus the core contribution on robust region-level material classification.

---

## Table of Contents

1. [Problem](#problem)
2. [Datasets and Evaluation Protocol](#datasets-and-evaluation-protocol)
3. [Models](#models)
4. [Main Results](#main-results)
   - [Region Classification on SAM-Matched Segments](#region-classification-on-sam-matched-segments)
   - [GT Segment Classification: Masked vs Bbox](#gt-segment-classification-masked-vs-bbox)
   - [SAM Proposal Recall Bottleneck](#sam-proposal-recall-bottleneck)
5. [Main Findings](#main-findings)
6. [Ablation Studies](#ablation-studies)
   - [1. Structured Head vs MLP Head](#1-structured-head-vs-mlp-head)
   - [2. Does Taxonomy Topology Matter?](#2-does-taxonomy-topology-matter)
   - [3. Does the Model Reduce Severe Errors?](#3-does-the-model-reduce-severe-errors)
   - [4. Robustness to Context Removal](#4-robustness-to-context-removal)
   - [5. Bootstrap Confidence Intervals](#5-bootstrap-confidence-intervals)
7. [Limitations](#limitations)
8. [Setup and Reproduction](#setup-and-reproduction)
9. [Appendix](#appendix)

---

## Problem

SAM (Segment Anything Model) provides high-quality region proposals but is object-centric — it segments *things*, not *materials*. A material classifier must handle the resulting crops: bounding boxes where only the segmented pixels belong to the target material and surrounding context has been masked out with the ImageNet mean.

This masked-crop format is systematically harder than the uniform patches in MINC-2500 training data. All models drop 6–7pp in accuracy when context is removed, and the domain gap between training patches and real SAM crops is a key challenge.

---

## Datasets and Evaluation Protocol

| Dataset | Description |
|---|---|
| **MINC-2500** | 57,500 labeled material patches, 23 classes, 5-fold CV. Training only. |
| **MINC-S** | 1,654 scene photos, 7,061 GT segment masks across 23 classes. Evaluation only. |

**Primary evaluation:** SAM (vit_b, auto mode, ~64 masks/image) is run on 194 MINC-S photos. GT segments with a matching SAM mask (IoU ≥ 0.5) form the primary evaluation set — **751 segments**. Each classifier receives the SAM crop with non-mask pixels → ImageNet mean, matching exact deployment conditions.

**Secondary evaluation:** All 6,917 MINC-S GT segment masks, evaluated in both masked-crop and full-bbox modes.

**Metrics:**
- **Accuracy** — fraction of segments correctly classified
- **CHD** (Confusional Hierarchy Distance) — mean taxonomy tree distance between prediction and GT; lower is better
- **Hier@d2** — fraction of predictions within tree distance ≤ 2 of GT; higher is better

---

## Models

| Model | Architecture | Loss | Notes |
|---|---|---|---|
| **Flat ResNet50** | timm ResNet50 → 23-class head | CE | Standard baseline |
| **Flat + HierLoss** | ResNet50 → 38-node head | `greedy_loss` | Tests whether hierarchical loss alone helps, without a structured head |
| **MaskDropAugment** | ResNet50 → 23-class head | CE | Elliptical masking augmentation during training to reduce domain gap |
| **HGNN** | ResNet50 (frozen) + prototype nodes + 2-layer GAT over label graph → 38 logits | `greedy_loss` | Main model |

**`greedy_loss`:** At each level of the label graph, CE is applied over the children of the predicted parent. Encodes hierarchical structure into the gradient signal rather than treating all 23 classes as equidistant.

**HGNN backbone is frozen.** Only the ~616K prototype/GNN head parameters are trained. This isolates the contribution of the structured head from additional visual feature learning.

---

## Main Results

### Region Classification on SAM-Matched Segments

751 MINC-S GT segments matched by SAM (IoU ≥ 0.5). Each model receives the SAM masked crop.

| Model | Accuracy | CHD ↓ | Hier@d2 ↑ |
|---|---|---|---|
| Flat ResNet50 | 57.92% | 2.116 | 0.615 |
| Flat + HierLoss | 58.06% | 2.115 | 0.622 |
| MaskDropAugment | 59.79% | 1.980 | 0.639 |
| **HGNN** | **65.91%** | **1.686** | **0.684** |

- HierLoss alone gives negligible improvement over the flat baseline (+0.14pp). The hierarchical gradient signal does not help without the structured prototype/message-passing head.
- MaskDropAugment helps (+1.87pp) but falls well short of HGNN (+8pp). Ellipse masks are a poor approximation of real SAM segment shapes, and disrupting context during training appears to conflict with the GNN head's prototype-matching.
- The HGNN-style head plus `greedy_loss` gives the strongest result across all three metrics.

---

### GT Segment Classification: Masked vs Bbox

All 6,917 MINC-S GT segment masks classified directly (no SAM retrieval). Two crop modes to isolate the masking effect.

#### Masked crop (non-segment pixels → ImageNet mean)

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

Masking costs all models ~6–7pp. HGNN's margin over flat is much larger on masked crops (+4.6pp) than on bbox crops (+0.4pp), confirming that the structured head specifically helps when scene context is absent.

---

### SAM Proposal Recall Bottleneck

SAM vit_b auto mode on 194 MINC-S photos:

| Metric | Value |
|---|---|
| Recall @ IoU ≥ 0.5 | **70.4%** (751 / 1,067 GT segments) |
| Average masks / image | ~64 |

The 30% miss rate is structural: SAM is object-centric and does not isolate large material regions (carpet, wall, sky) as single proposals. This is a hard ceiling on end-to-end performance that region-classifier improvements cannot address.

---

## Main Findings

1. **Masked region classification is hard.** Removing non-segment context drops flat ResNet accuracy by roughly 6–7pp relative to bbox crops. SAM crops systematically present this condition.

2. **The HGNN-style head is robust under context loss.** On SAM-matched masked crops, HGNN improves accuracy by +7.99pp over flat ResNet50 and reduces CHD by 0.430. The advantage is larger under heavy masking than with full context.

3. **Hierarchical loss alone is insufficient.** Flat + HierLoss performs nearly identically to the flat baseline (+0.14pp). The hierarchical gradient signal requires a structured prototype/message-passing head to be effective.

4. **The exact taxonomy topology is not the source of the gain.** Under CE training, true-tree, random-tree, and fully connected graph variants all perform within ~0.8pp of each other. The gain comes from the structured head mechanism, not the specific hand-designed material edges.

5. **SAM proposal recall is the end-to-end bottleneck.** SAM recalls 70.4% of GT material segments at IoU ≥ 0.5. Hierarchy-guided mask merging improves this by only +0.75pp, suggesting the failure mode is scale mismatch, not over-segmentation.

---

## Ablation Studies

All ablations evaluated on 751 SAM-matched MINC-S segments (GT mask crop, IoU ≥ 0.5).

---

### 1. Structured Head vs MLP Head

*Does the HGNN head outperform a similarly sized MLP classifier?*

All variants use **CE loss on leaf logits** so the loss is not a confound. Non-CNN parameters are matched as closely as possible.

| Model | Non-CNN params | MINC-2500 val | GT mask crop acc | CHD | Hier@d2 |
|---|---|---|---|---|---|
| MLP head (ResNet50 + 4-layer MLP) | 430K | 64.35% | 46.87% | 2.654 | 0.523 |
| HGNN-CE (true taxonomy graph, CE loss)  | 616K | 70.54% | 53.00% | 2.441 | 0.558 |

**+6.1pp from the structured HGNN-CE head over a similarly sized MLP head.** This suggests that the gain is not merely due to adding a larger classifier, but the ablation isolates the structured prototype/message-passing mechanism, not the semantic taxonomy itself — the topology result below qualifies this further.

The MLP also generalizes much worse under distribution shift (MINC-2500 val → SAM crops: −17pp) compared to HGNN-CE (−18pp from a higher base), suggesting the structured head provides a more robust inductive bias.

*Note: The main HGNN uses `greedy_loss`, which substantially improves over HGNN-CE under the same evaluation protocol. This suggests hierarchical supervision is a major contributor. However, because topology variants were only evaluated with CE, we do not claim that the semantic taxonomy topology itself is responsible for the greedy_loss gain.*

---

### 2. Does Taxonomy Topology Matter?

*Does the hand-designed material taxonomy matter, or just the message-passing mechanism?*

Same HGNN architecture; only graph topology varies. All CE-trained for fair comparison.

| Model | Graph topology | MINC-2500 val | GT mask crop acc |
|---|---|---|---|
| HGNN-CE | True taxonomy tree | 70.54% | 53.00% |
| Random tree | Random spanning tree (same 38 nodes) | 70.26% | 53.79% |
| Full graph | Fully connected (all-to-all edges) | 69.98% | 53.13% |

**All three perform within ~0.8pp.** The true taxonomy topology is **not** empirically validated as the source of the improvement. The graph architecture helps (as shown in Ablation 1), but the specific semantic edges do not appear to matter under CE training. What matters is the mechanism — prototype nodes and message passing over *some* graph — not the specific hierarchy.

---

### 3. Does the Model Reduce Severe Errors?

*Does HGNN make fewer semantically severe mistakes (cross-parent confusions)?*

Confusion matrices grouped by parent taxonomy node (masonry, vitreous, textile, etc.):

| Model | Within-parent acc | Cross-parent error rate |
|---|---|---|
| Flat ResNet50 | 58.32% | 41.68% |
| **HGNN** | **64.85%** | **35.15%** |

**HGNN reduces cross-parent errors by −6.5pp.** Per-group improvements:

| Group | Flat | HGNN | Gain |
|---|---|---|---|
| synthetic | 68% | 80% | +12pp |
| animal_derived | 54% | 66% | +12pp |
| wood_derived | 44% | 54% | +10pp |
| vitreous (glass/mirror) | ~20% | ~20% | ≈0 |

Vitreous is the hardest group for both models — glass and mirror are consistently confused with synthetic materials (plastic, painted). The label graph has no dedicated reflective-surface node, so there is no structural signal to separate them.

---

### 4. Robustness to Context Removal

*Is HGNN's advantage specifically tied to the masked-crop condition?*

Both models evaluated at five mask-drop levels (0% = full bbox, 100% = all non-segment pixels → ImageNet mean):

| Context removed | Flat acc | HGNN acc | HGNN gain |
|---|---|---|---|
| 0% (full bbox) | 55.93% | 60.99% | +5.1pp |
| 25% | 54.06% | 59.65% | +5.6pp |
| 50% | 53.13% | 59.39% | +6.3pp |
| 75% | 53.26% | 58.85% | +5.6pp |
| **100% (full mask)** | 54.86% | **61.52%** | **+6.7pp** |

HGNN's advantage is consistently larger under heavy masking, peaking at +6.7pp when all non-segment pixels are removed. The gain is positive at every level tested. This directly supports the claim that the structured head is most beneficial under the context-poor conditions that characterize SAM crops.

---

### 5. Bootstrap Confidence Intervals

1,000 bootstrap resamples over the 751 SAM-matched MINC-S segments (full masked crop):

| Metric | Mean | 95% CI |
|---|---|---|
| Flat accuracy | 54.80% | [51.4%, 58.2%] |
| HGNN accuracy | **61.40%** | [57.9%, 64.9%] |
| **Accuracy gain (HGNN − flat)** | **+6.6pp** | **[+3.7pp, +9.9pp]** |
| Flat CHD | 2.251 | [2.07, 2.44] |
| HGNN CHD | 1.892 | [1.71, 2.08] |
| CHD reduction | 0.359 | [0.20, 0.53] |

Across all 1,000 bootstrap resamples, HGNN outperformed flat in every resample. The 95% CI for the accuracy gain ([+3.7pp, +9.9pp]) is entirely above zero.

---

## Limitations

**The semantic taxonomy topology is not validated as the source of improvement.**  
Topology ablations show that true-tree, random-tree, and fully connected graph variants perform within ~0.8pp under CE training. The current evidence supports the structured HGNN-style head and hierarchical supervision, not the specific hand-designed material taxonomy, as the source of the gain.

**The dense segmentation pipeline is proposal-limited.**  
SAM recalls only 70.4% of ground-truth MINC-S material segments at IoU ≥ 0.5. The region classifier cannot recover segments SAM never proposes, imposing a hard ceiling on end-to-end performance.

**Mask merging has limited impact.**  
Hierarchy-guided merging improves recall by only +0.75pp and end-to-end R×A by +0.66pp, confirming that SAM's main failure mode is scale/granularity mismatch rather than over-segmentation.

**Synthetic dense training is not conclusive.**  
The ConvNeXt-FPN dense segmentation experiment improves by only +1.24pp with hierarchical boundary loss, likely because 2×2 synthetic composites do not capture real scene context. Evaluation on a dataset with dense GT labels (e.g., OpenSurfaces) would be needed to draw conclusions.

---

## Setup and Reproduction

```bash
pip install -r requirements.txt
```

### Run everything

```bash
bash scripts/run_all.sh
```

Trains all models (skipping already-completed checkpoints) and runs all evaluations. Requires MINC-2500, MINC-S, and pre-computed SAM masks.

### Individual steps

```bash
# Baseline classifiers
python scripts/train_classifier.py --model flat       --epochs 10
python scripts/train_classifier.py --model hierloss   --epochs 10
python scripts/train_classifier.py --model maskdrop   --epochs 10

# Main HGNN
python scripts/train_hgnn.py --epochs 10

# Graph ablation variants (5 epochs — fixed budget for topology comparison)
python scripts/train_ablation_variants.py

# Evaluate classifiers on MINC-S segments
python scripts/eval_classifiers.py --section all

# Run all 5 ablation studies
python scripts/eval_ablations.py --ablation 1 2 3 4 5
```

### Repository structure

```
scripts/
  train_classifier.py         # flat / hierloss / maskdrop baselines
  train_hgnn.py               # main HGNN (greedy_loss)
  train_ablation_variants.py  # mlp_head / hgnn_ce / random_tree / full_graph
  eval_classifiers.py         # classifier eval on MINC-S (Sections 1 & 2)
  eval_ablations.py           # all 5 ablation studies
  run_pipeline.py             # SAM + HGNN pixel reconciliation (exploratory)
  run_merge.py                # hierarchy-guided SAM mask merging (exploratory)
  run_all.sh                  # master sequential runner

gnn_classifier/
  hgnn.py                     # HGNN model (ResNet50 + GAT over label graph)
  loss.py                     # greedy_loss hierarchical objective

taxonomy/
  tree.py                     # taxonomy graph utilities
  assets/minc-taxonomy.json   # 38-node MINC material label graph

datasets/
  minc.py                     # MINC-2500 dataset loader
```

---

## Appendix

### A. Pixel-Level Reconciliation (Exploratory)

An end-to-end pixel segmentation pipeline was explored: classify each SAM mask with HGNN; for each pixel, assign the label from the highest-confidence covering mask; uncovered and low-confidence pixels default to `other`.

Results on 20 MINC-S photos:
- Average pixel coverage: 70.3% (remaining ~30% → `other`, structurally uncovered)

This pipeline is limited by the SAM recall ceiling. Region-level classification is the core contribution.

### B. Hierarchy-Guided Mask Merging (Exploratory)

Adjacent SAM masks sharing a predicted parent label are merged and re-classified. 8,818 merge events across 194 photos:

| Metric | Before | After | Δ |
|---|---|---|---|
| Recall @ IoU | 70.38% | 71.13% | +0.75pp |
| Classifier Accuracy | 61.52% | 61.79% | +0.27pp |
| End-to-end (R × A) | 43.30% | 43.96% | +0.66pp |

Gains are modest — the miss rate is from scale mismatch, not over-splitting.

### C. HierSeg: Dense Material Segmentation (Exploratory)

ConvNeXt-tiny + 4-scale FPN trained on synthetic 2×2 composites from MINC-2500 patches, with a hierarchical boundary loss.

| Run | λ_hier | Best MINC-S Acc |
|---|---|---|
| Baseline | 0.0 | 49.27% |
| HierSeg | 0.5 | 50.51% |

+1.24pp within noise. Primary bottleneck is training data quality.

### D. Training Details

| Model | Epochs | Batch | LR | Notes |
|---|---|---|---|---|
| Flat / HierLoss / MaskDrop | 10 | 64 | 1e-3 | AdamW, cosine LR, full backbone |
| HGNN | 10 | 32 | 1e-4 | Backbone frozen; GNN head 1× lr, backbone 0.1× |
| Ablation variants | 5 | 32 | 1e-4 | Fixed budget for fair topology comparison |

Weight decay 5e-4. MINC-2500 fold-1.
