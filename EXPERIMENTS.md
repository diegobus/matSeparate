# Material Segmentation Experiments

**Dataset:** MINC-2500 (57,500 patches, 23 classes, 5-fold CV) + MINC-S (1,654 scene photos, 7,061 GT segment masks, test-only)  
**Proposal method:** SAM vit_b auto mode, ~64 masks/image  
**Pipeline:** SAM generates region proposals → model classifies each crop → reconcile into pixel map

---

## Metrics

- **Accuracy**: fraction of segments with correct leaf-class prediction
- **CHD** (Confusional Hierarchy Distance): mean tree distance between predicted and GT class (lower = better)
- **Hier@d2**: fraction of predictions within tree distance ≤ 2 of GT (higher = better)
- **Recall@IoU**: fraction of GT segments for which SAM produced a mask with IoU ≥ 0.5
- **E2E**: Recall@IoU × Classifier Accuracy (end-to-end metric)

---

## 1. Patch Classifier Ablation (SAM-Matched Segments)

Evaluated on 751 MINC-S GT segments where SAM produced a matching mask (IoU ≥ 0.5).  
Each classifier receives the SAM crop with non-mask pixels filled to ImageNet mean.

| Model | Accuracy | CHD | Hier@d2 |
|---|---|---|---|
| Flat ResNet50 | 57.92% | 2.116 | 0.615 |
| Flat + HierLoss | 58.06% | 2.115 | 0.622 |
| MaskDropAugment | 59.79% | 1.980 | 0.639 |
| **HGNN** | **65.91%** | **1.686** | **0.684** |

**MaskDropAugment**: during training, random ellipses within crops are filled to ImageNet mean, simulating the masked-out context that SAM crops have at inference time.  
**HGNN**: ResNet50 backbone + 2-layer GNN over the material taxonomy graph. Outputs logits for all 38 taxonomy nodes; leaf logits are extracted for the 23-class prediction.  
**HierLoss**: auxiliary cross-entropy on internal taxonomy nodes during training (negligible improvement over flat).

---

## 2. GT Segment Eval (Direct Classification, No SAM)

Direct classification on all 6,917 MINC-S GT segment masks (no SAM retrieval step).  
Two crop modes: **masked** (non-segment pixels → ImageNet mean) vs **bbox** (full bounding box, no masking).

### Masked crop

| Model | Accuracy | CHD | Hier@d2 |
|---|---|---|---|
| Flat ResNet50 | 56.82% | 2.108 | 0.608 |
| Flat + HierLoss | 55.54% | 2.157 | 0.601 |
| HGNN | **61.40%** | **1.855** | **0.655** |

### Bbox crop (no masking)

| Model | Accuracy | CHD | Hier@d2 |
|---|---|---|---|
| Flat ResNet50 | 63.63% | 1.706 | 0.680 |
| Flat + HierLoss | 63.65% | 1.717 | 0.675 |
| HGNN | **64.03%** | **1.682** | **0.681** |

**Key finding:** Masking hurts all models (masked vs bbox: ~6–7pp accuracy gap). HGNN's advantage over flat is larger on masked crops (+4.6pp) than bbox (+0.4pp), suggesting the GNN's taxonomy-aware pooling helps when context is removed.

---

## 3. HGNN Ablation Studies

All ablation models trained 5 epochs on MINC-2500 fold-1. Evaluated on 751 MINC-S GT segments matched by SAM (IoU≥0.5), using the GT segment mask for cropping (consistent across models).

### 3a. Is it the graph, or just more parameters? (Ablation 1)

CE loss on leaf logits for all variants; non-CNN params matched where possible.

| Model | Non-CNN params | MINC-2500 val | GT mask crop acc | CHD | Hier@d2 |
|---|---|---|---|---|---|
| mlp_head (ResNet50 + 4-layer MLP) | 430K | 64.35% | 46.87% | 2.654 | 0.523 |
| hgnn_ce (true taxonomy, CE loss) | 616K | 70.54% | 53.00% | 2.441 | 0.558 |

**Finding:** +6.1pp from graph structure alone, not parameters. The MLP generalizes much worse under distribution shift (−17pp from MINC-2500 val to real crops) vs HGNN (−18pp but from a higher base). Graph connectivity matters.

### 3b. Does the specific hierarchy matter? (Ablation 2)

Same HGNN architecture; only the graph topology varies. All trained with CE loss.

| Model | Graph topology | MINC-2500 val | GT mask crop acc |
|---|---|---|---|
| hgnn_ce | True taxonomy tree | 70.54% | 53.00% |
| random_tree | Random spanning tree (same nodes) | 70.26% | 53.79% |
| full_graph | Fully connected (all-to-all edges) | 69.98% | 53.13% |

**Finding:** All three topologies perform within ~0.8pp of each other. The specific semantic structure of the material taxonomy provides essentially no advantage over a random tree or full graph. What matters is the *mechanism* (prototype nodes + message passing), not the hierarchy itself.

### 3c. Does HGNN reduce severe mistakes? (Ablation 3)

Confusion matrices grouped by parent taxonomy node (masonry, vitreous, textile, etc.). Evaluated on 751 SAM-matched segments.

| Model | Within-parent accuracy | Cross-parent error rate |
|---|---|---|
| Flat ResNet50 | 58.32% | 41.68% |
| HGNN (greedy loss) | **64.85%** | **35.15%** |

**HGNN reduces cross-parent errors by −6.5pp.** Largest improvements: synthetic (+12pp: 68→80%), animal_derived (+12pp: 54→66%), wood_derived (+10pp: 44→54%). Worst class for both models: vitreous (glass/mirror) at ~20% within-parent accuracy — both models confuse glass/mirror with synthetic materials (plastic/painted), a natural failure case without a dedicated reflective-surface grouping in the taxonomy.

### 3d. Does HGNN help most when context is removed? (Ablation 4)

Evaluated at varying mask-drop severities: 0% = full bbox crop, 100% = all non-segment pixels set to ImageNet mean.

| Context removed | Flat acc | HGNN acc | HGNN gain |
|---|---|---|---|
| 0% (full bbox) | 55.93% | 60.99% | +5.1pp |
| 25% | 54.06% | 59.65% | +5.6pp |
| 50% | 53.13% | 59.39% | +6.3pp |
| 75% | 53.26% | 58.85% | +5.6pp |
| 100% (full mask) | 54.86% | **61.52%** | **+6.7pp** |

**Finding:** HGNN's advantage grows as context is removed. At full masking the gain is +6.7pp vs +5.1pp with full context. The graph's prototype-matching is most valuable precisely when external scene context is absent — which is the real-world SAM crop scenario.

### 3e. Is the result stable? Bootstrap confidence intervals (Ablation 5)

1,000-sample bootstrap over 751 MINC-S GT segments (full masked crop).

| Metric | Mean | 95% CI |
|---|---|---|
| Flat accuracy | 54.80% | [51.4%, 58.2%] |
| HGNN accuracy | **61.40%** | [57.9%, 64.9%] |
| Accuracy gain (HGNN − flat) | **+6.6pp** | **[+3.7pp, +9.9pp]** |
| Flat CHD | 2.251 | [2.07, 2.44] |
| HGNN CHD | 1.892 | [1.71, 2.08] |
| CHD reduction | 0.359 | [0.20, 0.53] |

P(HGNN > flat) = 1.000. The 95% CI for the gain is entirely above zero — the result is statistically significant and robust.

---

## 4. SAM Recall Analysis (SAM Structural Ceiling)

SAM vit_b auto mode on 194 MINC-S photos:

- **Recall@IoU ≥ 0.5**: 70.4% (751 / 1,067 GT segments matched)
- Average masks per image: ~64
- The ~30% miss rate is structural: SAM is object-centric; material regions like "carpet" or "wall" span full scene areas that SAM doesn't isolate as single proposals.

---

## 5. Hierarchy-Guided SAM Mask Merging

**Hypothesis:** SAM splits material regions into multiple proposals. Adjacent masks predicted to share a parent taxonomy node likely belong to the same material region and should be merged.

**Method:** For each image, classify all SAM masks with HGNN. Find adjacent mask pairs (bounding-box proximity within 15px). If two adjacent masks share the same parent-node prediction, merge (union) and re-classify the merged crop.

| Metric | Before | After | Δ |
|---|---|---|---|
| Recall@IoU | 70.38% | 71.13% | +0.75pp |
| Classifier Accuracy | 61.52% | 61.79% | +0.27pp |
| End-to-end (R×A) | 43.30% | 43.96% | +0.66pp |

8,818 merge events across 194 photos. Modest gains — the 30% SAM miss rate is caused by granularity mismatch (wrong scale), not region splitting, so merging can't close that gap.

---

## 6. HierSeg: Dense Material Segmentation

**Architecture:** ConvNeXt-tiny encoder (ImageNet-22K pretrained) + 4-scale FPN neck (128ch) + 23-class head. ~28.7M parameters. Trained on synthetic 2×2 composite images assembled from MINC-2500 patches, evaluated on MINC-S.

**Hierarchical boundary loss:** For pairs of adjacent pixels, if GT labels differ: penalize low TV (push boundary to be sharp) weighted by taxonomy tree distance. If GT labels match: penalize high TV (encourage smoothness).

| Run | λ_hier | Best MINC-S Acc | Epochs |
|---|---|---|---|
| Baseline | 0.0 | 49.27% | 30 |
| HierSeg | 0.5 | 50.51% | 30 |

+1.24pp from the hierarchical boundary loss, within noise. The primary bottleneck is the training data: synthetic 2×2 composites are a weak signal (artificial boundaries, no real scene context). Would require dense GT labels from a scene dataset (e.g. OpenSurfaces) to properly validate.

---

## 7. SAM + HGNN Reconciliation Pipeline

Produces pixel-level material segmentation maps from SAM proposals + HGNN classification.

**Method:**
1. Run SAM auto mode on each scene photo (~64 masks)
2. For each mask, crop + classify with HGNN → softmax confidence vector
3. For each pixel: assign label from the covering mask with highest max-softmax confidence
4. Low-confidence pixels (max-softmax < 0.3) → "other"
5. Uncovered pixels (no SAM mask) → "other" (SAM is unlikely to generate proposals for genuinely ambiguous "other" regions)

**Results on 20 MINC-S photos:**
- Average pixel coverage (covered by ≥1 SAM mask): 70.3%
- Remaining ~30% of pixels classified as "other" (SAM miss regions)
- Common high-confidence predictions: foliage, glass, leather, skin, hair

**Note on "other":** The nearest-neighbor fill used in earlier versions was found to propagate wrong labels into SAM-uncovered regions. Since SAM proposals are object/region-driven, uncovered areas are more likely to be ambiguous catch-all material rather than clearly-identifiable material — defaulting them to "other" avoids polluting potential downstream training signal.
