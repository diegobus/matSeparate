# HGNN Segmentation Evaluation On MINC-S

## Branch Context

- `origin/sam` contains the reproducible SAM MINC-S baseline:
  - `scripts/select_minc_s_targeted_subset.py`
  - `scripts/build_minc_s_filtered_segments.py`
  - `scripts/evaluate_sam_minc.py`
  - `segmentation/sam_baseline.py`
  - `scripts/build_sam_report_assets.py`
- `origin/matador-c1-pipeline` contains the HGNN material segmentation pipeline:
  - `segmentation/pipeline.py`
  - `segmentation/classify.py`
  - `segmentation/metrics.py`
  - `scripts/segment_image.py`
  - `scripts/eval_segmentation.py`
  - `scripts/compare_masks.py`

The SAM branch evaluates binary MINC-S segment masks. The HGNN pipeline produces a dense
semantic label map in the Matador C1 taxonomy by classifying image patches, upsampling
leaf probabilities, optionally applying CRF refinement, then cutting the taxonomy.

## Existing SAM Protocol

The curated subset has 200 MINC-S images and 1067 annotated segments, selected for
overlap with Matador-relevant material categories.

SAM automatic mode is a proposal-quality evaluation:

- Generate automatic masks once per image.
- For each MINC-S ground-truth binary segment, choose the SAM proposal with highest IoU.
- Report mean best IoU, Recall@0.25/0.50/0.75, masks per image, and sec/image.

SAM oracle point mode is an upper-bound localization evaluation:

- Give SAM one oracle positive point per ground-truth segment.
- Keep the best returned multimask candidate by IoU.
- Report mean IoU, mean Dice, image setup time, and prompt time per segment.

These two protocols do not test semantic material classification. They test how well SAM
can recover material-shaped regions when either unprompted or given oracle localization.

## What The HGNN Pipeline Produces

The current HGNN segmentation pipeline is semantic, not proposal-only:

```text
image
-> grid/sliding patches
-> HGNN leaf probabilities per patch
-> dense probability map
-> CRF refinement
-> Matador taxonomy cut
-> label_map.png + connected material components
```

So the natural HGNN outputs are:

- a per-pixel Matador material label;
- connected components for each predicted material label;
- per-pixel confidence;
- optional re-cuts to coarser taxonomy levels without rerunning the classifier.

This means HGNN can be compared to SAM in two different ways:

1. **Class-agnostic region recovery**, matching SAM automatic proposal metrics.
2. **Semantic material segmentation**, where labels are mapped between Matador and MINC.

## Recommended Evaluation Extension

### 1. Use The Same 200-Image / 1067-Segment MINC-S Subset

Use the already built filtered segment list from the SAM branch:

```text
out/minc_s_targeted_subset/test_segments_top200_matador_overlap.txt
```

Keep the same `resize_photo_to_mask` alignment strategy:

- load the MINC-S binary segment mask;
- resize the full photo to the mask resolution before running segmentation;
- score predictions at the mask resolution.

This keeps the HGNN comparison aligned with the SAM results.

### 2. Class-Agnostic HGNN Region Proposal Score

For each image:

- run `MaterialMerger.segment(...)`;
- split the predicted `label_map` into connected components;
- optionally drop background and tiny regions using the pipeline's `min_object_area`;
- for each MINC-S ground-truth binary segment, compute IoU against every HGNN connected
  component and keep the best match.

Report the same automatic-proposal metrics as SAM:

- mean best IoU;
- Recall@0.25;
- Recall@0.50;
- Recall@0.75;
- mean predicted components per image;
- mean inference time per image.

This is the cleanest apples-to-apples comparison with SAM automatic masks because it does
not require the predicted material class to match the MINC label. It answers:

> Does the HGNN material segmentation pipeline recover material-region shapes as well as
> SAM proposals?

This is also the most defensible comparison when Matador and MINC vocabularies differ.

### 3. Semantic Matador-to-MINC Score

For the subset of MINC categories that have plausible Matador equivalents, score semantic
predictions using a crosswalk.

Existing mapping precedent from `scripts/eval_hgnn_on_minc.py`:

```text
brick -> brick
carpet -> carpet
ceramic/tile -> pottery
fabric -> natural_fiber, wool, satin, nylon, suede
foliage -> foliage, grass, ivy, moss
food -> bread, fruit, vegetable
leather -> leather
metal -> generic_metal
paper/wallpaper -> paper
plastic -> foam, wax
polishedstone -> marble, granite, limestone
stone -> shale, granite, limestone, marble, gravel, sand
wood -> timber, tree_bark
```

Skipped or weak categories should be reported explicitly:

```text
glass, hair, mirror, other, painted, skin, sky, water
```

For every ground-truth MINC-S segment:

- identify all pixels in the segment;
- map HGNN Matador labels inside that mask into MINC categories where possible;
- compute segment-level material accuracy by majority vote or mean probability;
- compute semantic IoU only for mapped classes.

Recommended semantic metrics:

- segment classification accuracy on mapped MINC categories;
- per-class accuracy;
- hierarchical correctness if using a MINC taxonomy;
- semantic mIoU over mapped classes;
- coverage count: total segments, mapped segments, skipped segments.

This should be reported as **mapped-label semantic transfer**, not as a direct full-MINC
segmentation benchmark.

### 4. Optional Oracle-Style HGNN Upper Bound

SAM oracle point is not directly comparable to the current HGNN pipeline because HGNN does
not take a point prompt. The closest upper-bound variant is:

- run HGNN once per image;
- for each ground-truth segment, choose the predicted connected component with highest IoU;
- additionally report whether that component's semantic label maps to the ground-truth
  MINC category.

This is effectively an oracle region-selection evaluation over HGNN components. It is fair
as an upper bound, but it must not be described as a deployable pipeline result.

## Proposed Output Schema

Mirror the SAM auto CSV where possible:

```text
mode
photo_id
shape_id
label_index
label_name
mask_path
photo_path
gt_area
best_component_index
best_iou
best_dice
matched_pred_area
matched_label
matched_confidence_mean
matched_maps_to_gt
alignment_strategy
oracle_component_selection
```

Aggregate JSON should mirror SAM:

```text
mode: "hgnn_components"
model: checkpoint/run dir
alignment_strategy: "resize_photo_to_mask"
dataset_summary: ...
metrics:
  overall:
    num_images
    num_segments
    mean_best_iou
    recall@0.25
    recall@0.50
    recall@0.75
    mean_num_components_per_image
    mean_inference_time_sec_per_image
  semantic_mapped:
    num_mapped_segments
    num_unmapped_segments
    mapped_segment_accuracy
    per_class_accuracy
```

## Main Caveats

- SAM automatic masks are generic object/region proposals; HGNN components are produced
  by semantic material classification plus smoothing. Equal IoU metrics are useful, but
  the systems are not optimizing exactly the same objective.
- MINC-S labels are 23 broad material categories. Matador C1 has a different hierarchy
  and leaf set. Any semantic comparison must disclose the crosswalk and the skipped
  categories.
- The HGNN pipeline may underperform SAM at exact boundaries if patch stride is coarse.
  Report stride/window/CRF settings and runtime together.
- If the HGNN patch classifier was trained only on local appearance crops, it may fail on
  broader context images. This should be treated as part of the study: local patch model
  vs larger-context model vs SAM.

## Recommended First Run

Use a fast but credible setting:

```text
sampling.type: sliding
sampling.window_size: 96
sampling.stride: 48
objects.bg_threshold: 0.0
objects.min_object_area: 64
crf.backend: superpixel or none for first pass; dense CRF for final pass
level.target: leaf
```

Then run a smaller ablation:

- stride 96 vs 48;
- CRF none vs superpixel/dense;
- taxonomy cut at leaf vs coarser mapped MINC-compatible frontier;
- old HGNN fixed head vs node-wise HGNN checkpoint, once available.

