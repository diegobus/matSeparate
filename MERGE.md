# MERGE.md — Material Merging & Segmentation Algorithm

Design and implementation plan for the **final stage** of the matSeparate pipeline:
turn a single input image into **material-based object segmentation masks**, using the
trained HGNN patch classifier, bilinear upsampling, and a dense CRF, with output
formatted to match the **MINC** dataset (the baseline is Segment Anything Model on MINC).

> Status: design doc only. No code is written yet. This file is the contract for the
> implementation that follows.

---

## 1. Goal

Given an RGB image, produce:

1. A **semantic per-pixel material label map** (MINC-style single-channel PNG of class
   IDs + a JSON legend), segmented at a **user-chosen level of the taxonomy** (e.g.
   `biotic` vs `abiotic` at depth 2, or full material leaves like `timber`/`marble` at
   depth 5).
2. **Per-object instance masks** derived from that label map, where an *object* is a
   **spatially-connected region of a single material** at the chosen level. (So a wooden
   bowl touching a wooden table becomes **one** wood object; two non-touching wood
   regions are two objects.)

The algorithm must be **customizable to any taxonomy level** and must mirror MINC's
ground-truth file conventions so results are directly comparable to the SAM-on-MINC
baseline.

### Locked design decisions (from requirements)

| Decision | Choice |
|---|---|
| Output format | **Both** a semantic label map (MINC-style) **and** per-object instance masks (COCO-style) |
| Definition of an "object" | **Spatially-connected components** of the same material |
| Taxonomy level selection | **Default = full leaf granularity** (all 37 leaf materials, most detailed). Coarsening is opt-in: cut to a shallower depth `d`, where each leaf maps to its ancestor at depth `d`. A leaf shallower than `d` keeps its own label (preserves detail; only relevant if `d` is pushed past a branch's end). |
| SAM's role | **Baseline only** — not used inside the merging algorithm |
| Low-confidence pixels | Assigned to a dedicated **`background/unknown`** class via a confidence threshold |
| Patch sampling | **Non-overlapping grid, single scale** (default; pluggable for overlap/multi-scale) |

---

## 2. Why bilinear upsampling + dense CRF (assessment)

This is the right call and is **deliberately aligned with MINC itself**. Bell et al. 2015
("Material Recognition in the Wild with the Materials in Context Database") perform full
scene material classification with exactly this recipe: a patch CNN turned into a sliding
predictor produces a coarse probability map, the maps are **upsampled and averaged**, and
a **fully-connected (dense) CRF** (Krähenbühl & Koltun) yields the final per-pixel label.
Matching that recipe makes our method a fair counterpart to the SAM-on-MINC baseline.

Key engineering points baked into this plan:

- **Operate in probability/logit space, never on hard labels.** Upsampling and any
  averaging happen on the per-class probability tensor; `argmax` is taken only at the very
  end. (This is the explicit warning in `INFERENCE_GUIDE.md`.)
- **Grid sampling is intentionally coarse**, so the CRF is the component that recovers
  material boundaries from a blocky unary. The pairwise bilateral term (position + RGB)
  snaps the blocky grid to real image edges. We therefore treat CRF quality as
  first-class, and keep sampling pluggable so overlap/multi-scale can be enabled later for
  a denser unary.
- **Taxonomy-aware CRF compatibility (enhancement):** the label-compatibility matrix can
  be initialized from taxonomy distance so confusing two sibling materials (e.g.
  `granite`↔`marble`) costs less than confusing distant ones (e.g. `granite`↔`fur`). This
  reuses the HGNN's hierarchy and is cheap to add.
- **Cut the taxonomy level *after* CRF**, on leaf-level probabilities, so any level can be
  re-derived without re-running classification or CRF.
- **Dependency caveat:** `pydensecrf` (the standard dense-CRF implementation) is
  unmaintained and can be awkward to build on recent Python/NumPy. We isolate it behind a
  `crf.py` interface with a documented install path and a pure-Python/superpixel fallback
  so the pipeline still runs if the wheel won't build.

Net: bilinear upsampling + dense CRF is appropriate and baseline-faithful; the main risks
are (a) the coarseness of grid unaries and (b) the `pydensecrf` install, both mitigated
above.

---

## 3. Inputs the algorithm builds on (current repo state)

- **Classifier:** `HGNN` (`gnn_classifier/hgnn.py`) wrapped by
  `HGNNInference` (`scripts/infer_api.py`).
  - `HGNNInference.from_run_dir(run_dir)` rebuilds the model from `config.yaml`,
    `node_index.json`, `checkpoint_best.pt`.
  - `api.infer(image)` returns per-image:
    - `leaf_probs` — softmax over the **37 leaf** materials,
    - `node_probs` — sigmoid over **all 58 taxonomy nodes**,
    - `path_nodes` / `path_indices` — decoded root→leaf path,
    - `leaf_idx`, `logits`.
  - `api.infer_batch(images)` does the same for a list (used to batch all patches).
  - Exposed structure we reuse: `api.leaf_indices`, `api.leaf_names`,
    `api.idx_to_node`, `api.node_to_idx`, `api.hierarchy_levels`, and
    `api.model.adjacency_matrix`.
- **Taxonomy:** `taxonomy/tree.py` + `taxonomy/assets/matador-c1-taxonomy.json`.
  Useful helpers we reuse: `get_taxonomy`, `get_hierarchy_levels`,
  `get_hierarchy_mask`, `nx.shortest_path`, `nx.descendants`.
- **Recommended checkpoint:** HGNN `avg_init` run (per `INFERENCE_GUIDE.md`).

### The Matador-C1 taxonomy is ragged (matters for level cuts)

Depths (root = 0) in `taxonomy/assets/matador-c1-taxonomy.json`:

```
0 root
1 solid
2 abiotic, biotic
3 metal, rock, ceramic, polymer, natural, derivative
4 generic_metal(leaf), solid_mass, aggregate, decorative, structural,
  textile, plastic, vegetation, terrain, wood, animal_hide, food
5 granite, limestone, marble, shale, gravel, sand, plaster, pottery, asphalt,
  brick, concrete, nylon, wool, carbon_fiber, carpet, satin, natural_fiber,
  foam, wax, flower, foliage, ivy, grass, moss, plant_litter, soil, straw,
  paper, timber, tree_bark, fur, leather, suede, fruit, vegetable, bread
```

The single irregularity: **`generic_metal` is the only leaf at depth 4**; every other
leaf is at depth 5. Because the **default frontier is the leaf set itself** (not a fixed
depth) and coarsening only ever goes *shallower* (`d ≤ 4`), every leaf always has a
well-defined ancestor at the requested depth — so this irregularity does **not** cause any
ambiguity in practice. It only matters if someone explicitly requests `d = 5`, in which
case `generic_metal` (being shallower than 5) simply keeps its own label.

---

## 4. Algorithm overview

```
input image (H×W×3, arbitrary size)
        │
        ▼
[1] Patch grid extraction            → list of patches + (row,col) grid coords
        │
        ▼
[2] Patch classification (HGNN)      → coarse leaf-prob grid  P_grid (gh×gw×37)
        │
        ▼
[3] Bilinear upsampling              → dense leaf-prob map     P_dense (H×W×37)
        │
        ▼
[4] Dense CRF refinement             → refined leaf-prob/label  (H×W×37)/(H×W)
        │
        ▼
[5] Taxonomy level cut (strict d)    → frontier-prob map        F (H×W×K_d)
        │                               + background/unknown via threshold
        ▼
[6] Label map + connected-component  → semantic map  L (H×W, int ids)
    instance extraction                + instances   [(mask, material, score)]
        │
        ▼
[7] MINC-style + COCO-style export    → label_map.png, labels.json,
                                        instances.json (+ optional viz)
```

Each numbered stage is one module (Section 6). Stages 5–7 are cheap and re-runnable for a
different level without recomputing 1–4 (we cache `P_dense` / the CRF output).

---

## 5. Stage-by-stage design

### [1] Patch grid extraction — `segmentation/patches.py`

- **Default:** tile the image into a **non-overlapping grid** of square patches of side
  `patch_size` (the model's training crop, 224). The bottom/right remainder is handled by
  reflect-padding the image up to a multiple of `patch_size` (so every patch is full-size
  and the grid is rectangular).
- Each patch records its grid index `(gi, gj)` and pixel bounds `(y0,y1,x0,x1)`.
- The **coarse grid resolution** is `gh = ceil(H/patch_size)`, `gw = ceil(W/patch_size)`.
  Each grid cell will hold one 37-vector of leaf probabilities.
- **Pluggable sampler interface** (`PatchSampler`) so `GridSampler` (default) can be
  swapped for `SlidingWindowSampler(stride, scales=[...])` later without touching stages
  2–7. Overlapping/multi-scale samplers accumulate probabilities into the coarse grid (or
  directly into a dense accumulator) with a count buffer for averaging.

**Scale note:** the HGNN was trained on 224 crops of close-up material images. A single
224 patch of a full scene may contain whole objects rather than a material close-up. This
is an inherent domain gap; the pluggable multi-scale sampler is the mitigation path if
leaf accuracy on scenes is poor. Documented as a known risk, not solved by default.

### [2] Patch classification — `segmentation/classify.py`

- Run `HGNNInference.infer_batch` over all patches (batched, on `device`).
- For each patch, keep **`leaf_probs` (37,)** as the canonical distribution. (We use leaf
  softmax, not the sigmoid `node_probs`, because leaf softmax is a proper normalized
  distribution; all higher-level node probabilities are derived by summing descendant
  leaves — Section 5[5].)
- Assemble `P_grid` with shape `(gh, gw, 37)`.
- Optionally also keep a per-cell max-prob (confidence) for diagnostics.
- Caching: `P_grid` keyed by `(image_hash, run_dir, sampler_config)`.

### [3] Bilinear upsampling — `segmentation/upsample.py`

- Upsample `P_grid (gh,gw,37)` → `P_dense (H,W,37)` with **bilinear** interpolation,
  `align_corners=False`, performed per-channel in probability space.
- Renormalize across the 37 channels after interpolation (bilinear mixing can break the
  simplex slightly) so each pixel is a valid distribution.
- Implementation via `torch.nn.functional.interpolate` on a `(1,37,gh,gw)` tensor.
- For overlapping/multi-scale samplers this stage instead reads the dense accumulator and
  divides by the count buffer; the grid path is the simple bilinear case.

### [4] Dense CRF refinement — `segmentation/crf.py`

- Input: original RGB image (`H×W×3`, uint8) + unary from `P_dense`
  (unary energy = `-log P_dense`, shape `(37, H*W)`).
- Use `pydensecrf.DenseCRF2D` with:
  - **Gaussian pairwise** (`addPairwiseGaussian`, `sxy=g_sxy`) — smoothness prior.
  - **Bilateral pairwise** (`addPairwiseBilateral`, `sxy=b_sxy, srgb=b_srgb`) — edge-aware
    term that snaps labels to image color boundaries (the key de-blocking step for grid
    unaries).
  - `n_iterations` mean-field steps (default 5–10).
- **Taxonomy-aware compatibility (optional, on by default):** build a `37×37` label
  compatibility matrix `μ(i,j)` from taxonomy distance between leaf `i` and leaf `j`
  (e.g. shortest-path hops in the tree, normalized), passed as the `compat` argument so
  sibling confusions are penalized less than distant ones. Falls back to Potts
  (`compat=3`) if disabled.
- Output: refined `(37, H, W)` probabilities (we keep soft output, then argmax later) so
  stage 5's level cut can still aggregate.
- **Fallbacks** (selected by config, used if `pydensecrf` unavailable):
  - `none`: skip CRF, use `P_dense` directly.
  - `superpixel`: SLIC/Felzenszwalb superpixels (`skimage.segmentation`) + majority-vote
    of `P_dense` within each superpixel (boundary snapping without the CRF dependency).

### [5] Taxonomy level cut — `segmentation/taxonomy_cut.py`

This is the customization layer. Given the refined **leaf** probability map and a target
`level`, produce a frontier probability map. The `level` is either:

- **`"leaf"` (default, most detailed):** the frontier is the full set of **37 leaf
  materials**. The aggregation matrix `A` is the identity (`K_d = 37`), so
  `generic_metal`, `timber`, `marble`, etc. are all kept distinct. This is the granularity
  used unless the caller asks for something coarser.
- **An integer depth `d` (coarsening, opt-in):** make the classification *less* specific.

**Frontier definition for an integer depth `d`:**
1. `frontier(d)` = the nodes that represent each leaf at depth `d`. Each of the 37 leaves
   maps to its **ancestor at depth `d`** (`nx.shortest_path(root, leaf)[d]`).
2. Shallow-leaf rule: if a leaf's own depth is `< d`, it has no depth-`d` descendant, so it
   **keeps its own label** (maximum detail preserved). Because coarsening always uses
   `d ≤ 4`, this rule is effectively a no-op for this taxonomy (only `generic_metal` at the
   never-coarsening case `d = 5` would trigger it).
3. Build the `(37 → K_d)` aggregation matrix `A` (0/1, row-stochastic) from that mapping.
4. **Frontier probabilities:** `F = P_leaf @ A` (sum the leaf probabilities of all leaves
   under each frontier node). `F` has shape `(H, W, K_d)` and remains a valid distribution.

Keeping `"leaf"` as the canonical cached representation means **any coarser level is one
cheap matrix multiply away** — `recut(level=d)` never recomputes classification or CRF.

**Background/unknown thresholding:**
- Let `conf(p) = max_k F[p, k]`. Pixels with `conf(p) < tau` (config `bg_threshold`) are
  assigned the reserved class id `0 = background/unknown`.
- The frontier classes get ids `1..K_d`.

**Helpers:** depth via `get_hierarchy_levels(graph, "root")`; ancestors via
`nx.shortest_path`; leaf set via out-degree 0 (matching `_get_leaf_indices` in
`scripts/infer_api.py`). `level` is accepted as `"leaf"` (default) or an int depth, with a
future hook for a named-frontier override.

### [6] Label map + connected-component instances — `segmentation/objects.py`

- **Semantic label map** `L (H×W, int)`: `argmax_k F` mapped to frontier class ids, with
  background applied from the threshold above.
- **Instance extraction (objects):**
  - For each frontier material class `c` (excluding background), take the binary mask
    `L == c` and run **connected-components** (`skimage.measure.label`,
    `connectivity=2` / 8-connectivity).
  - Each connected component = one **object** with:
    - `material`: the frontier node name (e.g. `wood`, or `timber` at d=5),
    - `mask`: binary `H×W`,
    - `area`, `bbox`,
    - `score`: mean `conf(p)` over the component's pixels.
  - **Min-area filter** (`min_object_area`) drops specks; optional **morphological
    opening** to clean ragged edges. (Note: gap-bridging/closing across occlusions was
    *not* requested — pure connectivity is used. A `morph_close` flag is left available
    but defaults off.)
- A wooden bowl touching a wooden table share class `wood`/`timber` and are 8-connected →
  a single object, exactly as required.

### [7] Export — `segmentation/formats.py`

Two coordinated outputs per image, in an output dir named after the image + level:

**A. MINC-style semantic segmentation**
- `label_map.png`: single-channel 8-bit PNG, pixel value = class id
  (`0=background/unknown`, `1..K_d` materials). (8-bit is sufficient; `K_d ≤ 37`.)
- `labels.json`: legend `{ "0": "background", "1": "<material>", ... }`, plus metadata
  (`level_depth`, `run_dir`, `bg_threshold`, taxonomy version, image size).
- `label_map_color.png`: optional colorized visualization using a fixed per-material
  palette (for qualitative comparison vs MINC/SAM figures).
- This mirrors MINC's per-pixel material label maps (integer ids + category legend);
  MINC does not ship a single canonical format, so an integer label PNG + JSON legend is
  the faithful, conventional representation.

**B. COCO-style instance masks (SAM-comparable)**
- `instances.json`: COCO-style list of objects:
  `{ id, material, category_id, bbox [x,y,w,h], area, score,
     segmentation: RLE }`. RLE via `pycocotools.mask.encode` (matches how SAM outputs are
  typically stored/evaluated). Image-level header carries `height`, `width`, `level_depth`.
- Optional `instances/` dir of per-object binary PNGs for quick inspection.

**Optional Matador→MINC category crosswalk** (stretch): a mapping file to relabel our
frontier classes into MINC's 23 categories, enabling direct numeric comparison against the
SAM-on-MINC baseline on MINC images. Off by default (kept separate from core export).

---

## 6. Module / file layout

```
segmentation/
  __init__.py
  config.py          # dataclass: SegmentationConfig (all knobs, with defaults)
  patches.py         # PatchSampler, GridSampler (default), SlidingWindowSampler (hook)
  classify.py        # PatchClassifier: wraps HGNNInference -> coarse leaf-prob grid
  upsample.py        # bilinear upsample of coarse grid -> dense prob map
  crf.py             # DenseCRFRefiner (+ superpixel / none fallbacks)
  taxonomy_cut.py    # frontier(depth) construction, strict folding, leaf->frontier matrix
  objects.py         # label map + connected-component instance extraction
  formats.py         # MINC-style label map + legend; COCO-style instances; viz palette
  metrics.py         # confusion matrix, mIoU / mean-class-acc / pixel-acc, legend alignment
  visualize.py       # semantic & instance overlays + composite panel figure
  pipeline.py        # MaterialMerger: orchestrates [1]-[7], caching, multi-level reuse

scripts/
  segment_image.py   # CLI: image + run-dir + level -> outputs (+ --viz, --stub)
  eval_segmentation.py  # CLI: predicted vs GT label maps -> metrics + confusion heatmap

configs/
  segmentation.yaml  # default SegmentationConfig values

tests/
  test_taxonomy_cut.py   # leaf default = identity; frontier sets per depth; A row-sums = 1
  test_objects.py        # connectivity merges adjacent same-material; min-area; bg
  test_formats.py        # label PNG round-trip, RLE round-trip, legend correctness
  test_pipeline_smoke.py # tiny synthetic image end-to-end on CPU
```

### Public API sketch (for review, not final code)

```python
from segmentation.pipeline import MaterialMerger
from segmentation.config import SegmentationConfig

merger = MaterialMerger.from_run_dir(
    "runs/c1_hgnn_baseline/avg_init_20260529_004313",
    config=SegmentationConfig(patch_size=224, bg_threshold=0.5),
    device="auto",
)

result = merger.segment("scene.jpg", level="leaf")     # default: full 37-leaf detail
result.save("out/scene/")                              # writes MINC + COCO outputs

# Re-cut to a less specific level without re-running classify/CRF:
result_l2 = merger.recut(level=2)                      # biotic vs abiotic
result_l2.save("out/scene_level2/")
```

`SegmentationResult` holds: `label_map (H,W)`, `legend`, `instances` (list),
`frontier_probs` (optional), and the cached `dense_leaf_probs` enabling `recut`.

---

## 7. Configuration (`configs/segmentation.yaml`)

```yaml
run_dir: runs/c1_hgnn_baseline/avg_init_20260529_004313
device: auto

sampling:
  type: grid            # grid | sliding (hook)
  patch_size: 224
  pad_mode: reflect
  # sliding-only (future): stride, scales

upsample:
  mode: bilinear
  align_corners: false
  renormalize: true

crf:
  backend: dense        # dense | superpixel | none
  n_iterations: 7
  gaussian_sxy: 3
  bilateral_sxy: 60
  bilateral_srgb: 13
  taxonomy_aware_compat: true

level:
  target: leaf          # "leaf" (default, most detailed) | integer depth to coarsen
                        # e.g. 2 = biotic/abiotic, 3 = metal/rock/.../wood, etc.
  shallow_leaf: keep    # leaves shallower than a requested depth keep their own label

objects:
  bg_threshold: 0.5     # below -> background/unknown (class 0)
  connectivity: 8
  min_object_area: 64   # pixels
  morph_close: false    # gap bridging NOT requested; off

output:
  write_color_viz: true
  write_instance_pngs: false
  minc_crosswalk: null  # optional path to Matador->MINC mapping
```

---

## 8. Data shapes & contracts (quick reference)

| Symbol | Shape | Meaning |
|---|---|---|
| image | `(H, W, 3)` uint8 | input |
| patches | `N × (224,224,3)` | grid tiles (`N = gh*gw`) |
| `P_grid` | `(gh, gw, 37)` | per-cell leaf softmax |
| `P_dense` | `(H, W, 37)` | bilinear-upsampled leaf probs |
| CRF out | `(H, W, 37)` | refined leaf probs |
| `A` | `(37, K_d)` | leaf→frontier aggregation (0/1, row-stochastic) |
| `F` | `(H, W, K_d)` | frontier probs at depth `d` |
| `L` | `(H, W)` int | semantic label map (`0`=bg, `1..K_d`) |
| instances | list | `{id, material, mask(H,W), bbox, area, score, rle}` |

---

## 9. Evaluation (comparability to SAM-on-MINC)

- **Semantic metrics** at a chosen level vs a ground-truth label map: pixel accuracy,
  **mean class accuracy**, and **mean IoU** (MINC reports mean class accuracy). Computed
  with background ignored or as its own class (configurable).
- **Instance metrics** (where instance GT exists): mask AP / mean-best-IoU against
  reference instances. Note MINC GT is *semantic*, not instance, so instance comparison
  against SAM is mostly qualitative unless instance GT is constructed.
- `scripts/eval_segmentation.py` (implemented) consumes our per-image output dirs
  (`label_map.png` + `labels.json`) and a GT directory (+ a GT legend, optional name
  crosswalk), aligns the two id spaces by material name, and prints/writes the metrics
  table plus an optional normalized confusion heatmap.
- `segmentation/visualize.py` (implemented) renders semantic + instance overlays and a
  composite panel (`input | materials | objects [| ground truth]`) for qualitative review;
  exposed via `segment_image.py --viz PATH`.
- Until the trained checkpoints are pulled off the compute node, `segment_image.py --stub`
  runs the full pipeline with a checkpoint-free stub classifier so the plumbing, outputs,
  and visuals can be validated immediately (predicted materials are meaningless).

---

## 10. Dependencies (additions)

| Package | Use | Notes |
|---|---|---|
| `pydensecrf` | dense CRF | unmaintained; document install (`pip install git+https://github.com/lucasb-eyer/pydensecrf.git`); fallback provided |
| `pycocotools` | RLE instance encode/decode | standard COCO/SAM format |
| `scikit-image` | connected components, superpixels, morphology | already in `requirements.txt` |
| `opencv-python` (optional) | fast resize / morphology | optional; torch/skimage cover defaults |

`numpy`, `torch`, `networkx`, `pillow` already present. `requirements.txt` to be updated
when implementation starts.

---

## 11. Edge cases & decisions

- **Ragged tree / default leaf detail:** the default `"leaf"` frontier keeps all 37 leaves
  distinct (including `generic_metal`). Coarsening goes shallower (`d ≤ 4`), where every
  leaf has a clean depth-`d` ancestor, so no folding ambiguity arises; unit-tested across
  depths 1–4. (`d = 5` would just keep `generic_metal` as itself.)
- **Liquid/gas:** Matador-C1 only contains `solid`; those branches never appear. Depth-1
  cut effectively yields a single `solid` class (plus background) — documented, not a bug.
- **Image-size invariance:** reflect-pad to a multiple of `patch_size`; crop outputs back
  to original `H×W` before export.
- **Renormalization:** after bilinear upsampling and after taxonomy aggregation, re-project
  onto the simplex so thresholds/CRF unaries stay valid.
- **Determinism:** fixed palette + sorted class ids so label PNGs are reproducible.
- **CRF absent:** auto-fallback to `superpixel` (or `none`) with a logged warning, so the
  pipeline never hard-fails on a missing native dependency.
- **Coarse grid blockiness:** acknowledged; CRF + (optional later) overlap/multi-scale are
  the levers. Grid is the requested default.

---

## 12. Implementation milestones

1. **Scaffolding & config** — `segmentation/` package, `SegmentationConfig`,
   `configs/segmentation.yaml`. (No model calls yet.)
2. **Patch → classify → upsample** — `patches.py`, `classify.py`, `upsample.py`; verify
   `P_dense` on a real image; visualize argmax (pre-CRF) sanity map.
3. **Taxonomy cut** — `taxonomy_cut.py` + `test_taxonomy_cut.py` (leaf default = identity,
   frontier sets, `A` row-sums = 1). Validate `"leaf"` plus coarsening depths 1–4.
4. **CRF** — `crf.py` with dense backend + superpixel/none fallbacks; before/after viz.
5. **Objects & export** — `objects.py`, `formats.py`; MINC label map + COCO instances;
   round-trip tests.
6. **Pipeline + CLI** — `pipeline.py` (`MaterialMerger`, caching, `recut`),
   `scripts/segment_image.py`; end-to-end smoke test.
7. **Evaluation (optional)** — `scripts/eval_segmentation.py`, mIoU/mean-class-acc,
   optional Matador→MINC crosswalk for direct baseline comparison.
8. **Tuning** — CRF params, `bg_threshold`, `min_object_area`; decide if multi-scale
   sampling is needed for scene-domain accuracy.

---

## 13. Open / deferred items (not blocking)

- Whether to enable **multi-scale sliding window** by default (depends on observed
  scene-domain leaf accuracy vs the 224 close-up training regime).
- Exact **Matador→MINC 23-category crosswalk** (only needed for numeric comparison on MINC
  images).
- Whether `background/unknown` should be **ignored** vs **scored** in evaluation.
- Optional **instance GT construction** if instance-level comparison to SAM is desired.
```
