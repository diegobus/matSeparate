# MERGE.md — Material Merging & Segmentation Algorithm

The **final stage** of the matSeparate pipeline: turn a single input image into
**material-based segmentation masks** using the trained HGNN patch classifier, then
upsampling, CRF refinement, and taxonomy-aware post-processing. Output is formatted to
match the **MINC** dataset so results are comparable to the Segment-Anything-on-MINC
baseline.

> Status: **implemented.** This document describes the current architecture and the
> rationale (and supporting experiments) behind each design choice. Module names and
> config keys match the code in `segmentation/`, `scripts/`, and `configs/segmentation.yaml`.

---

## 1. Goal

Given an RGB image, produce:

1. A **semantic per-pixel material label map** (MINC-style single-channel PNG of class IDs
   + a JSON legend), segmented at a **user-chosen level of the taxonomy** (e.g. `biotic`
   vs `abiotic` at depth 2, or full material leaves like `timber`/`marble`).
2. **Per-object instance masks** derived from that label map, where an *object* is a
   **spatially-connected region of a single material** at the chosen level. (A wooden bowl
   touching a wooden table is **one** wood object; two non-touching wood regions are two.)

### Locked design decisions

| Decision | Choice | Why |
|---|---|---|
| Output format | **Both** a semantic label map (MINC-style PNG + JSON) **and** per-object instance masks (COCO-style RLE/JSON) | matches both MINC (semantic) and SAM (instance) baselines |
| Definition of an "object" | **Spatially-connected components** of the same material | per requirement (bowl-on-table = one object) |
| Taxonomy level | **Default = full leaf granularity** (37 leaves). Coarsening is opt-in: cut to depth `d`, each leaf → its ancestor at depth `d`; a leaf shallower than `d` keeps its own label | most detail by default; coarsening is one matmul on cached leaf probs |
| SAM's role | **Baseline only** — never used inside the algorithm | |
| Low-confidence pixels | **Argmax everywhere by default** (`bg_threshold = 0.0`); a positive threshold re-enables a `background/unknown` class | for mask-similarity vs MINC, every pixel should carry a label (MINC GT labels every pixel) |
| Patch sampling | **Sliding window, default `window=48`, `stride=24`**, with optional `[min,max]` patch-count bounds and multi-scale context; non-overlapping `grid` still available | empirically the cleanest masks (Section 5.1); overlap-averaging suppresses per-tile noise |

---

## 2. Why bilinear upsampling + dense CRF (assessment)

Deliberately **aligned with MINC itself.** Bell et al. 2015 do full-scene material
classification with exactly this recipe: a patch CNN as a sliding predictor produces a
coarse probability map, the maps are **upsampled and averaged**, and a **fully-connected
(dense) CRF** (Krähenbühl & Koltun) yields the per-pixel label. Matching that recipe makes
our method a fair counterpart to SAM-on-MINC.

Engineering principles baked in:

- **Operate in probability space, never on hard labels.** Upsampling, multi-scale
  averaging, smoothing, and aggregation all happen on per-class probabilities; `argmax` is
  taken only at the very end (the explicit warning in `INFERENCE_GUIDE.md`).
- **Cut the taxonomy level *after* refinement**, on cached leaf probabilities, so any level
  is one cheap matmul away (`recut` never re-runs classify/CRF).
- **CRF is treated as first-class** for de-blocking grid unaries, but is **isolated behind
  a fallback chain** because of a real dependency problem (Section 5.4).

---

## 3. Inputs the algorithm builds on

- **Classifier:** `HGNN` (`gnn_classifier/hgnn.py`) wrapped by `HGNNInference`
  (`scripts/infer_api.py`). `from_run_dir` rebuilds it from `config.yaml` /
  `node_index.json` / `checkpoint_best.pt`. `infer_batch` returns `leaf_probs` (softmax
  over **37 leaves**), `node_probs` (sigmoid over **58 nodes**), and the decoded path.
  - **Checkpoint choice:** use an **HGNN** run, *not* the `c1_resnet50_baseline` (flat
    classifier with no hierarchy — it cannot drive level cuts, taxonomy-aware smoothing, or
    hierarchical consistency). Among HGNN runs (`avg_init` vs `rand_init`), pick by
    validation `metrics.json`; we default to `avg_init` (`prototypes.init: cnn_average`,
    generally the more stable init).
- **Taxonomy:** `taxonomy/tree.py` + `taxonomy/assets/matador-c1-taxonomy.json`.

### The Matador-C1 taxonomy is ragged (matters for level cuts)

`generic_metal` is the only **leaf at depth 4**; all other leaves are at depth 5. Because
the default frontier is the leaf set itself and coarsening only goes shallower (`d ≤ 4`),
every leaf always has a well-defined depth-`d` ancestor, so the irregularity causes no
ambiguity. It matters only if `d = 5` is explicitly requested, where `generic_metal`
simply keeps its own label.

---

## 4. Architecture overview

```
input image (H×W×3, arbitrary size)
        │
        ▼
[1] Patch sampling (sliding window, optional multi-scale)   → patches + grid coords
        │
        ▼
[2] Patch classification (HGNN)                             → coarse leaf-prob grid P_grid (gh×gw×37)
        │   (multi-scale: average P_grid over context scales)
        ▼
[2b] Taxonomy-aware grid smoothing (optional)               → smoothed P_grid
        │
        ▼
[3] Bilinear upsampling                                     → dense leaf-prob map P_dense (H×W×37)
        │
        ▼
[4] CRF refinement (dense → superpixel → none fallback)     → refined leaf probs (H×W×37)
        │
        ▼
[4b] Hierarchical consistency (optional)                    → branch-restricted leaf probs
        │
        ▼
[5] Taxonomy level cut (leaf default, or depth d)           → frontier-prob map F (H×W×K)
        │   (+ optional background via bg_threshold)
        ▼
[6] Label map + connected-component instances               → semantic map L + objects
        │
        ▼
[7] MINC-style + COCO-style export (+ optional viz)
```

Stages `[2b]`, `[4b]` and the multi-scale part of `[2]` are **optional, default-off /
identity** — turning them all off reproduces the original plain pipeline. They exist to
fight one observed failure mode (Section 5.6). Stages 5–7 are cheap and re-runnable for a
different level via `recut` (the refined leaf-prob map is cached on the result).

---

## 5. Stage-by-stage design & rationale

### 5.1 Patch sampling — `segmentation/patches.py`

Two samplers behind a small `PatchSampler` interface:

- **`GridSampler`** — non-overlapping `patch_size` (224) tiles, reflect-padded. Fast and
  coarse; kept as a fallback / baseline.
- **`SlidingWindowSampler` (default)** — overlapping windows of `window_size` at spacing
  `stride`, each later resized to 224. A *small* window both densifies the prediction grid
  **and** moves each crop closer to the close-up material training distribution.

**Why sliding window at ~48/24 by default — empirical.** A window-size sweep on a MINC
scene (leaf level) showed a clear **U-shape**, not "smaller is always better":

| window / stride | ≈ patches | objects | mask quality |
|---|---|---|---|
| 224 (grid) | 4 | 2 | two giant blobs |
| 64 / 32 | 121 | 28 | noisy, fragmented |
| **48 / 24** | ~576 | 15 | **cleanest — coherent wall/sky/gravel** |
| 32 / 16 | ~1369 | 13 | clean, slight speckle |
| 24 / 12 | ~2560 | 27 | degrades — regions shatter |

Below ~32px each tile loses context and the classifier flips labels between neighbors,
*re-introducing* fragmentation. The win at 48 comes mostly from **overlap-averaging** (each
pixel is a vote over several windows), not from tiles being "more single-material."

**Patch-count bounds (`min_patches`/`max_patches`).** Patch count scales with image area,
so a large image explodes (a 50 MB photo ≈ tens of thousands of patches). When bounds are
set, the sampler **adapts the stride per image** so total patches land in `[min, max]`
regardless of resolution (coarsen the stride if too many; densify if too few; shrink the
window only as a last resort for images barely larger than the window). `max_patches` is a
hard cap. Unset = configured stride used verbatim (unchanged behavior).

**Multi-scale context (`scales`, `sample_scales`).** See 5.6 — same grid sampled at several
crop scales centered on each position; the per-scale probability grids are averaged.

### 5.2 Patch classification — `segmentation/classify.py`

- `PatchClassifier` runs `HGNNInference.infer_batch` over patches (batched on `device`) and
  assembles `P_grid (gh, gw, 37)` from **`leaf_probs`** — the proper normalized simplex;
  all coarser node probabilities are derived later by summing descendant leaves.
- A `tqdm` **progress bar** over batches (graceful no-op if `tqdm` missing) and a `Done in
  Xs` timing line surface long CPU runs.
- `StubLeafPredictor` enables checkpoint-free end-to-end tests/plumbing validation.

### 5.3 Bilinear upsampling — `segmentation/upsample.py`

`P_grid → P_dense (H,W,37)` per-channel via `F.interpolate` (`align_corners=False`), then
renormalized onto the simplex (bilinear mixing breaks normalization slightly).

### 5.4 CRF refinement — `segmentation/crf.py`

Edge-aware refinement that snaps blocky grid unaries to image boundaries, with a
**graceful fallback chain**:

```
dense (pydensecrf)  ──missing──▶  superpixel (skimage SLIC/Felzenszwalb)  ──missing──▶  none
```

- **Dense backend:** `pydensecrf.DenseCRF2D` with Gaussian + bilateral pairwise terms.
- **Taxonomy-aware compatibility (default on for dense):** a `37×37` label-compatibility
  matrix built from normalized taxonomic distance (`build_taxonomy_compat`), so confusing
  siblings (`granite`↔`marble`) costs less than confusing distant materials
  (`granite`↔`fur`).
- **Reality:** `pydensecrf` is unmaintained and **fails to build on the compute node's
  Python 3.13** (Cython/Eigen errors) and is awkward on macOS. In practice the pipeline
  runs on the **superpixel** backend, which averages probabilities within SLIC superpixels
  — most of the boundary-snapping benefit with zero native-build risk. The fallback is
  automatic with a logged warning, so the pipeline never hard-fails.

### 5.5 Taxonomy level cut — `segmentation/taxonomy_cut.py`

The customization layer. `build_frontier(level)` returns a `(37 → K)` 0/1 aggregation
matrix `A`; `apply_frontier` computes `F = P_leaf @ A`.

- **`"leaf"` (default):** `A = I`, all 37 leaves distinct.
- **integer depth `d`:** each leaf → ancestor at depth `d`; shallow leaves keep their own
  label. Unit-tested across depths 1–4.

Caching `"leaf"` as canonical means **any coarser level is one matmul away** (`recut`).

### 5.6 Global-context post-processing (the flip fix)

**Observed failure mode.** A per-patch diagnostic (`scripts/patch_diagnostic.py`) showed
the classifier is **low-confidence (≈0.3–0.5) and spatially inconsistent**: adjacent,
clearly-single-material tiles get different labels — e.g. one brick wall reads as
`pottery`/`generic_metal`/`foliage`, and a single lemon surface flips between `fruit`,
`wax`, and `foam` (i.e. across the **biotic/abiotic** split). This is a **domain gap** (the
HGNN was trained on close-up material crops, not scene tiles), and it fragments masks even
when tiles are pure. Three levers attack it **without retraining**:

**(a) Multi-scale context windows** — `patches.SlidingWindowSampler.sample_scales`,
`sampling.scales`. For each grid position, classify the local crop *and* larger crops
(`window×scale`) centered on the same point, then average. The larger crop "sees" the whole
lemon → stable `fruit`; the local crop preserves boundaries. Cost: one forward pass per
scale.

**(b) Taxonomy-aware grid smoothing** — `context.smooth_grid_taxonomy`, `smoothing` config
(default off). A mean-field update on the coarse grid:
`P ← normalize(P · exp(weight · (neighbor_avg @ sim)))`, where `sim = 1 − normalized
taxonomic distance`. Neighbors reinforce **taxonomically compatible** labels, so a
biotic↔abiotic flip is penalized far more than a sibling swap, cleaning isolated flips.

**(c) Hierarchical consistency** — `context.enforce_hierarchical_consistency`, `hierarchy`
config (default off, **experimental**). Per pixel, decide the coarse branch (argmax of
probabilities aggregated to `decision_depth`, e.g. 2 = biotic/abiotic), then **zero leaves
outside the winning branch** and renormalize. This *structurally* forbids a pixel from
being both biotic and abiotic. Applied to the refined dense leaf probs (so all level cuts
inherit the decision).

All three are config-/CLI-gated and default to off/identity; the baseline pipeline is
unchanged unless they are enabled.

### 5.7 Label map + connected-component instances — `segmentation/objects.py`

- **`build_label_map`**: `argmax_k F` → ids `1..K`. With the default `bg_threshold = 0.0`
  no pixel is ever background (every pixel takes its best label — what mask-similarity vs
  MINC wants); a positive threshold re-enables `0 = background/unknown` for low-confidence
  pixels.
- **`extract_instances`**: per material class, connected components (`scipy.ndimage`,
  4/8-connectivity) → one `Instance` each (`material`, `mask`, `bbox`, `area`, `score` =
  mean confidence). `min_object_area` drops specks; optional `morph_close` (off) bridges
  gaps. Adjacent same-material regions merge into one object as required.

### 5.8 Export — `segmentation/formats.py`

- **MINC-style:** `label_map.png` (8-bit class ids), `labels.json` (legend + metadata),
  optional `label_map_color.png` (fixed palette).
- **COCO-style:** `instances.json` with **RLE implemented natively** (no `pycocotools`
  dependency), `{id, material, category_id, bbox, area, score, segmentation: RLE}`.

---

## 6. Module / file layout

```
segmentation/
  config.py          # SegmentationConfig + nested dataclasses (all knobs)
  patches.py         # PatchSampler, GridSampler, SlidingWindowSampler (+min/max bounds, multi-scale)
  classify.py        # PatchClassifier (HGNN adapter, progress bar) + StubLeafPredictor
  upsample.py        # bilinear upsample of coarse grid -> dense prob map
  crf.py             # CRF refine: dense (pydensecrf) -> superpixel -> none, + taxonomy compat
  context.py         # taxonomy-aware grid smoothing + hierarchical consistency
  taxonomy_cut.py    # build_frontier(level) -> leaf->frontier matrix; apply_frontier
  objects.py         # label map (argmax/bg) + connected-component instance extraction
  formats.py         # MINC label map + legend; COCO RLE instances; viz palette
  metrics.py         # semantic metrics + class-agnostic mask-similarity metrics
  visualize.py       # semantic & instance overlays, composite + level-comparison panels
  pipeline.py        # MaterialMerger: orchestrates [1]-[7], caching, recut

scripts/
  segment_image.py   # main CLI (sampler/scales/smooth/hierarchy/bg-threshold/viz/stub)
  patch_diagnostic.py# per-patch prediction montage (isolates classifier vs merging)
  compare_masks.py   # class-agnostic mask-similarity vs a reference (e.g. MINC GT)
  eval_segmentation.py # semantic metrics vs GT label maps

configs/segmentation.yaml   # default config
tests/                       # taxonomy_cut, objects, formats, pipeline_smoke, metrics,
                             # visualize, patches (bounds), context (scales/smooth/hierarchy)
```

### Public API

```python
from segmentation.pipeline import MaterialMerger
from segmentation.config import SegmentationConfig

merger = MaterialMerger.from_run_dir(
    "runs/c1_hgnn_baseline/avg_init_20260529_004313",
    config=SegmentationConfig(), device="cpu",
)
result = merger.segment("scene.jpg", level="leaf")  # default: full 37-leaf detail
result.save("out/scene/")
result_l2 = merger.recut(level=2)                   # biotic vs abiotic, no re-classify
```

---

## 7. Configuration (`configs/segmentation.yaml`)

```yaml
run_dir: runs/c1_hgnn_baseline/avg_init_20260529_004313
device: cpu             # torch_geometric scatter ops crash on Apple MPS; cpu locally, cuda on the node

sampling:
  type: sliding         # sliding (default, finer) | grid (coarse)
  patch_size: 224       # grid tile size / model input
  window_size: 96       # sliding crop size in image px (resized to 224)
  stride: 48            # sliding spacing; smaller -> finer/denser/slower
  pad_mode: reflect
  batch_size: 32
  min_patches: null     # if set, stride auto-adapts so patches >= this
  max_patches: null     # if set, stride auto-adapts so patches <= this
  scales: [1.0]         # context scales; e.g. [1.0, 2.0, 3.0] averages multi-scale crops

upsample: {mode: bilinear, align_corners: false, renormalize: true}

crf:
  backend: dense        # dense (pydensecrf) | superpixel | none  (auto-falls back)
  n_iterations: 7
  gaussian_sxy: 3; bilateral_sxy: 60; bilateral_srgb: 13
  gaussian_compat: 3; bilateral_compat: 10
  taxonomy_aware_compat: true
  superpixel_method: slic; superpixel_n_segments: 400

smoothing:              # taxonomy-aware coarse-grid smoothing (flip fix)
  enabled: false; n_iter: 2; weight: 1.0; connectivity: 8

hierarchy:              # hierarchical consistency (experimental)
  enabled: false; decision_depth: 2; shallow_leaf: keep

level:
  target: leaf          # "leaf" (default) | integer depth to coarsen
  shallow_leaf: keep

objects:
  bg_threshold: 0.0     # 0 = argmax everywhere; raise to gate low-confidence px
  connectivity: 8; min_object_area: 64; morph_close: false

output:
  write_color_viz: true; write_instance_pngs: false; minc_crosswalk: null
```

Key CLI overrides on `scripts/segment_image.py`: `--sampler`, `--window-size`, `--stride`,
`--min-patches`, `--max-patches`, `--scales`, `--smooth`/`--smooth-iters`/`--smooth-weight`,
`--hierarchical`/`--decision-depth`, `--bg-threshold`, `--level`, `--compare-levels`,
`--viz`, `--stub`, `--device`.

---

## 8. Data shapes (quick reference)

| Symbol | Shape | Meaning |
|---|---|---|
| `P_grid` | `(gh, gw, 37)` | per-cell leaf softmax (averaged over scales if multi-scale) |
| `P_dense` | `(H, W, 37)` | bilinear-upsampled leaf probs |
| refined | `(H, W, 37)` | CRF (+ optional hierarchy) refined leaf probs |
| `A` | `(37, K)` | leaf→frontier aggregation (0/1) |
| `F` | `(H, W, K)` | frontier probs at chosen level |
| `L` | `(H, W)` int | semantic label map (`1..K`, `0`=bg only if threshold>0) |

---

## 9. Evaluation

**Class-agnostic mask similarity is the primary metric** (`segmentation/metrics.py`,
`scripts/compare_masks.py`). The goal is "do our masks match MINC's masks?", **independent
of label identity** — so we compare the two segmentations purely as **partitions of the
pixels**: Adjusted Rand Index, Variation of Information, Segmentation Covering (both
directions), mean best-IoU, and boundary-F1. `--connected` compares spatial regions
(instance-like) rather than label classes. This is the right lens because the classifier's
labels are unreliable (Section 5.6) but spatially *consistent* labels still yield masks
that match MINC even when the material name is "wrong."

Also available:
- **Semantic metrics** (`scripts/eval_segmentation.py`): pixel acc, mean class acc, mIoU,
  confusion matrix vs a GT label map, aligning id spaces by material name (+ optional
  crosswalk). Useful only if/when label correctness matters.
- **Visualization** (`segmentation/visualize.py`): semantic/instance overlays, composite
  `input | materials | objects` panel, and multi-level comparison; via `--viz`.
- **Diagnostics** (`scripts/patch_diagnostic.py`): per-patch prediction montage to separate
  *classifier* quality from *merging* quality.

---

## 10. Dependencies

| Package | Use | Notes |
|---|---|---|
| `torch`, `torch_geometric`, `timm` | HGNN inference | `timm` is the CNN backbone; PyG needs CUDA (or CPU) — **MPS unsupported** |
| `scikit-image` | superpixel CRF, morphology | the working CRF backend in practice |
| `scipy` | connected components, smoothing convolution | |
| `pyyaml`, `pillow`, `numpy`, `networkx`, `matplotlib`, `tqdm` | config, IO, viz, progress | |
| `pydensecrf` | dense CRF | **optional; fails to build on Py3.13** — superpixel fallback used |
| `pycocotools` | — | **not needed**; RLE implemented natively in `formats.py` |

**Runtime / device.** `torch_geometric`'s scatter ops crash on Apple **MPS**
(`Placeholder storage has not been allocated`), so `device` defaults to **cpu** locally; the
**CUDA compute node** (where checkpoints live) is the place for fast / large / multi-scale
runs.

---

## 11. Edge cases & decisions

- **Argmax default, optional background:** `bg_threshold=0.0` labels every pixel (best for
  MINC mask comparison); raising it reinstates `background/unknown`.
- **Image-size invariance:** reflect-pad to fit windows; outputs cropped back to `H×W`.
- **Patch-count bounds** prevent huge images from exploding (and tiny ones from
  under-sampling); `max_patches` is a hard cap.
- **Renormalize** onto the simplex after every probability-space op (upsample, smoothing,
  aggregation, hierarchy masking).
- **Determinism:** fixed palette + sorted class ids → reproducible label PNGs.
- **CRF/MPS/pydensecrf** failures degrade gracefully (fallback backend, cpu device) rather
  than hard-failing.

---

## 12. Known limitations & future work

- **Classifier domain gap is the dominant error source**, not the merging algorithm. The
  per-patch diagnostic confirms low-confidence, spatially-inconsistent predictions on scene
  tiles. The real fix is classifier-side (fine-tune / domain-adapt on scene patches); the
  Section 5.6 levers are inference-time mitigations.
- **Dense CRF** is currently unavailable on the node (Py3.13); a native NumPy mean-field /
  joint-bilateral implementation would restore taxonomy-aware dense refinement without the
  `pydensecrf` build.
- **Matador→MINC crosswalk** is only needed for *semantic* comparison; the class-agnostic
  mask metrics sidestep it.
- **Checkpoint selection** (`avg_init` vs `rand_init`) should be confirmed against
  validation `metrics.json` on the node.
```
