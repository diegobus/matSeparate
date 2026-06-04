# Material Segmentation — Session Findings

A running record of what we built, what worked, and what didn't while developing the
patch-classifier → segmentation "merging" pipeline. Intended as a quick reference for the
writeup and for teammates picking this up. See `MERGE.md` for the full design doc.

---

## Goal

Turn the trained HGNN **patch classifier** into a full-image **material segmentation**
pipeline, and compare its masks against the MINC dataset (with SAM as the external
baseline). Core flow:

```
image → overlapping patches → HGNN leaf probs → bilinear upsample
      → dense CRF → taxonomy-level cut → connected-component objects → MINC/COCO export
```

---

## What worked

| Component | Status | Notes |
|---|---|---|
| **Sliding-window sampler** | ✅ kept as default | Overlapping windows (default `window=96, stride=48`) gave far finer, less "blocky" masks than the original non-overlapping grid. Overlap-averaging also reduces per-patch noise. |
| **Dynamic patch-count bounds** | ✅ | `--min-patches / --max-patches` auto-adapt the stride (then window as a last resort) so patch count is stable regardless of image size. |
| **Bilinear upsampling in probability space** | ✅ | Interpolating per-leaf distributions (not hard labels), then renormalizing to the simplex, gives the CRF soft confidence to work with. Mirrors MINC's recipe. |
| **Dense CRF (real)** | ✅ locally | `pydensecrf` is installed in the local `cs231n-project` env, so the genuine Krähenbühl–Koltun fully-connected CRF runs on the Mac. Snaps fuzzy boundaries to image edges. |
| **Graceful CRF fallback** | ✅ | `dense → superpixel (SLIC averaging) → none`. Lets the pipeline run on the VM where `pydensecrf` won't build. |
| **Taxonomy-aware CRF compatibility** | ✅ (default on) | Off-diagonal penalties scale with taxonomic distance, so sibling-material confusions cost less than distant ones. |
| **Connected-component objects** | ✅ | Same-material spatially-connected regions become one object; `min_object_area` drops specks. |
| **Taxonomy-level recut** | ✅ | Cheap re-cut to coarser levels (e.g. biotic/abiotic) from cached refined leaf probs — no re-classification. |
| **Argmax labelling (`bg_threshold=0`)** | ✅ | Every pixel takes its highest-probability label (no "unknown" holes), which is what we want for mask-similarity comparison. |
| **Class-agnostic mask metrics** | ✅ | ARI, Variation of Information, segmentation covering, mean best-IoU, boundary-F1 (`compare_masks.py`) — the right family for "is the partition similar?" vs SAM, independent of class names. |
| **Patch diagnostic montage** | ✅ | Per-patch prediction grid isolates *classifier* quality from *merging* quality. Very informative (see findings below). |
| **Visualization + outputs** | ✅ | Composite panels, level comparisons, MINC label maps + native COCO-style RLE instances. |
| **Progress bars + timing** | ✅ | `tqdm` + runtime prints on the long-running CLIs. |
| **MINC classifier is strongly in-domain** | ✅ key finding | On scene patches it is confident and spatially consistent; C1 is not (see below). |

---

## What didn't work (and was removed or worked around)

| Thing | Outcome | Why |
|---|---|---|
| **Taxonomy-aware grid smoothing** | ❌ removed | The neighbor-agreement penalty out-voted thin structures: on `minc_1` it merged the cacti into the wall/background and dropped object count **15 → 11**. The exact failure mode it was meant to fix (legitimate boundaries) is where it does the most damage. |
| **Multi-scale context windows** | ❌ removed | Larger context crops blurred small objects and, on its own, produced output essentially identical to baseline — no measurable benefit for the cost. |
| **Hierarchical consistency** | ❌ removed | Part of the same "global context" experiment; reduced label diversity without a clear win. |
| → *Net:* the entire `context.py` experiment was reverted | ✅ clean baseline restored | Pipeline is back to sampler → upsample → CRF → objects. All 43 tests pass. |
| **C1 classifier on full scenes** | ⚠️ domain gap | Trained on close-up isolated material crops; on scene patches it's low-confidence and noisy (sky → foam/pottery/flower at 0.3–0.8; brick wall flips between pottery/metal/foliage). Produces fragmented masks (15 objects on `minc_1`). |
| **Apple MPS acceleration** | ❌ unusable | `torch_geometric` scatter ops are broken on MPS (`Placeholder storage not allocated`). Local runs default to **CPU**; GPU work goes to the VM (CUDA). |
| **`pydensecrf` on the VM** | ❌ won't build | Fails to compile on the VM's Python 3.13 (Cython/Eigen). VM runs use the **superpixel** CRF fallback; the Mac uses the real dense CRF. |
| **Segmentation-mask eval vs MINC** | ⛔ blocked | The MINC data in the repo is **MINC-2500 (patch classification)** — single-material crops, **no segmentation masks / no full scenes**. Mask-similarity-vs-MINC (and the SAM comparison) still needs a full-scene + GT-mask MINC subset that we don't have locally. |

---

## Key empirical observations

- **Window size has a U-curve.** Too large → blocky, too few regions; too small → noisy.
  Mid-range (~48 px window / 24 px stride) was the sweet spot for these scenes.
- **The classifier, not the merger, is the bottleneck.** The patch diagnostic shows the
  merging steps (upsample/CRF/CC) are sound; segmentation quality tracks classifier
  confidence/consistency.
- **C1 vs MINC classifier on `minc_1.jpg`:**

  | | C1 HGNN (Matador, 37 cls) | MINC HGNN (MINC, 23 cls) |
  |---|---|---|
  | objects | 15 (fragmented, detailed) | 3 (coherent, coarse) |
  | per-patch confidence | low (0.3–0.7), inconsistent | high (0.5–1.0), consistent |
  | behaviour | picks out cacti/plants but speckles gravel/wall | clean brick/foliage/sky regions, loses cactus-vs-gravel detail |

  Trade-off: C1's finer partition may score closer to SAM's many-small-mask style on
  class-agnostic metrics; the MINC model gives cleaner, more semantically coherent regions
  and is the only valid choice for MINC-label mIoU.

---

## Open / in-progress

- **MINC patch-classification eval** (`scripts/eval_minc_patches.py`, + `minc_to_matador.json`
  crosswalk): built to score either checkpoint on a balanced MINC-2500 test subset and
  compare them per material. **Not yet run on the GPU** (local smoke-test was interrupted);
  needs the MINC-2500 image tree (`data/external/minc/minc-2500`) present on the VM.
- **Cross-taxonomy caveat:** the MINC model scores in its native 23-class space; the C1
  model is scored via an *approximate* MINC→Matador crosswalk over the ~15 mappable classes
  (8 MINC categories — glass, mirror, other, painted, plastic, skin, sky, water — have no
  Matador equivalent and are excluded). The C1 number is therefore a rough lower bound.
- **Mask-similarity vs SAM / MINC segmentation:** still blocked on obtaining full-scene MINC
  images with ground-truth masks. `compare_masks.py` is ready once we have them.
- **CRF backend provenance:** metadata records the *configured* backend, not the branch that
  actually executed. A one-line improvement would log which backend really ran per image.

---

## Environment notes

- Local: conda env `cs231n-project` (Python 3.10), `pydensecrf` + `skimage` installed,
  runs on **CPU** (MPS unsupported). `~/.zshrc` patched to keep the conda `bin/` first on `PATH`.
- VM (AWS DL AMI): CUDA available; activate `/opt/pytorch`, `pip install` deps
  (`timm`, etc.); `pydensecrf` unavailable → superpixel CRF.
- `out/` is git-ignored (avoids committing large result artifacts).
