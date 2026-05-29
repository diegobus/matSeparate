# HGNN Inference Guide for Segmentation Collaborators

## Completed Runs Summary

| Experiment | Config | Best Val Leaf Acc | Best Val Exact Path | Best Val Path F1 | Notes |
|---|---|---|---|---|---|
| **HGNN avg_init** `20260529_004313` | CNN-average prototypes, 10 epochs, batch 64, lr 1e-4 | **84.3%** (epoch 9) | **59.8%** | **91.1%** | Slight overfit by epoch 10; strongest overall |
| **HGNN rand_init** `20260528_233513` | Random prototypes, same hparams | **84.0%** (epoch 9) | **56.9%** | **90.8%** | Very close to avg_init; init matters but not dramatically |
| **ResNet50 flat** `20260528_231409` | Flat 37-class CE, batch 128, lr 1e-3 | **83.5%** (epoch 9) | N/A | N/A | Heavy overfitting (97.4% train -> 81.7% val by epoch 10) |

**Recommendation:** Use the HGNN `avg_init` checkpoint. It has the highest leaf accuracy and provides hierarchical path predictions (full 58-node taxonomy) that the flat ResNet cannot.

---

## Quick Start

### 1. Load a checkpoint

```python
from scripts.infer_api import HGNNInference

api = HGNNInference.from_run_dir(
    "runs/c1_hgnn_baseline/avg_init_20260529_004313",
    device="auto"
)
```

### 2. Single-image inference

```python
result = api.infer("photo.jpg", decode_path=True)

result["leaf_label"]   # "concrete"
result["leaf_probs"]   # (37,) softmax probabilities
result["path_nodes"]   # ["root", "solid", "abiotic", "ceramic", "structural", "concrete"]
result["node_probs"]   # (58,) sigmoid over all taxonomy nodes
```

### 3. Batch inference

```python
results = api.infer_batch(["img1.jpg", "img2.jpg"], decode_path=True)
```

---

## Important Notes

### ResNet50 checkpoints are NOT supported by this API

`scripts/infer_api.py` only loads `HGNN` checkpoints (it rebuilds the taxonomy graph and GNN layers). For the flat ResNet50 baseline, use a separate wrapper or simply use the HGNN since it is stronger and provides hierarchical outputs.

### For segmentation: aggregate probabilities, not hard labels

The API returns **per-patch leaf probabilities** (`leaf_probs`). For segmentation you should:

1. Slide a window over the full image, call `api.infer()` on each patch
2. Map the 224x224 patch prediction back to the image coordinate
3. **Average overlapping `leaf_probs`** before taking argmax (do not average hard labels)
4. Use `result["node_probs"]` if you want taxonomy-aware smoothing (e.g. penalize adjacent pixels with low taxonomy similarity)

---

## Deployment Checklist

Your collaborator only needs these three files from the run directory:

- `config.yaml`
- `node_index.json`
- `checkpoint_best.pt`

Everything else (taxonomy graph shape, transforms, node ordering) is reconstructed automatically from those three files.

Example tree for the `avg_init` run:

```
runs/c1_hgnn_baseline/avg_init_20260529_004313/
  config.yaml
  node_index.json
  checkpoint_best.pt
```

---

## CLI Sanity Check

```bash
python scripts/infer_api.py \
  --run-dir runs/c1_hgnn_baseline/avg_init_20260529_004313 \
  --image path/to/image.jpg
```
