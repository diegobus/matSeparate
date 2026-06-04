# MatSeparate HGNN Quickstart Guide

One-shot setup for training the Matador-C1 HGNN on a fresh machine (VM, cloud instance, or new laptop).

---

## 1. What's in the repo vs. what you need

**Already in git (clone and go):**
- `data/processed/matador/manifest.csv` — full Matador manifest
- `data/processed/matador_c1/manifest.csv` — C1-filtered manifest
- `data/processed/matador_c1/splits/{train,val,test}.csv` — deterministic splits
- `data/processed/matador_c1/node_index.json` — canonical 58-node ordering
- `taxonomy/assets/matador-c1-taxonomy.json` — C1 taxonomy tree
- All source code (`scripts/`, `gnn_classifier/`, `datasets/`, `configs/`)

**Not in git (must download or generate):**
| Artifact | Size | How to get it |
|----------|------|---------------|
| `data/downloads/matador.appearance.tar` | ~11 GB | Download from Matador release |
| `data/downloads/matador.label.tar` | ~7 MB | Download from Matador release |
| `data/external/matador/` (extracted images) | ~11 GB | Extract from appearance tar (optional but recommended for speed) |
| `data/processed/matador_c1/prototypes/hgnn_prototypes.pt` | ~9 MB | Generate with `init_hgnn_prototypes.py` or `scp` from existing machine |

---

## 2. Environment setup

```bash
# 1. Clone repo
git clone <repo-url> matSeparate && cd matSeparate

# 2. Create virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt** includes:
- `torch>=2.6.0`
- `torchvision>=0.21.0`
- `torch-geometric>=2.6.1`
- `timm>=1.0.15`
- `networkx>=3.4`, `numpy>=2.2`, `pillow>=11.0`, `tifffile>=2025.2`, `scikit-image>=0.25`, `matplotlib>=3.10`, `pandas>=2.2`

> **CUDA note:** If your VM has CUDA, install the appropriate PyTorch wheels before `pip install -r requirements.txt` to avoid the CPU-only default:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> pip install -r requirements.txt
> ```

---

## 3. Download raw Matador data

Download the two tarballs from the Matador dataset release and place them in `data/downloads/`:

```bash
mkdir -p data/downloads
# Replace with actual download URLs
wget -O data/downloads/matador.appearance.tar https://fpcv.cs.columbia.edu/static/cvclass/downloads/matador.appearance.tar
wget -O data/downloads/matador.label.tar https://fpcv.cs.columbia.edu/static/cvclass/downloads/matador.label.tar
```

Verify integrity:

```bash
python scripts/inspect_matador_release.py --tar data/downloads/matador.appearance.tar
python scripts/inspect_matador_release.py --tar data/downloads/matador.label.tar
```

---

## 4. (Optional but recommended) Extract images for faster I/O

Training from tar is slower due to on-the-fly extraction. Extract once:

```bash
mkdir -p data/external/matador
tar -xf data/downloads/matador.appearance.tar -C data/external/matador
```

This creates `data/external/matador/matador/appearance/` with all images.

Update `configs/experiments/c1_hgnn_baseline.yaml` to use extracted images instead of tar:

```yaml
data:
  extracted_root: data/external/matador
  appearance_tar: null
```

If you skip extraction, keep:

```yaml
data:
  extracted_root: null
  appearance_tar: data/downloads/matador.appearance.tar
```

---

## 5. Smoke-test the dataset

Before training, verify everything loads correctly:

```bash
python scripts/smoke_test_matador_dataset.py
```

You should see:
- `Dataset length: ~4800`
- `Num taxonomy nodes: 58`
- 4 sample inspections with matching multihot sums
- Batch image shape: `[4, 3, 224, 224]`

---

## 6. Generate prototypes

The HGNN requires prototype embeddings for each taxonomy node. Generate them from the training set:

```bash
python scripts/init_hgnn_prototypes.py \
    --config configs/experiments/hgnn_prototype_init.yaml \
    --mode cnn_average
```

This:
1. Loads the C1 training split
2. Extracts ResNet50 features for every training image
3. Averages features per taxonomy node
4. Saves `data/processed/matador_c1/prototypes/hgnn_prototypes.pt` (~9 MB)

> **Speed tip:** On a VM with many CPU cores, increase `num_workers` in the config or add `--batch-size 128`.

> **Copying instead of generating:** If you already have `hgnn_prototypes.pt` on another machine, just `scp` it to `data/processed/matador_c1/prototypes/`.

---

## 7. Train the HGNN

### Dry run (recommended first step)

```bash
python scripts/train_c1_hgnn.py \
    --config configs/experiments/c1_hgnn_baseline.yaml \
    --dry-run
```

Expected output:
```
Device: cuda
Taxonomy nodes: 58
Hierarchy levels: 6
Leaf nodes: 37
Graph: patched bidirectional global edges (172 → 230 edges)
  Sync: loaded artifact projection into ImageEncoder.classifier
  Sync: set HGNN.projection to identity
Prototype init: loaded cnn_average from data/processed/matador_c1/prototypes/hgnn_prototypes.pt
--- DRY RUN ---
  Val loss: ~2.7
  leaf_acc=0.0000 node_acc=0.47 exact=0.0000 f1=0.17 hier_acc=0.50
Dry run complete. Exiting.
```

### Full training

```bash
python scripts/train_c1_hgnn.py \
    --config configs/experiments/c1_hgnn_baseline.yaml \
    --epochs 5 \
    --batch-size 32
```

The script will:
- Save checkpoints to `runs/c1_hgnn_baseline/<timestamp>/checkpoint_best.pt`
- Save metrics to `runs/c1_hgnn_baseline/<timestamp>/metrics.json`
- Save config + node index to the same run directory

### CLI overrides

Override any config value without editing YAML:

```bash
# Fewer epochs for a quick experiment
python scripts/train_c1_hgnn.py --config configs/experiments/c1_hgnn_baseline.yaml --epochs 2

# Larger batch size on a GPU with more VRAM
python scripts/train_c1_hgnn.py --config configs/experiments/c1_hgnn_baseline.yaml --batch-size 64

# Use random prototypes instead of cnn_average
python scripts/train_c1_hgnn.py --config configs/experiments/c1_hgnn_baseline.yaml --prototype-init random

# Force CPU (for debugging)
python scripts/train_c1_hgnn.py --config configs/experiments/c1_hgnn_baseline.yaml --device cpu

# Warm-start CNN from a flat ResNet checkpoint
python scripts/train_c1_hgnn.py \
    --config configs/experiments/c1_hgnn_baseline.yaml \
    --epochs 10 \
    --batch-size 64
# ^ edit config: model.cnn_checkpoint: runs/c1_resnet50_baseline/.../checkpoint_best.pt
```

---

## 8. Evaluate a checkpoint

```bash
python scripts/eval_c1_hgnn.py \
    --run-dir runs/c1_hgnn_baseline/20260528_141412 \
    --split test
```

Replace `20260528_141412` with your actual run directory.

---

## 9. Ablation checklist

Toggle features in `configs/experiments/c1_hgnn_baseline.yaml`:

| Config key | Default | Effect |
|------------|---------|--------|
| `prototypes.sync_projection` | `true` | Load artifact projection into CNN classifier; set HGNN projection to identity |
| `prototypes.allow_projection_mismatch` | `false` | Allow training even if prototype artifact lacks projection weights |
| `graph.bidirectional_global_edges` | `true` | Add taxonomy→global reverse edges for richer message passing |
| `model.skip_connection` | `true` | GNN residual connections |
| `model.cnn_checkpoint` | `null` | Path to flat ResNet checkpoint for warm-start |

Example: disable projection sync and bidirectional edges for an ablation baseline:

```yaml
prototypes:
  sync_projection: false
graph:
  bidirectional_global_edges: false
```

---

## 10. Expected metrics (2-epoch sanity check)

With `sync_projection=true`, `bidirectional_global_edges=true`, and `cnn_average` prototypes on MPS:

| Epoch | val_loss | val_leaf_acc | val_node_acc | val_exact_match | val_path_f1 | val_hier_acc |
|-------|----------|--------------|--------------|-----------------|-------------|--------------|
| 1 | 1.06 | 0.59 | 0.81 | 0.00 | 0.51 | 0.82 |
| 2 | 0.65 | 0.73 | 0.92 | 0.06 | 0.70 | 0.87 |

If your numbers are wildly different (e.g., val_leaf_acc < 0.10), check:
1. Prototype artifact exists and was generated with the same `node_index.json`
2. Images are loading correctly (run smoke test)
3. Device is CUDA not CPU (check `Device: cuda` in logs)

---

## 11. Directory reference

```
matSeparate/
├── configs/experiments/          # YAML configs for all experiments
│   ├── c1_hgnn_baseline.yaml     # Main HGNN training config
│   ├── c1_resnet50_baseline.yaml # Flat ResNet baseline config
│   └── hgnn_prototype_init.yaml  # Prototype generation config
├── data/
│   ├── downloads/                # Raw tarballs (gitignored)
│   ├── external/matador/         # Extracted images (gitignored)
│   └── processed/
│       ├── matador/manifest.csv  # Full manifest (in git)
│       └── matador_c1/           # C1 processed data (mostly in git)
│           ├── manifest.csv
│           ├── node_index.json
│           ├── splits/
│           └── prototypes/       # Generated, not in git
├── datasets/
│   └── matador.py                # MatadorC1Dataset
├── gnn_classifier/
│   ├── hgnn.py                   # HGNN model
│   └── loss.py                   # Greedy hierarchical loss
├── scripts/
│   ├── train_c1_hgnn.py          # Main training script
│   ├── eval_c1_hgnn.py           # Evaluation script
│   ├── init_hgnn_prototypes.py   # Prototype generation
│   ├── build_matador_c1_manifest.py
│   ├── build_matador_c1_splits.py
│   ├── build_matador_c1_taxonomy.py
│   └── smoke_test_matador_dataset.py
├── taxonomy/
│   └── assets/
│       ├── matador-c1-taxonomy.json
│       └── taxonomy-tree.json
├── tests/
│   └── test_hgnn_metrics.py      # Metric smoke tests
└── docs/
    └── QUICKSTART.md             # This file
```

---

## Troubleshooting

### `RuntimeError: Passed CPU tensor to MPS op`
This was fixed by creating device-local tensors in `_patch_bidirectional_global_edges`. If you see it elsewhere, check that all tensors created in the script are on the same device as the model.

### `Artifact missing projection_state_dict`
Your prototype artifact was generated before projection syncing was added. Regenerate it:
```bash
python scripts/init_hgnn_prototypes.py --config configs/experiments/hgnn_prototype_init.yaml --mode cnn_average
```
Or set `prototypes.allow_projection_mismatch: true` in the config.

### `Expected 37 leaf nodes, got X`
The taxonomy graph changed. Re-run `build_matador_c1_taxonomy.py` and verify `node_index.json` is regenerated.

### Training is very slow
- Extract images from tar (step 4)
- Increase `num_workers` in config (e.g., 8 for a VM with many cores)
- Use a larger batch size if GPU memory allows
- Ensure you're using CUDA: check `Device: cuda` in logs
