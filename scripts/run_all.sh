#!/usr/bin/env bash
# =============================================================================
# run_all.sh  —  Full training + evaluation pipeline for SAM + HGNN material
#                segmentation experiments.
#
# Runs in order:
#   1. Train flat ResNet50 baseline
#   2. Train flat + hierarchical loss baseline
#   3. Train MaskDropAugment baseline
#   4. Train HGNN (main model)
#   5. Train graph-structure ablation variants
#   6. Evaluate all classifiers on MINC-S segments (Sections 1 & 2)
#   7. Run all five HGNN ablation studies
#
# Each training step is skipped if a completed checkpoint already exists.
# Results are written to runs/ and logged under runs/logs/.
#
# Prerequisites:
#   - data/external/minc/minc-2500/  (MINC-2500 patch dataset)
#   - data/external/minc/minc-s/     (MINC-S scene photos + GT segments)
#   - out/sam_eval_*/                (pre-computed SAM masks + segment CSV)
#
# Usage:
#   bash scripts/run_all.sh
#   bash scripts/run_all.sh --epochs 10   # override default epoch count
# =============================================================================

set -euo pipefail

PYTHON=/opt/pytorch/bin/python3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO/runs/logs"
mkdir -p "$LOG_DIR"

EPOCHS=${1:-10}   # default 10 epochs for main models; ablation variants use 5

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

step() {
    local name="$1"; shift
    local logfile="$LOG_DIR/${name}.log"
    log "START  $name"
    if "$@" 2>&1 | tee "$logfile"; then
        log "DONE   $name"
    else
        die "$name failed — see $logfile"
    fi
}

# Check if a training run is complete (has both checkpoint and config)
is_trained() {
    local run_dir="$1"
    local ckpt config
    # Find the most recent run subdirectory
    ckpt=$(find "$run_dir" -name "checkpoint_best.pt" 2>/dev/null | sort | tail -1)
    config=$(find "$run_dir" -name "config.json" 2>/dev/null | sort | tail -1)
    [ -n "$ckpt" ] && [ -n "$config" ]
}

# =============================================================================
# 1–3: Baseline classifiers
# =============================================================================

for MODEL in flat hierloss maskdrop; do
    RUN_DIR="$REPO/runs/minc_$MODEL"
    if is_trained "$RUN_DIR"; then
        log "SKIP   train_$MODEL (checkpoint exists)"
    else
        step "train_$MODEL" \
            "$PYTHON" -u "$REPO/scripts/train_classifier.py" \
                --model "$MODEL" \
                --epochs "$EPOCHS" \
                --batch-size 64 \
                --runs-dir "runs/minc_$MODEL"
    fi
done

# =============================================================================
# 4: HGNN (main model)
# =============================================================================

HGNN_DIR="$REPO/runs/minc_hgnn"
if is_trained "$HGNN_DIR"; then
    log "SKIP   train_hgnn (checkpoint exists)"
else
    step "train_hgnn" \
        "$PYTHON" -u "$REPO/scripts/train_hgnn.py" \
            --epochs "$EPOCHS" \
            --batch-size 32 \
            --runs-dir runs/minc_hgnn
fi

# =============================================================================
# 5: Graph-structure ablation variants (5 epochs each, fixed budget for comparison)
# =============================================================================

for VARIANT in mlp_head hgnn_ce random_tree full_graph; do
    CKPT="$REPO/runs/graph_ablations/$VARIANT/checkpoint_best.pt"
    CFG="$REPO/runs/graph_ablations/$VARIANT/config.json"
    if [ -f "$CKPT" ] && [ -f "$CFG" ]; then
        log "SKIP   train_ablation_$VARIANT (complete)"
    else
        step "train_ablation_$VARIANT" \
            "$PYTHON" -u "$REPO/scripts/train_ablation_variants.py" \
                --variant "$VARIANT" \
                --epochs 5 \
                --batch-size 32 \
                --runs-dir runs/graph_ablations
    fi
done

# =============================================================================
# 6: Evaluate all classifiers on MINC-S
# =============================================================================

step "eval_classifiers" \
    "$PYTHON" -u "$REPO/scripts/eval_classifiers.py" \
        --section all \
        --out runs/eval_classifiers.json

# =============================================================================
# 7: Ablation studies
# =============================================================================

step "eval_ablations" \
    "$PYTHON" -u "$REPO/scripts/eval_ablations.py" \
        --ablation 1 2 3 4 5 \
        --out runs/eval_ablations.json

# =============================================================================
log "ALL EXPERIMENTS COMPLETE"
log "Results:"
log "  Classifier eval:  $REPO/runs/eval_classifiers.json"
log "  Ablation results: $REPO/runs/eval_ablations.json"
log "  Step logs:        $LOG_DIR/"
