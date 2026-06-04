#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all_ablations.sh
#
# Sequential runner for all 5 ablation experiments.
# Skips any training variant that already has a checkpoint.
# Outputs a timestamped log file per step under runs/ablations_run/.
#
# Usage:
#   bash scripts/run_all_ablations.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PYTHON=/opt/pytorch/bin/python3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO/runs/ablations_run"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

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

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 & 2: Train graph ablation variants (skip if checkpoint exists)
# ─────────────────────────────────────────────────────────────────────────────

for VARIANT in mlp_head hgnn_ce random_tree full_graph; do
    CKPT="$REPO/runs/graph_ablations/$VARIANT/checkpoint_best.pt"
    CFG="$REPO/runs/graph_ablations/$VARIANT/config.json"
    # Only skip if BOTH checkpoint AND config.json exist (config written only on successful completion)
    if [ -f "$CKPT" ] && [ -f "$CFG" ]; then
        log "SKIP   train_$VARIANT (training complete)"
    else
        step "train_$VARIANT" \
            "$PYTHON" -u "$REPO/scripts/train_graph_ablations.py" \
                --variant "$VARIANT" \
                --epochs 5 --batch-size 32 --lr 1e-4 \
                --runs-dir runs/graph_ablations
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SAM eval on all graph ablation variants (ablations 1 + 2)
# ─────────────────────────────────────────────────────────────────────────────

step "eval_graph_ablations" \
    "$PYTHON" -u "$REPO/scripts/eval_graph_ablations.py" \
        --iou-threshold 0.5 \
        --out runs/graph_ablation_eval.json

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Ablation 3 — confusion matrices by parent category
# ─────────────────────────────────────────────────────────────────────────────

step "ablation3_confusion" \
    "$PYTHON" -u "$REPO/scripts/analyze_hgnn_ablations.py" --ablation 3

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Ablation 4 — context removal levels
# ─────────────────────────────────────────────────────────────────────────────

step "ablation4_context" \
    "$PYTHON" -u "$REPO/scripts/analyze_hgnn_ablations.py" --ablation 4

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Ablation 5 — bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

step "ablation5_bootstrap" \
    "$PYTHON" -u "$REPO/scripts/analyze_hgnn_ablations.py" --ablation 5

# ─────────────────────────────────────────────────────────────────────────────
log "ALL ABLATIONS COMPLETE"
log "Results:"
log "  Graph ablation eval: $REPO/runs/graph_ablation_eval.json"
log "  Confusion matrices:  $REPO/runs/ablation3_confusion.json"
log "  Context levels:      $REPO/runs/ablation4_context.json"
log "  Bootstrap CIs:       $REPO/runs/ablation5_bootstrap.json"
log "  Step logs:           $LOG_DIR/"
