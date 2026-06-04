#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

wait_log="logs/global_resnet_corrected_wait_$(date +%Y%m%d_%H%M%S).log"
train_log="logs/global_resnet_corrected_$(date +%Y%m%d_%H%M%S).log"

{
  echo "Started wait loop at $(date)"
  echo "Waiting for active global HGNN training to finish..."
} | tee -a "$wait_log"

while pgrep -f "scripts/train_c1_global_hgnn.py" >/dev/null; do
  {
    echo "$(date): HGNN still running"
    pgrep -af "scripts/train_c1_global_hgnn.py" || true
  } >> "$wait_log"
  sleep 300
done

{
  echo "$(date): HGNN no longer running"
  echo "$(date): launching corrected global ResNet"
  echo "Train log: $train_log"
} | tee -a "$wait_log"

python3 scripts/train_c1_global_resnet.py \
  --config configs/experiments/c1_global_resnet50_corrected.yaml \
  > "$train_log" 2>&1

{
  echo "$(date): corrected global ResNet finished"
  tail -20 "$train_log"
} | tee -a "$wait_log"
