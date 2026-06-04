#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-/opt/matseparate/repo}"
log="logs/long_hgnn_30ep_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log") 2>&1

configs=(
  "runs/patch_classifier_long_configs/long_hgnn_fixed_lr3e-04_wd1e-04_drop0.1_combined_cnn_average_30ep.yaml"
  "runs/patch_classifier_long_configs/long_hgnn_nodewise_lr3e-04_wd5e-04_drop0.1_combined_cnn_average_30ep.yaml"
)

for cfg in "${configs[@]}"; do
  echo "[$(date -Is)] START $cfg"
  python3 scripts/train_c1_hgnn.py --config "$cfg"
  echo "[$(date -Is)] DONE $cfg"
done

python3 scripts/plot_training_curves.py \
  runs/long_hgnn_fixed_lr3e-04_wd1e-04_drop0.1_combined_cnn_average_30ep \
  runs/long_hgnn_nodewise_lr3e-04_wd5e-04_drop0.1_combined_cnn_average_30ep \
  --summary-csv runs/patch_classifier_long_hgnn_30ep_summary.csv
