#!/usr/bin/env bash
set -euo pipefail

bucket="${MATADOR_BUCKET:-s3://matseparate-context-dataset-251995574236-us-west-2}"
repo_dir="${REPO_DIR:-/opt/matseparate/repo}"
expected_context_size="${EXPECTED_CONTEXT_SIZE:-529514946560}"
poll_seconds="${POLL_SECONDS:-300}"

cd "$repo_dir"
mkdir -p logs data/downloads data/external/matador data/processed/matador_c1/generated_configs

log_path="logs/global_context_overnight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_path") 2>&1

echo "[$(date -Is)] global context overnight bootstrap"
echo "[$(date -Is)] repo=$repo_dir bucket=$bucket"

wait_for_s3_context() {
  while true; do
    size="$(
      aws s3api head-object \
        --bucket "${bucket#s3://}" \
        --key raw/matador.context.tar \
        --query ContentLength \
        --output text 2>/dev/null || true
    )"
    if [[ "$size" == "$expected_context_size" ]]; then
      echo "[$(date -Is)] S3 context archive is ready: $size bytes"
      return 0
    fi
    echo "[$(date -Is)] waiting for S3 context archive; saw '${size:-missing}', expected $expected_context_size"
    sleep "$poll_seconds"
  done
}

extract_context() {
  local marker="data/external/matador/.context_extract_complete"
  if [[ -f "$marker" ]] && [[ -d "data/external/matador/matador/context_img" ]]; then
    echo "[$(date -Is)] context already extracted"
    return 0
  fi
  echo "[$(date -Is)] streaming context archive from S3 into tar extraction"
  rm -f data/downloads/matador.context.tar
  aws s3 cp "$bucket/raw/matador.context.tar" - --only-show-errors \
    | tar -xf - -C data/external/matador
  touch "$marker"
  echo "[$(date -Is)] extracted context image count: $(find data/external/matador/matador/context_img -type f | wc -l)"
}

build_manifest() {
  echo "[$(date -Is)] building global manifest"
  python3 scripts/build_matador_c1_global_manifest.py \
    --manifest data/processed/matador_c1/manifest.csv \
    --context-root data/external/matador \
    --out data/processed/matador_c1/global_manifest.csv
}

wait_for_training_slot() {
  while pgrep -f "scripts/train_c1_hgnn.py|scripts/train_c1_resnet.py|scripts/train_c1_global_resnet.py" >/dev/null; do
    echo "[$(date -Is)] waiting for active classifier training to finish before global-context runs"
    pgrep -af "scripts/train_c1_hgnn.py|scripts/train_c1_resnet.py|scripts/train_c1_global_resnet.py" || true
    sleep "$poll_seconds"
  done
}

write_config() {
  local out="$1"
  local experiment_name="$2"
  local lr="$3"
  local wd="$4"
  local dropout="$5"
  local label_smoothing="$6"
  local local_checkpoint="$7"
  python3 - "$out" "$experiment_name" "$lr" "$wd" "$dropout" "$label_smoothing" "$local_checkpoint" <<'PY'
import sys
from pathlib import Path
import yaml

out, experiment_name, lr, wd, dropout, label_smoothing, local_checkpoint = sys.argv[1:]
cfg = yaml.safe_load(Path("configs/experiments/c1_global_resnet50.yaml").read_text())
cfg["experiment_name"] = experiment_name
cfg["model"]["dropout"] = float(dropout)
cfg["model"]["local_checkpoint"] = local_checkpoint if local_checkpoint != "null" else None
cfg["training"]["learning_rate"] = float(lr)
cfg["training"]["weight_decay"] = float(wd)
cfg["training"]["label_smoothing"] = float(label_smoothing)
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

run_training_queue() {
  local best_local="runs/grid_resnet50_lr3e-04_wd1e-04_drop0.0_ls0.05/20260604_050732/checkpoint_best.pt"
  if [[ ! -f "$best_local" ]]; then
    echo "[$(date -Is)] WARNING: best local checkpoint not found, using ImageNet init only"
    best_local="null"
  fi

  local cfg_dir="data/processed/matador_c1/generated_configs"
  write_config "$cfg_dir/global_resnet_warm_lr3e-4_wd1e-4_ls05.yaml" \
    "global_resnet_warm_lr3e-4_wd1e-4_ls05" "3.0e-4" "1.0e-4" "0.1" "0.05" "$best_local"
  write_config "$cfg_dir/global_resnet_warm_lr1e-4_wd5e-4_ls05.yaml" \
    "global_resnet_warm_lr1e-4_wd5e-4_ls05" "1.0e-4" "5.0e-4" "0.1" "0.05" "$best_local"

  echo "[$(date -Is)] dry run"
  python3 scripts/train_c1_global_resnet.py \
    --config "$cfg_dir/global_resnet_warm_lr3e-4_wd1e-4_ls05.yaml" \
    --dry-run \
    --num-workers 2

  for cfg in "$cfg_dir"/global_resnet_warm_*.yaml; do
    echo "[$(date -Is)] starting training config: $cfg"
    python3 scripts/train_c1_global_resnet.py --config "$cfg" --num-workers 4
    echo "[$(date -Is)] finished training config: $cfg"
  done
}

wait_for_s3_context
extract_context
build_manifest
wait_for_training_slot
run_training_queue

echo "[$(date -Is)] global context overnight bootstrap complete"
