#!/usr/bin/env python3
"""Run a focused patch-classifier hyperparameter grid on Matador-C1."""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _set_common_training(config, args):
    config["training"]["num_epochs"] = args.epochs
    config["training"]["num_workers"] = args.num_workers
    config["training"]["device"] = args.device
    config.setdefault("logging", {})["log_interval"] = 1


def _resnet_jobs(args):
    base = _load_yaml(args.resnet_config)
    jobs = []
    grid = [
        {"lr": 1.0e-3, "wd": 1.0e-4, "drop": 0.0, "smooth": 0.0},
        {"lr": 3.0e-4, "wd": 1.0e-4, "drop": 0.0, "smooth": 0.0},
        {"lr": 1.0e-3, "wd": 5.0e-4, "drop": 0.0, "smooth": 0.0},
        {"lr": 3.0e-4, "wd": 5.0e-4, "drop": 0.0, "smooth": 0.0},
        {"lr": 3.0e-4, "wd": 1.0e-4, "drop": 0.2, "smooth": 0.0},
        {"lr": 3.0e-4, "wd": 1.0e-4, "drop": 0.0, "smooth": 0.05},
    ]
    for item in grid:
        cfg = copy.deepcopy(base)
        _set_common_training(cfg, args)
        cfg["experiment_name"] = (
            f"grid_resnet50_lr{item['lr']:.0e}_wd{item['wd']:.0e}"
            f"_drop{item['drop']}_ls{item['smooth']}"
        )
        cfg["training"]["batch_size"] = args.resnet_batch_size
        cfg["training"]["learning_rate"] = item["lr"]
        cfg["training"]["weight_decay"] = item["wd"]
        cfg["training"]["label_smoothing"] = item["smooth"]
        cfg["model"]["drop_rate"] = item["drop"]
        path = args.config_dir / f"{cfg['experiment_name']}.yaml"
        jobs.append(("resnet", cfg, path, [sys.executable, "scripts/train_c1_resnet.py", "--config", str(path)]))
    return jobs


def _hgnn_jobs(args):
    base = _load_yaml(args.hgnn_config)
    jobs = []
    grid = [
        {"lr": 1.0e-4, "wd": 5.0e-4, "drop": 0.1, "loss": "combined", "init": "cnn_average", "head": "fixed_global_pool"},
        {"lr": 3.0e-4, "wd": 5.0e-4, "drop": 0.1, "loss": "combined", "init": "cnn_average", "head": "fixed_global_pool"},
        {"lr": 1.0e-4, "wd": 1.0e-4, "drop": 0.1, "loss": "combined", "init": "cnn_average", "head": "fixed_global_pool"},
        {"lr": 3.0e-4, "wd": 1.0e-4, "drop": 0.1, "loss": "combined", "init": "cnn_average", "head": "fixed_global_pool"},
        {"lr": 1.0e-4, "wd": 5.0e-4, "drop": 0.2, "loss": "combined", "init": "cnn_average", "head": "fixed_global_pool"},
        {"lr": 1.0e-4, "wd": 5.0e-4, "drop": 0.1, "loss": "max", "init": "cnn_average", "head": "fixed_global_pool"},
    ]
    nodewise_grid = [
        {"lr": 1.0e-4, "wd": 5.0e-4, "drop": 0.1, "loss": "combined", "init": "cnn_average", "head": "nodewise_shared"},
        {"lr": 3.0e-4, "wd": 5.0e-4, "drop": 0.1, "loss": "combined", "init": "cnn_average", "head": "nodewise_shared"},
        {"lr": 1.0e-4, "wd": 5.0e-4, "drop": 0.1, "loss": "max", "init": "cnn_average", "head": "nodewise_shared"},
    ]
    if args.models == "hgnn_nodewise":
        grid = nodewise_grid
    elif "hgnn_nodewise" in {item.strip() for item in args.models.split(",")}:
        grid.extend(nodewise_grid)
    for item in grid:
        cfg = copy.deepcopy(base)
        _set_common_training(cfg, args)
        head_label = "nodewise" if item["head"] == "nodewise_shared" else "fixed"
        cfg["experiment_name"] = (
            f"grid_hgnn_{head_label}_lr{item['lr']:.0e}_wd{item['wd']:.0e}"
            f"_drop{item['drop']}_{item['loss']}_{item['init']}"
        )
        cfg["training"]["batch_size"] = args.hgnn_batch_size
        cfg["training"]["learning_rate"] = item["lr"]
        cfg["training"]["weight_decay"] = item["wd"]
        cfg["training"]["loss_mode"] = item["loss"]
        cfg["model"]["dropout"] = item["drop"]
        cfg["model"]["head_type"] = item["head"]
        cfg["prototypes"]["init"] = item["init"]
        path = args.config_dir / f"{cfg['experiment_name']}.yaml"
        jobs.append(("hgnn", cfg, path, [sys.executable, "scripts/train_c1_hgnn.py", "--config", str(path)]))
    return jobs


def _latest_run_dir(experiment_name: str, runs_dir: Path):
    root = runs_dir / experiment_name
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="resnet,hgnn", help="Comma-separated: resnet,hgnn,hgnn_nodewise")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resnet-batch-size", type=int, default=128)
    parser.add_argument("--hgnn-batch-size", type=int, default=64)
    parser.add_argument("--config-dir", type=Path, default=Path("runs/patch_classifier_grid_configs"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--resnet-config", type=Path, default=Path("configs/experiments/c1_resnet50_baseline.yaml"))
    parser.add_argument("--hgnn-config", type=Path, default=Path("configs/experiments/c1_hgnn_baseline.yaml"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    jobs = []
    if "resnet" in selected:
        jobs.extend(_resnet_jobs(args))
    if "hgnn" in selected:
        jobs.extend(_hgnn_jobs(args))
    elif "hgnn_nodewise" in selected:
        jobs.extend(_hgnn_jobs(args))

    manifest = []
    for model_name, cfg, path, cmd in jobs:
        cfg["logging"]["runs_dir"] = str(args.runs_dir)
        _write_yaml(path, cfg)
        manifest.append({"model": model_name, "experiment_name": cfg["experiment_name"], "config": str(path), "cmd": cmd})

    args.config_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared {len(jobs)} jobs")
    for item in manifest:
        print(" ".join(item["cmd"]))
    if args.plan_only:
        return

    completed = []
    for _, cfg, _, cmd in jobs:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] RUN {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        run_dir = _latest_run_dir(cfg["experiment_name"], args.runs_dir)
        if run_dir:
            completed.append(str(run_dir))

    if completed:
        plot_cmd = [sys.executable, "scripts/plot_training_curves.py", *completed, "--summary-csv", str(args.runs_dir / "patch_classifier_grid_summary.csv")]
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] RUN {' '.join(plot_cmd)}", flush=True)
        subprocess.run(plot_cmd, check=True)


if __name__ == "__main__":
    main()
