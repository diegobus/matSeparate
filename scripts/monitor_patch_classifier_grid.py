#!/usr/bin/env python3
"""Print compact status for the patch-classifier grid run."""

from __future__ import annotations

import argparse
import glob
import re
import time
from datetime import datetime, timedelta
from pathlib import Path


EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+).*(?:val_acc|leaf_acc)=([0-9.]+)")
RUN_RE = re.compile(r"\[(.*?)\]\s+RUN\s+(.+)")
RUN_DIR_RE = re.compile(r"Run directory:\s+(.+)")
BEST_RE = re.compile(r"Training complete\.\s+Best val (?:acc|loss):\s+([0-9.]+)")


def _read(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def _latest_log(pattern: str) -> Path | None:
    paths = [Path(p) for p in glob.glob(pattern)]
    paths = [p for p in paths if p.is_file()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def _parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _tensorboard_last(run_dir: Path) -> tuple[int | None, dict[str, float]]:
    tb_dir = run_dir / "tensorboard"
    if not tb_dir.exists():
        return None, {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return None, {}

    try:
        ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
    except Exception:
        return None, {}

    wanted = [
        "Loss/val",
        "Accuracy/val",
        "Accuracy/leaf_val",
        "Accuracy/hier_val",
        "Accuracy/path_f1_val",
        "Accuracy/exact_val",
    ]
    values: dict[str, float] = {}
    last_step = None
    for tag in wanted:
        if tag not in tags:
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        ev = events[-1]
        values[tag] = float(ev.value)
        last_step = max(last_step or ev.step, ev.step)
    return last_step, values


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    return str(timedelta(seconds=int(seconds)))


def build_status(args: argparse.Namespace) -> str:
    log = _latest_log(args.log_glob)
    if log is None:
        return "No grid log found."

    text = _read(log)
    lines = text.splitlines()
    prepared = None
    for line in lines:
        if line.startswith("Prepared "):
            try:
                prepared = int(line.split()[1])
            except Exception:
                pass
            break

    run_matches = list(RUN_RE.finditer(text))
    complete_count = len(BEST_RE.findall(text))
    current_cmd = run_matches[-1].group(2) if run_matches else "unknown"
    current_started = _parse_datetime(run_matches[-1].group(1)) if run_matches else None
    current_elapsed = (datetime.now() - current_started).total_seconds() if current_started else None

    run_dirs = list(RUN_DIR_RE.finditer(text))
    run_dir = Path(run_dirs[-1].group(1).strip()) if run_dirs else None

    epoch = None
    total_epochs = args.epochs
    metrics: dict[str, float] = {}
    if run_dir:
        tb_epoch, tb_metrics = _tensorboard_last(run_dir)
        if tb_epoch is not None:
            epoch = tb_epoch
            metrics = tb_metrics

    if epoch is None:
        for line in reversed(lines):
            match = EPOCH_RE.search(line)
            if match:
                epoch = int(match.group(1))
                total_epochs = int(match.group(2))
                metrics["last_val_metric"] = float(match.group(3))
                break

    remaining_current = None
    if current_elapsed is not None and epoch and epoch > 0:
        sec_per_epoch = current_elapsed / epoch
        remaining_current = max(total_epochs - epoch, 0) * sec_per_epoch

    total = prepared or len(run_matches) or args.total_jobs
    jobs_remaining_after_current = max(total - complete_count - 1, 0)
    eta = None
    if remaining_current is not None:
        eta = remaining_current + jobs_remaining_after_current * args.minutes_per_job * 60

    metric_parts = []
    for key, value in metrics.items():
        metric_parts.append(f"{key}={value:.4f}")

    return "\n".join(
        [
            f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"log: {log}",
            f"jobs: completed {complete_count}/{total}",
            f"current: {Path(current_cmd.split()[-1]).name if current_cmd else 'unknown'}",
            f"run_dir: {run_dir or 'unknown'}",
            f"epoch: {epoch or '?'} / {total_epochs}",
            f"elapsed current job: {_fmt_seconds(current_elapsed)}",
            f"eta current job: {_fmt_seconds(remaining_current)}",
            f"eta grid: {_fmt_seconds(eta)}",
            "metrics: " + (", ".join(metric_parts) if metric_parts else "not available yet"),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-glob", default="runs/logs/patch_classifier_grid_*.log")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--total-jobs", type=int, default=12)
    parser.add_argument("--minutes-per-job", type=float, default=12.5)
    parser.add_argument("--watch", type=int, default=0, help="repeat every N seconds")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    while True:
        status = build_status(args)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(status + "\n")
        print(status, flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
