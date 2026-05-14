#!/usr/bin/env python3
"""
Build deterministic stratified train/val/test splits for Matador-C1.

Usage:
    python scripts/build_matador_c1_splits.py \
        --manifest data/processed/matador_c1/manifest.csv \
        --out-dir data/processed/matador_c1/splits

    python scripts/build_matador_c1_splits.py --smoke-test
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def _load_manifest(path: Path) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _warn_existing_split(rows: List[Dict]) -> bool:
    return any("split" in r for r in rows)


def _stratified_split(
    rows: List[Dict],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Dict[str, List[Dict]]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    random.seed(seed)

    # Group by c1_label
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_label[r["c1_label"]].append(r)

    train, val, test = [], [], []
    for label, group in sorted(by_label.items()):
        random.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Avoid off-by-one: assign remainder to test
        n_test = n - n_train - n_val

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    # Shuffle globally to avoid label blocks
    for split in (train, val, test):
        random.shuffle(split)

    return {"train": train, "val": val, "test": test}


def _validate_splits(
    splits: Dict[str, List[Dict]],
    original_rows: List[Dict],
) -> None:
    total = sum(len(s) for s in splits.values())
    assert total == len(original_rows), (
        f"Split total ({total}) != manifest rows ({len(original_rows)})"
    )

    # No overlap
    train_ids = {r["sample_id"] for r in splits["train"]}
    val_ids = {r["sample_id"] for r in splits["val"]}
    test_ids = {r["sample_id"] for r in splits["test"]}

    assert len(train_ids & val_ids) == 0, "Train/val overlap detected"
    assert len(train_ids & test_ids) == 0, "Train/test overlap detected"
    assert len(val_ids & test_ids) == 0, "Val/test overlap detected"

    # All 37 classes in train
    train_labels = {r["c1_label"] for r in splits["train"]}
    all_labels = {r["c1_label"] for r in original_rows}
    assert train_labels == all_labels, (
        f"Train missing classes: {sorted(all_labels - train_labels)}"
    )

    # Warn if val/test missing classes
    for split_name in ("val", "test"):
        split_labels = {r["c1_label"] for r in splits[split_name]}
        missing = all_labels - split_labels
        if missing:
            print(
                f"WARNING: {split_name} missing {len(missing)} class(es): {sorted(missing)}",
                file=sys.stderr,
            )


def _print_per_class_counts(splits: Dict[str, List[Dict]]) -> None:
    for name, rows in splits.items():
        counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r["c1_label"]] += 1
        print(f"\n{name.upper()} ({len(rows)} rows):")
        for label in sorted(counts):
            print(f"  {label:20s} {counts[label]:4d}")


def _write_csv(rows: List[Dict], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _build_summary(splits: Dict[str, List[Dict]], seed: int) -> Dict:
    summary = {"seed": seed, "total": sum(len(s) for s in splits.values())}
    for name, rows in splits.items():
        summary[name] = {
            "count": len(rows),
            "classes": sorted({r["c1_label"] for r in rows}),
            "class_counts": {
                label: sum(1 for r in rows if r["c1_label"] == label)
                for label in sorted({r["c1_label"] for r in rows})
            },
        }
    return summary


def build_splits(
    manifest_path: Path,
    out_dir: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 1337,
    use_existing_split: bool = False,
) -> None:
    rows = _load_manifest(manifest_path)

    if _warn_existing_split(rows) and not use_existing_split:
        print(
            "WARNING: manifest already contains a 'split' column. "
            "Use --use-existing-split to respect it, otherwise it will be ignored.",
            file=sys.stderr,
        )

    splits = _stratified_split(rows, train_frac, val_frac, test_frac, seed)

    # Add split column
    fieldnames = list(rows[0].keys())
    if "split" not in fieldnames:
        fieldnames.append("split")

    for name, split_rows in splits.items():
        for r in split_rows:
            r["split"] = name

    _validate_splits(splits, rows)
    _print_per_class_counts(splits)

    # Write CSVs
    for name in ("train", "val", "test"):
        _write_csv(splits[name], out_dir / f"{name}.csv", fieldnames)

    # Write summary JSON
    summary = _build_summary(splits, seed)
    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSplits written to {out_dir}")


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

def _run_smoke_test():
    import tempfile

    rows = []
    for label in ["alpha", "beta", "gamma"]:
        for i in range(10):
            rows.append({
                "sample_id": f"{label}_{i}",
                "image_path": f"img/{label}_{i}.tiff",
                "label_path": f"lbl/{label}_{i}.txt",
                "material_label": label,
                "c1_label": label,
            })

    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.csv"
        out_dir = Path(tmp) / "splits"

        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

        build_splits(manifest, out_dir, seed=1337)

        for name in ("train", "val", "test"):
            path = out_dir / f"{name}.csv"
            assert path.exists(), f"{name}.csv not created"

        summary_path = out_dir / "split_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["total"] == 30
        assert len(summary["train"]["classes"]) == 3

        print("\nSmoke test passed.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Build Matador-C1 train/val/test splits.")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--use-existing-split", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    if args.manifest is None or args.out_dir is None:
        parser.error("--manifest and --out-dir are required (unless --smoke-test)")

    build_splits(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        use_existing_split=args.use_existing_split,
    )


if __name__ == "__main__":
    main()
