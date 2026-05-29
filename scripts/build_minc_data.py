#!/usr/bin/env python3
"""
Build manifest and split CSVs for MINC-2500.

Outputs to data/processed/minc/:
  manifest.csv          -- all 57500 images
  splits/train.csv      -- from labels/train1.txt
  splits/val.csv        -- from labels/validate1.txt
  splits/test.csv       -- from labels/test1.txt

Usage:
    python scripts/build_minc_data.py \
        --minc-root data/external/minc/minc-2500 \
        --out-dir data/processed/minc
"""

import argparse
import csv
import json
from pathlib import Path

CATEGORIES = [
    "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass",
    "hair", "leather", "metal", "mirror", "other", "painted", "paper",
    "plastic", "polishedstone", "skin", "sky", "stone", "tile",
    "wallpaper", "water", "wood",
]

EXPECTED_PER_CLASS = 2500
EXPECTED_TRAIN = 48875
EXPECTED_VAL = 2875
EXPECTED_TEST = 5750


def build_manifest(minc_root: Path, out_dir: Path) -> Path:
    rows = []
    for cat in CATEGORIES:
        img_dir = minc_root / "images" / cat
        files = sorted(img_dir.glob("*.jpg"))
        for i, p in enumerate(files):
            rows.append({
                "sample_id": f"minc_{cat}_{i:04d}",
                "image_path": f"images/{cat}/{p.name}",
                "material_label": cat,
                "c1_label": cat,
            })

    out_path = out_dir / "manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "material_label", "c1_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest: {len(rows)} rows -> {out_path}")
    return out_path


def build_splits(minc_root: Path, manifest_csv: Path, out_dir: Path, fold: int = 1):
    # Build image_path -> sample_id lookup
    path_to_id = {}
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            path_to_id[row["image_path"]] = row

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": f"train{fold}.txt",
        "val": f"validate{fold}.txt",
        "test": f"test{fold}.txt",
    }

    for split_name, label_file in split_map.items():
        label_path = minc_root / "labels" / label_file
        rows = []
        with open(label_path) as f:
            for line in f:
                img_path = line.strip()
                if not img_path:
                    continue
                if img_path not in path_to_id:
                    raise KeyError(f"Image path not in manifest: {img_path}")
                rows.append(path_to_id[img_path])

        out_path = splits_dir / f"{split_name}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "material_label", "c1_label"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {split_name}: {len(rows)} rows -> {out_path}")

    return splits_dir


def validate(manifest_csv: Path, splits_dir: Path):
    with open(manifest_csv) as f:
        manifest_rows = list(csv.DictReader(f))
    assert len(manifest_rows) == len(CATEGORIES) * EXPECTED_PER_CLASS, (
        f"Expected {len(CATEGORIES) * EXPECTED_PER_CLASS} rows, got {len(manifest_rows)}"
    )

    split_ids = {}
    expected = {"train": EXPECTED_TRAIN, "val": EXPECTED_VAL, "test": EXPECTED_TEST}
    for split_name, exp_count in expected.items():
        with open(splits_dir / f"{split_name}.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == exp_count, f"{split_name}: expected {exp_count}, got {len(rows)}"
        split_ids[split_name] = {r["sample_id"] for r in rows}

    # No overlap between splits
    for a in ["train", "val", "test"]:
        for b in ["train", "val", "test"]:
            if a >= b:
                continue
            overlap = split_ids[a] & split_ids[b]
            assert not overlap, f"Overlap between {a} and {b}: {len(overlap)} samples"

    total_split = sum(len(v) for v in split_ids.values())
    assert total_split == len(manifest_rows), (
        f"Split total {total_split} != manifest total {len(manifest_rows)}"
    )
    print(f"Validation passed: {len(manifest_rows)} total, splits sum to {total_split}, no overlap.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minc-root", default="data/external/minc/minc-2500")
    parser.add_argument("--out-dir", default="data/processed/minc")
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args()

    minc_root = Path(args.minc_root)
    out_dir = Path(args.out_dir)

    print("Building manifest...")
    manifest_csv = build_manifest(minc_root, out_dir)

    print("Building splits...")
    splits_dir = build_splits(minc_root, manifest_csv, out_dir, fold=args.fold)

    print("Validating...")
    validate(manifest_csv, splits_dir)


if __name__ == "__main__":
    main()
