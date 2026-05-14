#!/usr/bin/env python3
"""
Build Matador-C1 consolidated manifest from full Matador manifest.

Usage:
    python scripts/build_matador_c1_manifest.py \
        --manifest data/processed/matador/manifest.csv \
        --out-dir data/processed/matador_c1
"""

import argparse
import collections
import csv
import json
import os
import sys
import tempfile
from pathlib import Path


# S2.5 consolidation rules
DROPPED_LABELS = {
    "thermoplastic",
    "thermoset",
    "elastomer",
    "paint",
    "glass",
}

MERGE_MAP = {
    # metal -> generic metal
    "aluminum": "generic_metal",
    "steel": "generic_metal",
    "brass": "generic_metal",
    "iron": "generic_metal",
    "bronze": "generic_metal",
    "copper": "generic_metal",
    # ceramic -> pottery
    "stoneware": "pottery",
    "terracotta": "pottery",
    "porcelain": "pottery",
    # aggregate/terrain -> soil
    "dirt": "soil",
    # vegetation -> foliage
    "shrub": "foliage",
    # rock -> shale
    "sandstone": "shale",
    # rock -> marble
    "quartz": "marble",
    # textile -> satin
    "polyester": "satin",
    "silk": "satin",
    # textile -> natural_fiber
    "cotton": "natural_fiber",
    "linen": "natural_fiber",
    # wood -> paper
    "cardboard": "paper",
    # structural -> concrete
    "cement": "concrete",
}


def _build_c1_label(original_label: str) -> str | None:
    """Return C1 label or None if dropped."""
    if original_label in DROPPED_LABELS:
        return None
    return MERGE_MAP.get(original_label, original_label)


def _build_label_map(all_original_labels: set) -> dict:
    """Build and validate the full original -> C1 mapping."""
    label_map = {}
    for lbl in all_original_labels:
        c1 = _build_c1_label(lbl)
        label_map[lbl] = c1
    # Check for unmapped labels
    unmapped = [lbl for lbl, c1 in label_map.items() if c1 is None and lbl not in DROPPED_LABELS]
    if unmapped:
        raise ValueError(f"Labels with no mapping decision: {unmapped}")
    return label_map


def build_c1_manifest(manifest_path: Path, out_dir: Path) -> dict:
    """Read full manifest, apply C1 rules, write outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read original manifest
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    original_labels = collections.Counter(r["material_label"] for r in rows)
    all_original = set(original_labels.keys())

    # Build and validate mapping
    label_map = _build_label_map(all_original)

    # Write label_map.json
    label_map_path = out_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    # Build C1 manifest (drop rows with dropped labels)
    c1_rows = []
    for r in rows:
        c1_label = label_map[r["material_label"]]
        if c1_label is None:
            continue
        c1_rows.append({
            "sample_id": r["sample_id"],
            "image_path": r["image_path"],
            "label_path": r["label_path"],
            "material_label": r["material_label"],
            "c1_label": c1_label,
        })

    # Write C1 manifest
    manifest_out = out_dir / "manifest.csv"
    with open(manifest_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "label_path", "material_label", "c1_label"])
        writer.writeheader()
        writer.writerows(c1_rows)

    c1_counts = collections.Counter(r["c1_label"] for r in c1_rows)

    return {
        "original_rows": len(rows),
        "original_unique_labels": len(original_labels),
        "c1_rows": len(c1_rows),
        "c1_unique_labels": len(c1_counts),
        "dropped_counts": {lbl: original_labels[lbl] for lbl in sorted(DROPPED_LABELS) if lbl in original_labels},
        "merged_counts": dict(c1_counts),
        "label_map_path": str(label_map_path),
        "manifest_path": str(manifest_out),
    }


def _run_smoke_test():
    """Tiny fake manifest -> C1 to verify logic."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake_manifest = tmp / "manifest.csv"
        with open(fake_manifest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "image_path", "label_path", "material_label"])
            writer.writerow(["s1", "img/1.tif", "lbl/1.txt", "iron"])
            writer.writerow(["s2", "img/2.tif", "lbl/2.txt", "glass"])
            writer.writerow(["s3", "img/3.tif", "lbl/3.txt", "cotton"])
            writer.writerow(["s4", "img/4.tif", "lbl/4.txt", "concrete"])
            writer.writerow(["s5", "img/5.tif", "lbl/5.txt", "timber"])

        out_dir = tmp / "c1"
        stats = build_c1_manifest(fake_manifest, out_dir)

        assert stats["original_rows"] == 5
        assert stats["original_unique_labels"] == 5
        assert stats["c1_rows"] == 4  # glass dropped
        assert stats["c1_unique_labels"] == 4  # iron->generic_metal, cotton->natural_fiber, concrete->concrete, timber
        assert stats["dropped_counts"] == {"glass": 1}

        # Verify manifest contents
        with open(out_dir / "manifest.csv") as f:
            rows = list(csv.DictReader(f))
        c1_labels = {r["c1_label"] for r in rows}
        assert c1_labels == {"generic_metal", "natural_fiber", "concrete", "timber"}

        # Verify label_map.json
        with open(out_dir / "label_map.json") as f:
            lm = json.load(f)
        assert lm["iron"] == "generic_metal"
        assert lm["glass"] is None
        assert lm["timber"] == "timber"

        print("Smoke test passed.")


def main():
    parser = argparse.ArgumentParser(description="Build Matador-C1 consolidated manifest.")
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--out-dir", type=Path, required=False)
    parser.add_argument("--smoke-test", action="store_true", help="Run internal smoke test and exit.")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    for attr in ("manifest", "out_dir"):
        if getattr(args, attr) is None:
            print(f"Error: --{attr.replace('_', '-')} is required.", file=sys.stderr)
            sys.exit(1)

    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    stats = build_c1_manifest(args.manifest, args.out_dir)

    # Validation
    ok = True
    print(f"Original rows:            {stats['original_rows']}")
    print(f"Original unique labels:   {stats['original_unique_labels']}")
    print(f"C1 rows:                  {stats['c1_rows']}")
    print(f"C1 unique labels:         {stats['c1_unique_labels']}")
    print()

    if stats["original_rows"] != 7238:
        print(f"WARNING: expected 7238 original rows, got {stats['original_rows']}", file=sys.stderr)
        ok = False
    if stats["original_unique_labels"] != 57:
        print(f"WARNING: expected 57 original labels, got {stats['original_unique_labels']}", file=sys.stderr)
        ok = False
    if stats["c1_rows"] < 6600 or stats["c1_rows"] > 6650:
        print(f"WARNING: expected ~6614 C1 rows, got {stats['c1_rows']}", file=sys.stderr)
        ok = False
    if stats["c1_unique_labels"] != 37:
        print(f"WARNING: expected 37 C1 labels, got {stats['c1_unique_labels']}", file=sys.stderr)
        ok = False

    if stats["dropped_counts"]:
        print("Dropped counts by original label:")
        for lbl, cnt in sorted(stats["dropped_counts"].items()):
            print(f"  {lbl:20s} {cnt}")
        print()

    print("Merged counts by C1 label:")
    for lbl, cnt in sorted(stats["merged_counts"].items()):
        print(f"  {lbl:20s} {cnt}")

    if ok:
        print(f"\nOutputs written to: {args.out_dir}")
        print(f"  {stats['label_map_path']}")
        print(f"  {stats['manifest_path']}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
