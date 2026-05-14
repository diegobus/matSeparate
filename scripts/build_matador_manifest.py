#!/usr/bin/env python3
"""
Build a manifest CSV joining Matador appearance images with their material labels.

Usage:
    python scripts/build_matador_manifest.py \
        --appearance-tar data/downloads/matador.appearance.tar \
        --label-tar data/downloads/matador.label.tar \
        --out data/processed/matador/manifest.csv
"""

import argparse
import collections
import csv
import os
import sys
import tarfile
import tempfile
from pathlib import Path


def _extract_sample_id(path: str) -> str:
    return Path(path).stem


def _stream_labels(tar_path: Path) -> dict:
    """Return {sample_id: material_label} from label tar."""
    labels = {}
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            # Read label text (single line)
            f = tf.extractfile(m)
            if f is None:
                continue
            raw = f.read(1024).decode("utf-8", errors="replace").strip()
            sid = _extract_sample_id(m.name)
            labels[sid] = raw
    return labels


def _stream_appearance_paths(tar_path: Path) -> dict:
    """Return {sample_id: tar_internal_path} from appearance tar."""
    paths = {}
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            sid = _extract_sample_id(m.name)
            paths[sid] = m.name
    return paths


def build_manifest(appearance_tar: Path, label_tar: Path, out_csv: Path) -> dict:
    """Build and write manifest, return validation stats."""
    labels = _stream_labels(label_tar)
    images = _stream_appearance_paths(appearance_tar)

    # Find common sample ids
    common_ids = sorted(set(labels.keys()) & set(images.keys()))
    label_only = set(labels.keys()) - set(images.keys())
    image_only = set(images.keys()) - set(labels.keys())

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "image_path", "label_path", "material_label"])
        for sid in common_ids:
            writer.writerow([
                sid,
                images[sid],
                f"matador/label/{sid}.txt",
                labels[sid],
            ])

    label_counts = collections.Counter(labels.values())

    return {
        "total_labels": len(labels),
        "total_images": len(images),
        "matched": len(common_ids),
        "label_only_ids": len(label_only),
        "image_only_ids": len(image_only),
        "unique_labels": len(label_counts),
        "label_counts": label_counts,
    }


def _run_smoke_test():
    """Create a minimal fake tar pair, build manifest, assert correctness."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Build fake label tar
        label_tar = tmp / "labels.tar"
        with tarfile.open(label_tar, "w") as tf:
            for sid, mat in [("001", "iron"), ("002", "wood")]:
                data = mat.encode("utf-8")
                info = tarfile.TarInfo(name=f"labels/{sid}.txt")
                info.size = len(data)
                tf.addfile(info, tarfile.io.BytesIO(data))
        # Build fake appearance tar
        app_tar = tmp / "appearance.tar"
        with tarfile.open(app_tar, "w") as tf:
            for sid in ["001", "002"]:
                data = b"\x00" * 64  # dummy tiff bytes
                info = tarfile.TarInfo(name=f"texture_img/{sid}.tiff")
                info.size = len(data)
                tf.addfile(info, tarfile.io.BytesIO(data))

        out_csv = tmp / "manifest.csv"
        stats = build_manifest(app_tar, label_tar, out_csv)

        assert stats["total_labels"] == 2, stats
        assert stats["total_images"] == 2, stats
        assert stats["matched"] == 2, stats
        assert stats["unique_labels"] == 2, stats
        assert stats["label_only_ids"] == 0, stats
        assert stats["image_only_ids"] == 0, stats

        # Check CSV contents
        with open(out_csv) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["sample_id", "image_path", "label_path", "material_label"]
        assert rows[1] == ["001", "texture_img/001.tiff", "matador/label/001.txt", "iron"]
        assert rows[2] == ["002", "texture_img/002.tiff", "matador/label/002.txt", "wood"]

        print("Smoke test passed.")


def main():
    parser = argparse.ArgumentParser(description="Build Matador manifest CSV.")
    parser.add_argument("--appearance-tar", type=Path, required=False)
    parser.add_argument("--label-tar", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--smoke-test", action="store_true", help="Run internal smoke test and exit.")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    for attr in ("appearance_tar", "label_tar", "out"):
        if getattr(args, attr) is None:
            print(f"Error: --{attr.replace('_', '-')} is required.", file=sys.stderr)
            sys.exit(1)

    for p in (args.appearance_tar, args.label_tar):
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    stats = build_manifest(args.appearance_tar, args.label_tar, args.out)

    print(f"Total labels:  {stats['total_labels']}")
    print(f"Total images:  {stats['total_images']}")
    print(f"Matched:       {stats['matched']}")
    print(f"Label-only:    {stats['label_only_ids']}")
    print(f"Image-only:    {stats['image_only_ids']}")
    print(f"Unique labels: {stats['unique_labels']}")
    print()
    print("Label counts (sorted by label):")
    for label, cnt in sorted(stats["label_counts"].items()):
        print(f"  {label:20s} {cnt}")

    # Hard validation
    ok = True
    if stats["total_labels"] != 7238:
        print(f"WARNING: expected 7238 labels, got {stats['total_labels']}", file=sys.stderr)
        ok = False
    if stats["total_images"] != 7238:
        print(f"WARNING: expected 7238 images, got {stats['total_images']}", file=sys.stderr)
        ok = False
    if stats["matched"] != 7238:
        print(f"WARNING: expected 7238 matched, got {stats['matched']}", file=sys.stderr)
        ok = False
    if stats["unique_labels"] != 57:
        print(f"WARNING: expected 57 unique labels, got {stats['unique_labels']}", file=sys.stderr)
        ok = False

    if ok:
        print(f"\nManifest written to: {args.out}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
