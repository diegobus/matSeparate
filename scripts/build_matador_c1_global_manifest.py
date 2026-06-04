#!/usr/bin/env python3
"""
Build a Matador-C1 manifest with paired local appearance and global context paths.

The input C1 manifest keeps the local swatch path in image_path. This script adds
context_path by matching file stems against sample_id.
"""

import argparse
import csv
import sys
import tarfile
import tempfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _sample_id_from_path(path: str) -> str:
    return Path(path).stem


def _context_path_score(path: str) -> int:
    """Prefer true global context images when duplicate sample IDs exist."""
    parts = Path(path).parts
    if "context_img" in parts:
        return 2
    return 1


def _scan_context_tar(context_tar: Path) -> dict:
    paths = {}
    scores = {}
    with tarfile.open(context_tar, "r:*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            if Path(m.name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            sid = _sample_id_from_path(m.name)
            score = _context_path_score(m.name)
            if sid not in paths or score > scores[sid]:
                paths[sid] = m.name
                scores[sid] = score
    return paths


def _scan_context_root(context_root: Path) -> dict:
    paths = {}
    scores = {}
    for p in context_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        sid = _sample_id_from_path(p.name)
        rel_path = str(p.relative_to(context_root))
        score = _context_path_score(rel_path)
        if sid not in paths or score > scores[sid]:
            paths[sid] = rel_path
            scores[sid] = score
    return paths


def build_global_manifest(
    manifest_csv: Path,
    out_csv: Path,
    context_tar: Path | None = None,
    context_root: Path | None = None,
    fail_on_missing: bool = True,
) -> dict:
    if context_tar is None and context_root is None:
        raise ValueError("Provide context_tar or context_root")
    if context_tar is not None and context_root is not None:
        raise ValueError("Provide only one of context_tar or context_root")

    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    context_paths = (
        _scan_context_tar(context_tar)
        if context_tar is not None
        else _scan_context_root(context_root)
    )

    paired = []
    missing = []
    for row in rows:
        sid = row["sample_id"]
        context_path = context_paths.get(sid)
        if context_path is None:
            missing.append(sid)
            if fail_on_missing:
                continue
        out_row = dict(row)
        out_row["context_path"] = context_path or ""
        paired.append(out_row)

    if missing and fail_on_missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing context paths for {len(missing)} samples, e.g. {preview}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fields = [f for f in fieldnames if f != "context_path"] + ["context_path"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(paired)

    unused_context = set(context_paths) - {r["sample_id"] for r in rows}
    return {
        "input_rows": len(rows),
        "context_images": len(context_paths),
        "paired_rows": len(paired),
        "missing_context": len(missing),
        "unused_context": len(unused_context),
        "out_csv": str(out_csv),
    }


def _run_smoke_test():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        manifest = tmp / "manifest.csv"
        with open(manifest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "image_path", "label_path", "material_label", "c1_label"])
            writer.writerow(["001", "matador/texture_img/001.tiff", "matador/label/001.txt", "wood", "wood"])
            writer.writerow(["002", "matador/texture_img/002.tiff", "matador/label/002.txt", "iron", "generic_metal"])

        context_root = tmp / "context"
        (context_root / "matador" / "context_img").mkdir(parents=True)
        for sid in ("001", "002"):
            (context_root / "matador" / "context_img" / f"{sid}.jpg").write_bytes(b"fake")

        out = tmp / "global_manifest.csv"
        stats = build_global_manifest(manifest, out, context_root=context_root)
        assert stats["paired_rows"] == 2, stats
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["context_path"] == "matador/context_img/001.jpg"
        assert rows[1]["context_path"] == "matador/context_img/002.jpg"

        # If the scan root contains both appearance and context images with the
        # same stem, the manifest must keep context_img.
        (context_root / "matador" / "texture_img").mkdir(parents=True)
        (context_root / "matador" / "texture_img" / "001.tiff").write_bytes(b"fake")
        stats = build_global_manifest(manifest, out, context_root=context_root)
        assert stats["paired_rows"] == 2, stats
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["context_path"] == "matador/context_img/001.jpg"
        print("Smoke test passed.")


def main():
    parser = argparse.ArgumentParser(description="Build Matador-C1 global context manifest.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--context-tar", type=Path)
    parser.add_argument("--context-root", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    if args.manifest is None or args.out is None:
        print("Error: --manifest and --out are required.", file=sys.stderr)
        sys.exit(1)

    stats = build_global_manifest(
        manifest_csv=args.manifest,
        out_csv=args.out,
        context_tar=args.context_tar,
        context_root=args.context_root,
        fail_on_missing=not args.allow_missing,
    )
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
