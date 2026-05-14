#!/usr/bin/env python3
"""
Inspect downloaded Matador tarballs or extracted directories without
fully loading images.  Reports structure, file counts, label previews,
and cross-tar sample-id overlap.

Usage:
    python scripts/inspect_matador_release.py --tar data/downloads/matador.label.tar
    python scripts/inspect_matador_release.py --tar data/downloads/matador.appearance.tar
    python scripts/inspect_matador_release.py --root data/external/matador
"""

import argparse
import collections
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional


def _collect_files_from_tar(tar_path: Path) -> List[str]:
    """Return list of file member names (skip directories)."""
    files = []
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if m.isfile():
                files.append(m.name)
    return files


def _collect_files_from_dir(root: Path) -> List[Path]:
    """Return list of file paths under root."""
    return [p for p in root.rglob("*") if p.is_file()]


def _extension_histogram(paths: List[str]) -> Dict[str, int]:
    """Count file extensions (lowercased)."""
    counter: Dict[str, int] = collections.Counter()
    for p in paths:
        ext = Path(p).suffix.lower()
        counter[ext] += 1
    return dict(counter)


def _top_level_dirs(paths: List[str]) -> List[str]:
    """Return sorted unique top-level directory names."""
    tops = set()
    for p in paths:
        parts = Path(p).parts
        if len(parts) > 1:
            tops.add(parts[0])
        else:
            tops.add(".")
    return sorted(tops)


def _is_text_file(name: str) -> bool:
    """Heuristic: is the file likely human-readable text?"""
    text_exts = {".json", ".csv", ".txt", ".yaml", ".yml", ".xml", ".tsv", ".md"}
    return Path(name).suffix.lower() in text_exts


def _peek_tar_member(tar_path: Path, member_name: str, limit: int = 1024) -> str:
    """Read up to `limit` bytes from a tar member and decode as text."""
    with tarfile.open(tar_path, "r:*") as tf:
        f = tf.extractfile(member_name)
        if f is None:
            return ""
        raw = f.read(limit)
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _peek_file(file_path: Path, limit: int = 1024) -> str:
    """Read up to `limit` bytes from a file and decode as text."""
    try:
        with open(file_path, "rb") as f:
            raw = f.read(limit)
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _find_label_candidates(paths: List[str]) -> List[str]:
    """Return paths that look like label/annotation files."""
    candidates = []
    for p in paths:
        name = Path(p).name.lower()
        if any(k in name for k in ("label", "annot", "meta", "class", "map")):
            candidates.append(p)
    if not candidates:
        # fall back to any text-ish file
        for p in paths:
            if _is_text_file(p):
                candidates.append(p)
    return candidates[:10]  # cap


def _extract_sample_id(path: str, mode: str = "full_stem") -> Optional[str]:
    """Heuristic: derive a sample id from a file path."""
    stem = Path(path).stem
    if mode == "full_stem":
        return stem
    elif mode == "before_first_underscore":
        if "_" in stem:
            return stem.split("_")[0]
        return stem
    else:
        raise ValueError(f"Unknown sample_id mode: {mode}")


def _cross_tar_overlap(tar_path: Path, files: List[str], sample_id_mode: str = "full_stem") -> Optional[Dict]:
    """If sibling tars exist in the same directory, report sample-id overlap."""
    parent = tar_path.parent
    if not parent.exists():
        return None

    stem = tar_path.stem  # e.g. matador.label
    prefix = stem.split(".")[0]  # e.g. matador
    siblings = [p for p in parent.iterdir() if p.is_file() and p.suffix == ".tar" and p != tar_path and p.stem.startswith(prefix + ".")]

    if not siblings:
        return None

    ids = set(_extract_sample_id(f, sample_id_mode) for f in files if _extract_sample_id(f, sample_id_mode))
    result: Dict[str, any] = {"current_tar": str(tar_path.name), "siblings": []}
    for sib in siblings:
        sib_files = _collect_files_from_tar(sib)
        sib_ids = set(_extract_sample_id(f, sample_id_mode) for f in sib_files if _extract_sample_id(f, sample_id_mode))
        overlap = len(ids & sib_ids)
        result["siblings"].append({
            "name": sib.name,
            "member_count": len(sib_files),
            "shared_sample_ids": overlap,
            "current_unique": len(ids - sib_ids),
            "sibling_unique": len(sib_ids - ids),
        })
    return result


def inspect_tar(tar_path: Path, sample_id_mode: str = "full_stem") -> Dict:
    files = _collect_files_from_tar(tar_path)
    ext_hist = _extension_histogram(files)
    tops = _top_level_dirs(files)
    label_cands = _find_label_candidates(files)

    report = {
        "source_type": "tar",
        "source_path": str(tar_path.resolve()),
        "member_count": len(files),
        "top_level_dirs": tops,
        "extensions": ext_hist,
        "sample_paths": files[:10],
        "label_candidates": [],
        "cross_tar": _cross_tar_overlap(tar_path, files, sample_id_mode),
    }

    for cand in label_cands:
        preview = _peek_tar_member(tar_path, cand, limit=2048)
        report["label_candidates"].append({
            "path": cand,
            "preview": preview[:1000],
        })

    # Infer sample-id convention
    sample_ids = [_extract_sample_id(f, sample_id_mode) for f in files]
    id_counts = collections.Counter([s for s in sample_ids if s])
    if id_counts:
        report["sample_id_examples"] = id_counts.most_common(10)
        report["inferred_sample_id_convention"] = "full stem" if sample_id_mode == "full_stem" else "before first '_'"
    else:
        report["sample_id_examples"] = []
        report["inferred_sample_id_convention"] = None

    return report


def inspect_dir(root: Path, sample_id_mode: str = "full_stem") -> Dict:
    file_objs = _collect_files_from_dir(root)
    files = [str(p.relative_to(root)) for p in file_objs]
    ext_hist = _extension_histogram(files)
    tops = _top_level_dirs(files)
    label_cands = _find_label_candidates(files)

    report = {
        "source_type": "directory",
        "source_path": str(root.resolve()),
        "file_count": len(files),
        "top_level_dirs": tops,
        "extensions": ext_hist,
        "sample_paths": files[:10],
        "label_candidates": [],
    }

    for cand in label_cands:
        preview = _peek_file(root / cand, limit=2048)
        report["label_candidates"].append({
            "path": cand,
            "preview": preview[:1000],
        })

    sample_ids = [_extract_sample_id(f, sample_id_mode) for f in files]
    id_counts = collections.Counter([s for s in sample_ids if s])
    if id_counts:
        report["sample_id_examples"] = id_counts.most_common(10)
        report["inferred_sample_id_convention"] = "full stem" if sample_id_mode == "full_stem" else "before first '_'"
    else:
        report["sample_id_examples"] = []
        report["inferred_sample_id_convention"] = None

    return report


def _pretty_print(report: Dict) -> None:
    print("=" * 60)
    print(f"Source: {report['source_type']} → {report['source_path']}")
    print("=" * 60)

    if report["source_type"] == "tar":
        print(f"Member count: {report['member_count']}")
    else:
        print(f"File count: {report['file_count']}")

    print(f"\nTop-level dirs: {', '.join(report['top_level_dirs'])}")

    print("\nExtension counts:")
    for ext, cnt in sorted(report["extensions"].items(), key=lambda x: -x[1]):
        print(f"  {ext or '(no ext)':12s} {cnt}")

    print("\nSample paths:")
    for p in report["sample_paths"]:
        print(f"  {p}")

    if report.get("label_candidates"):
        print("\nLabel/annotation candidates:")
        for cand in report["label_candidates"]:
            print(f"\n  → {cand['path']}")
            preview = cand["preview"].replace("\n", "\n     ")
            print(f"     {preview}")
    else:
        print("\nNo label/annotation candidates found.")

    if report.get("sample_id_examples"):
        print(f"\nInferred sample_id convention: {report['inferred_sample_id_convention']}")
        print("Sample ID examples:")
        for sid, cnt in report["sample_id_examples"][:10]:
            print(f"  {sid}  ({cnt} files)")

    if report.get("cross_tar"):
        print("\nCross-tar sample-id overlap:")
        ct = report["cross_tar"]
        print(f"  Current tar: {ct['current_tar']}")
        for sib in ct["siblings"]:
            print(f"    vs {sib['name']}:")
            print(f"      shared ids: {sib['shared_sample_ids']}")
            print(f"      current-only: {sib['current_unique']}")
            print(f"      sibling-only: {sib['sibling_unique']}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect Matador release tarballs or directories.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tar", type=Path, help="Path to a .tar file to inspect.")
    group.add_argument("--root", type=Path, help="Path to an extracted directory to inspect.")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of pretty text.")
    parser.add_argument("--sample-id-mode", choices=["full_stem", "before_first_underscore"], default="full_stem", help="How to derive sample IDs from filenames.")
    args = parser.parse_args()

    if args.tar:
        if not args.tar.exists():
            print(f"Error: tar file not found: {args.tar}", file=sys.stderr)
            sys.exit(1)
        report = inspect_tar(args.tar, args.sample_id_mode)
    else:
        if not args.root.exists():
            print(f"Error: directory not found: {args.root}", file=sys.stderr)
            sys.exit(1)
        report = inspect_dir(args.root, args.sample_id_mode)

    if args.json:
        json.dump(report, sys.stdout, indent=2)
    else:
        _pretty_print(report)


if __name__ == "__main__":
    main()
