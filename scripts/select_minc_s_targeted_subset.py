#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PhotoStats:
    photo_id: str
    total_segments: int
    overlap_any_segments: int
    overlap_leaf_segments: int
    overlap_any_fraction: float
    overlap_leaf_fraction: float
    category_counts: dict[str, int]


def load_categories(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def collect_taxonomy_names(node: dict, all_names: set[str], leaf_names: set[str]) -> None:
    name = node["name"]
    all_names.add(name)
    children = node.get("children", [])
    if children:
        for child in children:
            collect_taxonomy_names(child, all_names, leaf_names)
    else:
        leaf_names.add(name)


def load_taxonomy_sets(path: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(path.read_text())
    all_names: set[str] = set()
    leaf_names: set[str] = set()
    collect_taxonomy_names(payload, all_names, leaf_names)
    return all_names, leaf_names


def gather_photo_stats(
    segments_txt: Path,
    categories: list[str],
    photos_dir: Path,
    overlap_any: set[str],
    overlap_leaf: set[str],
    only_missing: bool,
) -> tuple[list[PhotoStats], dict[str, int]]:
    available_photos = {path.stem for path in photos_dir.glob("*") if path.is_file()}
    per_photo_total: Counter[str] = Counter()
    per_photo_overlap_any: Counter[str] = Counter()
    per_photo_overlap_leaf: Counter[str] = Counter()
    per_photo_categories: dict[str, Counter[str]] = defaultdict(Counter)

    with open(segments_txt) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            label_str, photo_id, _ = parts
            label_name = categories[int(label_str)]
            if only_missing and photo_id in available_photos:
                continue
            if not only_missing and photo_id not in available_photos:
                continue
            per_photo_total[photo_id] += 1
            per_photo_categories[photo_id][label_name] += 1
            if label_name in overlap_any:
                per_photo_overlap_any[photo_id] += 1
            if label_name in overlap_leaf:
                per_photo_overlap_leaf[photo_id] += 1

    stats: list[PhotoStats] = []
    for photo_id, total_segments in per_photo_total.items():
        overlap_any_segments = per_photo_overlap_any[photo_id]
        overlap_leaf_segments = per_photo_overlap_leaf[photo_id]
        stats.append(
            PhotoStats(
                photo_id=photo_id,
                total_segments=total_segments,
                overlap_any_segments=overlap_any_segments,
                overlap_leaf_segments=overlap_leaf_segments,
                overlap_any_fraction=overlap_any_segments / total_segments if total_segments else 0.0,
                overlap_leaf_fraction=overlap_leaf_segments / total_segments if total_segments else 0.0,
                category_counts=dict(per_photo_categories[photo_id]),
            )
        )

    summary = {
        "photo_count": len(stats),
        "segment_count": sum(item.total_segments for item in stats),
        "overlap_any_segments": sum(item.overlap_any_segments for item in stats),
        "overlap_leaf_segments": sum(item.overlap_leaf_segments for item in stats),
    }
    return stats, summary


def sort_key(item: PhotoStats) -> tuple:
    return (
        -item.overlap_leaf_segments,
        -item.overlap_any_segments,
        -item.total_segments,
        -item.overlap_leaf_fraction,
        -item.overlap_any_fraction,
        item.photo_id,
    )


def write_txt(photo_ids: list[str], path: Path) -> None:
    path.write_text("\n".join(photo_ids) + ("\n" if photo_ids else ""))


def write_csv(rows: list[PhotoStats], path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "photo_id",
                "total_segments",
                "overlap_any_segments",
                "overlap_leaf_segments",
                "overlap_any_fraction",
                "overlap_leaf_fraction",
                "dominant_categories",
            ]
        )
        for row in rows:
            dominant = "; ".join(
                f"{name}:{count}" for name, count in sorted(row.category_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            )
            writer.writerow(
                [
                    row.photo_id,
                    row.total_segments,
                    row.overlap_any_segments,
                    row.overlap_leaf_segments,
                    f"{row.overlap_any_fraction:.4f}",
                    f"{row.overlap_leaf_fraction:.4f}",
                    dominant,
                ]
            )


def aggregate_categories(rows: list[PhotoStats]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row.category_counts)
    return dict(counter)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-txt", type=Path, default=Path("data/external/minc/minc-s/test-segments.txt"))
    parser.add_argument("--categories", type=Path, default=Path("data/external/minc/minc-s/categories.txt"))
    parser.add_argument("--photos-dir", type=Path, default=Path("data/external/minc/minc-s/photos"))
    parser.add_argument("--taxonomy-json", type=Path, default=Path("taxonomy/assets/matador-c1-taxonomy.json"))
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("out/minc_s_targeted_subset"))
    parser.add_argument("--only-missing", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    categories = load_categories(args.categories)
    all_names, leaf_names = load_taxonomy_sets(args.taxonomy_json)
    overlap_any = set(categories) & all_names
    overlap_leaf = set(categories) & leaf_names

    stats, pool_summary = gather_photo_stats(
        segments_txt=args.segments_txt,
        categories=categories,
        photos_dir=args.photos_dir,
        overlap_any=overlap_any,
        overlap_leaf=overlap_leaf,
        only_missing=args.only_missing,
    )
    ranked = sorted(stats, key=sort_key)
    selected = ranked[: args.top_k]
    selected_ids = [item.photo_id for item in selected]

    mode_name = "missing" if args.only_missing else "present"
    prefix = f"targeted_{mode_name}_top{len(selected)}"

    write_txt(selected_ids, args.output_dir / f"{prefix}_photo_ids.txt")
    write_csv(selected, args.output_dir / f"{prefix}_photo_stats.csv")

    selected_summary = {
        "mode": mode_name,
        "top_k": len(selected),
        "pool_summary": pool_summary,
        "selected_photo_count": len(selected),
        "selected_segment_count": sum(item.total_segments for item in selected),
        "selected_overlap_any_segments": sum(item.overlap_any_segments for item in selected),
        "selected_overlap_leaf_segments": sum(item.overlap_leaf_segments for item in selected),
        "selected_overlap_any_fraction": (
            sum(item.overlap_any_segments for item in selected) / sum(item.total_segments for item in selected)
            if selected
            else 0.0
        ),
        "selected_overlap_leaf_fraction": (
            sum(item.overlap_leaf_segments for item in selected) / sum(item.total_segments for item in selected)
            if selected
            else 0.0
        ),
        "overlap_any_labels": sorted(overlap_any),
        "overlap_leaf_labels": sorted(overlap_leaf),
        "selected_category_counts": aggregate_categories(selected),
        "top_20_photo_ids": selected_ids[:20],
        "top_20_rows": [asdict(item) for item in selected[:20]],
    }
    (args.output_dir / f"{prefix}_summary.json").write_text(json.dumps(selected_summary, indent=2))

    print(json.dumps(
        {
            "mode": mode_name,
            "selected_photo_count": len(selected),
            "selected_segment_count": selected_summary["selected_segment_count"],
            "selected_overlap_any_fraction": round(selected_summary["selected_overlap_any_fraction"], 4),
            "selected_overlap_leaf_fraction": round(selected_summary["selected_overlap_leaf_fraction"], 4),
            "top_10_photo_ids": selected_ids[:10],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
