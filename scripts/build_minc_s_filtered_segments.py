#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def load_photo_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-segments", type=Path, default=Path("data/external/minc/minc-s/test-segments.txt"))
    parser.add_argument("--photo-ids", type=Path, required=True)
    parser.add_argument("--output-segments", type=Path, required=True)
    args = parser.parse_args()

    keep_photo_ids = load_photo_ids(args.photo_ids)
    rows = []
    total_rows = 0
    kept_rows = 0

    with open(args.input_segments) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            total_rows += 1
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            _, photo_id, _ = parts
            if photo_id in keep_photo_ids:
                rows.append(line)
                kept_rows += 1

    args.output_segments.parent.mkdir(parents=True, exist_ok=True)
    args.output_segments.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"input_rows={total_rows}")
    print(f"kept_rows={kept_rows}")
    print(f"kept_photo_ids={len(keep_photo_ids)}")
    print(f"output_segments={args.output_segments}")


if __name__ == "__main__":
    main()
