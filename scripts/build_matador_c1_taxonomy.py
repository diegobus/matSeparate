#!/usr/bin/env python3
"""
Build the Matador-C1 taxonomy tree from the full taxonomy and C1 manifest.

Usage:
    python scripts/build_matador_c1_taxonomy.py \
        --manifest data/processed/matador_c1/manifest.csv \
        --out-json taxonomy/assets/matador-c1-taxonomy.json \
        --out-csv data/processed/matador_c1/taxonomy_paths.csv
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set


# C1 taxonomy defined explicitly as nested dict (same format as taxonomy-tree.json)
# Pruned: liquid, gas (no samples in dataset)
# Dropped: thermoplastic, thermoset, elastomer, paint, glass
# Merged leaves replaced by C1 equivalents
MATADOR_C1_TAXONOMY = {
    "name": "root",
    "children": [
        {
            "name": "solid",
            "children": [
                {
                    "name": "abiotic",
                    "children": [
                        {
                            "name": "metal",
                            "children": [
                                {"name": "generic_metal"}
                            ]
                        },
                        {
                            "name": "rock",
                            "children": [
                                {
                                    "name": "solid_mass",
                                    "children": [
                                        {"name": "granite"},
                                        {"name": "limestone"},
                                        {"name": "marble"},
                                        {"name": "shale"},
                                    ]
                                },
                                {
                                    "name": "aggregate",
                                    "children": [
                                        {"name": "gravel"},
                                        {"name": "sand"},
                                    ]
                                }
                            ]
                        },
                        {
                            "name": "ceramic",
                            "children": [
                                {
                                    "name": "decorative",
                                    "children": [
                                        {"name": "plaster"},
                                        {"name": "pottery"},
                                    ]
                                },
                                {
                                    "name": "structural",
                                    "children": [
                                        {"name": "asphalt"},
                                        {"name": "brick"},
                                        {"name": "concrete"},
                                    ]
                                }
                            ]
                        },
                        {
                            "name": "polymer",
                            "children": [
                                {
                                    "name": "textile",
                                    "children": [
                                        {"name": "nylon"},
                                        {"name": "wool"},
                                        {"name": "carbon_fiber"},
                                        {"name": "carpet"},
                                        {"name": "satin"},
                                        {"name": "natural_fiber"},
                                    ]
                                },
                                {
                                    "name": "plastic",
                                    "children": [
                                        {"name": "foam"},
                                        {"name": "wax"},
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    "name": "biotic",
                    "children": [
                        {
                            "name": "natural",
                            "children": [
                                {
                                    "name": "vegetation",
                                    "children": [
                                        {"name": "flower"},
                                        {"name": "foliage"},
                                        {"name": "ivy"},
                                    ]
                                },
                                {
                                    "name": "terrain",
                                    "children": [
                                        {"name": "grass"},
                                        {"name": "moss"},
                                        {"name": "plant_litter"},
                                        {"name": "soil"},
                                        {"name": "straw"},
                                    ]
                                }
                            ]
                        },
                        {
                            "name": "derivative",
                            "children": [
                                {
                                    "name": "wood",
                                    "children": [
                                        {"name": "paper"},
                                        {"name": "timber"},
                                        {"name": "tree_bark"},
                                    ]
                                },
                                {
                                    "name": "animal_hide",
                                    "children": [
                                        {"name": "fur"},
                                        {"name": "leather"},
                                        {"name": "suede"},
                                    ]
                                },
                                {
                                    "name": "food",
                                    "children": [
                                        {"name": "fruit"},
                                        {"name": "vegetable"},
                                        {"name": "bread"},
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}


def _collect_leaves(node: Dict, path: Optional[List[str]] = None) -> Dict[str, str]:
    """Recursively collect leaf names and their root-to-leaf paths."""
    if path is None:
        path = []
    current_path = path + [node["name"]]
    if "children" not in node:
        return {node["name"]: "/".join(current_path)}
    leaves = {}
    for child in node["children"]:
        leaves.update(_collect_leaves(child, current_path))
    return leaves


def build_c1_taxonomy(manifest_path: Path, out_json: Path, out_csv: Path) -> Dict:
    """Write C1 taxonomy JSON and paths CSV, validate against manifest."""
    # Read manifest labels
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        manifest_labels = {row["c1_label"] for row in reader}

    # Collect leaves and paths
    leaf_paths = _collect_leaves(MATADOR_C1_TAXONOMY)
    leaf_labels = set(leaf_paths.keys())

    # Validate: every manifest label must be a leaf
    missing = manifest_labels - leaf_labels
    if missing:
        raise ValueError(f"Manifest labels not found in C1 taxonomy: {sorted(missing)}")

    # Validate: no unexpected leaves (optional but useful)
    extra = leaf_labels - manifest_labels

    # Write JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(MATADOR_C1_TAXONOMY, f, indent=4)

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["c1_label", "taxonomy_path"])
        for label in sorted(leaf_paths.keys()):
            writer.writerow([label, leaf_paths[label]])

    return {
        "manifest_labels": len(manifest_labels),
        "taxonomy_leaves": len(leaf_labels),
        "missing": sorted(missing),
        "extra_leaves": sorted(extra),
        "leaf_paths": leaf_paths,
    }


def _run_smoke_test():
    """Tiny fake manifest -> C1 taxonomy to verify logic."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake_manifest = tmp / "manifest.csv"
        with open(fake_manifest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "image_path", "label_path", "material_label", "c1_label"])
            writer.writerow(["s1", "img/1.tif", "lbl/1.txt", "iron", "generic_metal"])
            writer.writerow(["s2", "img/2.tif", "lbl/2.txt", "concrete", "concrete"])
            writer.writerow(["s3", "img/3.tif", "lbl/3.txt", "cotton", "natural_fiber"])
            writer.writerow(["s4", "img/4.tif", "lbl/4.txt", "glass", "generic_metal"])  # fake remap

        out_json = tmp / "c1_taxonomy.json"
        out_csv = tmp / "paths.csv"

        stats = build_c1_taxonomy(fake_manifest, out_json, out_csv)

        assert stats["manifest_labels"] == 3  # generic_metal, concrete, natural_fiber
        assert stats["taxonomy_leaves"] == 37
        assert not stats["missing"]

        # Verify JSON
        with open(out_json) as f:
            tree = json.load(f)
        assert tree["name"] == "root"

        # Verify CSV
        with open(out_csv) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["c1_label", "taxonomy_path"]
        paths = {r[0]: r[1] for r in rows[1:]}
        assert "generic_metal" in paths
        assert paths["generic_metal"] == "root/solid/abiotic/metal/generic_metal"

        print("Smoke test passed.")


def main():
    parser = argparse.ArgumentParser(description="Build Matador-C1 taxonomy.")
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--out-json", type=Path, required=False)
    parser.add_argument("--out-csv", type=Path, required=False)
    parser.add_argument("--smoke-test", action="store_true", help="Run internal smoke test and exit.")
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    for attr in ("manifest", "out_json", "out_csv"):
        if getattr(args, attr) is None:
            print(f"Error: --{attr.replace('_', '-')} is required.", file=sys.stderr)
            sys.exit(1)

    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    stats = build_c1_taxonomy(args.manifest, args.out_json, args.out_csv)

    ok = True
    print(f"Manifest unique C1 labels: {stats['manifest_labels']}")
    print(f"Taxonomy leaf count:       {stats['taxonomy_leaves']}")
    print()

    if stats["manifest_labels"] != 37:
        print(f"WARNING: expected 37 manifest labels, got {stats['manifest_labels']}", file=sys.stderr)
        ok = False
    if stats["taxonomy_leaves"] != 37:
        print(f"WARNING: expected 37 taxonomy leaves, got {stats['taxonomy_leaves']}", file=sys.stderr)
        ok = False
    if stats["missing"]:
        print(f"ERROR: missing labels: {stats['missing']}", file=sys.stderr)
        ok = False
    if stats["extra_leaves"]:
        print(f"Note: taxonomy leaves not in manifest: {stats['extra_leaves']}")

    print("All C1 taxonomy paths:")
    for label in sorted(stats["leaf_paths"].keys()):
        print(f"  {label:20s} -> {stats['leaf_paths'][label]}")

    if ok:
        print(f"\nOutputs written:")
        print(f"  {args.out_json}")
        print(f"  {args.out_csv}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
