#!/usr/bin/env python3
"""
Smoke-test MatadorC1Dataset by loading a few real samples.

Usage:
    python scripts/smoke_test_matador_dataset.py
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow imports from repo root
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.matador import MatadorC1Dataset


def main():
    manifest_csv = repo_root / "data" / "processed" / "matador_c1" / "manifest.csv"
    taxonomy_json = repo_root / "taxonomy" / "assets" / "matador-c1-taxonomy.json"
    appearance_tar = repo_root / "data" / "downloads" / "matador.appearance.tar"

    if not manifest_csv.exists():
        print(f"ERROR: Manifest not found: {manifest_csv}")
        sys.exit(1)
    if not taxonomy_json.exists():
        print(f"ERROR: Taxonomy not found: {taxonomy_json}")
        sys.exit(1)
    if not appearance_tar.exists():
        print(f"ERROR: Appearance tar not found: {appearance_tar}")
        sys.exit(1)

    print("Loading dataset...")
    ds = MatadorC1Dataset(
        manifest_csv=manifest_csv,
        taxonomy_json=taxonomy_json,
        appearance_tar=appearance_tar,
    )

    print(f"Dataset length: {len(ds)}")
    print(f"Num taxonomy nodes: {ds.num_nodes}")
    print()

    # Inspect first 4 samples
    for i in range(min(4, len(ds))):
        sample = ds[i]
        img = sample["image"]
        print(f"--- Sample {i} ---")
        print(f"  sample_id:      {sample['sample_id']}")
        print(f"  c1_label:       {sample['c1_label']}")
        print(f"  material_label: {sample['material_label']}")
        print(f"  taxa:           {' -> '.join(sample['taxa'])}")
        print(f"  image shape:    {tuple(img.shape)}")
        print(f"  image dtype:    {img.dtype}")
        print(f"  image min/max:  {img.min():.4f} / {img.max():.4f}")

        # Validate multihot
        multihot_sum = sample["target_multihot"].sum().item()
        taxa_len = len(sample["taxa"])
        assert multihot_sum == taxa_len, (
            f"multihot sum ({multihot_sum}) != len(taxa) ({taxa_len})"
        )
        print(f"  multihot sum:   {multihot_sum} (== len(taxa) {taxa_len}) ✓")

        # Validate indices
        indices = sample["target_indices"].tolist()
        expected_indices = [ds.node_to_idx[n] for n in sample["taxa"]]
        assert indices == expected_indices, (
            f"indices mismatch: {indices} vs {expected_indices}"
        )
        print(f"  indices match:  ✓")
        print()

    # Test DataLoader batching
    print("Testing DataLoader batch_size=4 ...")
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    images = batch["image"]
    print(f"  Batch image shape: {tuple(images.shape)}")
    print(f"  Batch dtype:       {images.dtype}")
    print()

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
