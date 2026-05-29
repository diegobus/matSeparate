"""
PyTorch Dataset for MINC-2500 material classification.

Images are JPEG patches loaded from an extracted directory tree.
Returns the same sample dict format as MatadorC1Dataset so the
existing HGNN training script works without modification.

Example:
    from datasets.minc import MINCDataset
    ds = MINCDataset(
        manifest_csv="data/processed/minc/manifest.csv",
        taxonomy_json="taxonomy/assets/minc-taxonomy.json",
        images_root="data/external/minc/minc-2500",
    )
    sample = ds[0]
    # sample["image"]           -> Tensor[3, H, W]
    # sample["target_multihot"] -> Tensor[num_nodes]
    # sample["taxa"]            -> ["root", "manufactured", "construction", "brick"]
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from taxonomy.tree import get_taxonomy


class MINCDataset(Dataset):
    def __init__(
        self,
        manifest_csv: Union[str, Path],
        taxonomy_json: Union[str, Path],
        images_root: Union[str, Path],
        node_index_json: Optional[Union[str, Path]] = None,
        transform=None,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.images_root = Path(images_root)
        self.transform = transform

        self.samples = self._load_manifest(self.manifest_csv)

        self.taxonomy = get_taxonomy(taxonomy_json)
        self.nodes = sorted(self.taxonomy.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = self.nodes
        self.num_nodes = len(self.nodes)

        if node_index_json is None:
            node_index_json = self.manifest_csv.parent / "node_index.json"
        self.node_index_json = Path(node_index_json)
        self._persist_node_index()

        self.root_name = "root"
        self.c1_label_to_path: Dict[str, List[str]] = {}
        for c1_label in {s["c1_label"] for s in self.samples}:
            try:
                path = nx.shortest_path(self.taxonomy, self.root_name, c1_label)
                self.c1_label_to_path[c1_label] = path
            except nx.NetworkXNoPath:
                raise ValueError(f"No path from '{self.root_name}' to '{c1_label}' in taxonomy")

    def _load_manifest(self, path: Path) -> List[Dict]:
        import csv
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    def _persist_node_index(self):
        if self.node_index_json.exists():
            with open(self.node_index_json) as f:
                saved = json.load(f)
            if saved["idx_to_node"] != self.nodes:
                raise ValueError(
                    f"node_index.json ordering mismatch.\n"
                    f"Saved:   {saved['idx_to_node']}\n"
                    f"Current: {self.nodes}"
                )
            return
        self.node_index_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.node_index_json, "w") as f:
            json.dump(
                {"node_to_idx": self.node_to_idx, "idx_to_node": self.idx_to_node},
                f,
                indent=2,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: str) -> torch.Tensor:
        full_path = self.images_root / image_path
        img = Image.open(full_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC [0,1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW
        if self.transform:
            tensor = self.transform(tensor)
        return tensor

    def __getitem__(self, idx: int) -> Dict:
        row = self.samples[idx]
        sample_id = row["sample_id"]
        c1_label = row["c1_label"]
        material_label = row["material_label"]
        image_path = row["image_path"]

        taxa = self.c1_label_to_path[c1_label]

        target_multihot = torch.zeros(self.num_nodes, dtype=torch.float32)
        for node in taxa:
            target_multihot[self.node_to_idx[node]] = 1.0

        target_indices = torch.tensor(
            [self.node_to_idx[node] for node in taxa],
            dtype=torch.long,
        )

        image = self._load_image(image_path)

        return {
            "image": image,
            "sample_id": sample_id,
            "material_label": material_label,
            "c1_label": c1_label,
            "taxa": taxa,
            "target_multihot": target_multihot,
            "target_indices": target_indices,
            "image_path": image_path,
        }
