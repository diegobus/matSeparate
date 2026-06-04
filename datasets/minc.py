"""MINC-2500 patch classification dataset."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
import networkx as nx


class MINC2500Dataset(Dataset):
    """MINC-2500 patch classification dataset.

    Each sample is a 362x362 patch cropped from a photo, annotated with
    a single material category.

    Args:
        root: path to minc-2500 directory (contains images/, labels/, categories.txt)
        split_file: path to labels/trainN.txt (or validateN.txt / testN.txt)
        graph: optional taxonomy DiGraph for multihot target generation
        transform: optional image transform (applied to PIL Image)
        node_to_idx: mapping from node name -> index (required if graph provided)
    """

    CATEGORIES = [
        "brick", "carpet", "ceramic", "fabric", "foliage", "food", "glass",
        "hair", "leather", "metal", "mirror", "other", "painted", "paper",
        "plastic", "polishedstone", "skin", "sky", "stone", "tile",
        "wallpaper", "water", "wood",
    ]

    def __init__(
        self,
        root: str,
        split_file: str,
        graph: Optional[nx.DiGraph] = None,
        transform: Optional[Callable] = None,
        node_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.graph = graph
        self.node_to_idx = node_to_idx
        self.cat_to_id = {c: i for i, c in enumerate(self.CATEGORIES)}

        # Load split file: each line is "images/category/filename.jpg"
        with open(split_file) as f:
            self.samples = [line.strip() for line in f if line.strip()]

        # Precompute ancestor sets if graph provided
        if graph is not None and node_to_idx is not None:
            self._ancestors = self._precompute_ancestors()

    def _precompute_ancestors(self) -> Dict[str, List[int]]:
        """For each leaf, compute indices of all ancestors including itself."""
        ancestors = {}
        for node in self.graph.nodes:
            if self.graph.out_degree(node) == 0:  # leaf
                path_nodes = list(nx.ancestors(self.graph, node)) + [node]
                ancestors[node] = [self.node_to_idx[n] for n in path_nodes
                                   if n in self.node_to_idx]
        return ancestors

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        rel_path = self.samples[idx]
        # rel_path is like "images/brick/brick_000001.jpg"
        parts = rel_path.split("/")
        category = parts[1]  # e.g. "brick"
        img_path = self.root / rel_path

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        label_idx = self.cat_to_id[category]

        result = {
            "image": img,
            "label": label_idx,
            "category": category,
        }

        # Multihot target for hierarchical loss
        if self.graph is not None and self.node_to_idx is not None:
            num_nodes = len(self.node_to_idx)
            multihot = torch.zeros(num_nodes, dtype=torch.float32)
            for i in self._ancestors.get(category, []):
                multihot[i] = 1.0
            result["target_multihot"] = multihot
            result["leaf_idx"] = self.node_to_idx[category]

        return result
