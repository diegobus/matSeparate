#!/usr/bin/env python3
"""Inference API for the dual-encoder local/global ResNet C1 classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def _resolve_device(device_str: str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


class DualEncoderResNet(nn.Module):
    def __init__(self, backbone: str, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        import timm

        self.local_encoder = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        self.context_encoder = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.local_encoder.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, local_image, context_image):
        local_feat = self.local_encoder(local_image)
        context_feat = self.context_encoder(context_image)
        return self.classifier(torch.cat([local_feat, context_feat], dim=1))


class GlobalResNetInference:
    def __init__(self, model, device, local_transform, context_transform, class_to_idx):
        self.model = model
        self.device = device
        self.local_transform = local_transform
        self.context_transform = context_transform
        self.class_to_idx = dict(class_to_idx)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.leaf_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class)]

    @classmethod
    def from_run_dir(cls, run_dir: Union[str, Path], device: str = "auto"):
        import yaml

        run_dir = Path(run_dir)
        with open(run_dir / "config.yaml") as handle:
            config = yaml.safe_load(handle)
        with open(run_dir / "class_to_idx.json") as handle:
            class_to_idx = json.load(handle)

        device_obj = _resolve_device(device)
        model = DualEncoderResNet(
            backbone=config["model"]["backbone"],
            num_classes=len(class_to_idx),
            hidden_dim=config["model"].get("hidden_dim", 1024),
            dropout=config["model"].get("dropout", 0.1),
        )
        checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location=device_obj)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device_obj).eval()

        mean = config["data"].get("mean", [0.485, 0.456, 0.406])
        std = config["data"].get("std", [0.229, 0.224, 0.225])
        local_size = config["training"].get("local_image_size", 224)
        context_size = config["training"].get("context_image_size", 224)
        local_transform = cls._build_transform(local_size, mean, std)
        context_transform = cls._build_transform(context_size, mean, std)
        return cls(model, device_obj, local_transform, context_transform, class_to_idx)

    @staticmethod
    def _build_transform(image_size: int, mean, std):
        return T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.Normalize(mean=mean, std=std),
        ])

    @staticmethod
    def _to_tensor(image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr /= 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return torch.from_numpy(arr).permute(2, 0, 1)

    @torch.inference_mode()
    def infer_batch(
        self,
        local_images: List[Union[str, Path, Image.Image, np.ndarray]],
        context_images: List[Union[str, Path, Image.Image, np.ndarray]],
    ):
        if len(local_images) != len(context_images):
            raise ValueError("local_images and context_images must have the same length")
        local = torch.stack([
            self.local_transform(self._to_tensor(image)) for image in local_images
        ]).to(self.device)
        context = torch.stack([
            self.context_transform(self._to_tensor(image)) for image in context_images
        ]).to(self.device)
        logits = self.model(local, context)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        results = []
        for i in range(len(local_images)):
            results.append({
                "leaf_label": self.idx_to_class[int(pred[i].item())],
                "leaf_probs": probs[i].detach().cpu().numpy(),
                "logits": logits[i].detach().cpu().numpy(),
            })
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--local-image", type=Path, required=True)
    parser.add_argument("--context-image", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    api = GlobalResNetInference.from_run_dir(args.run_dir, device=args.device)
    result = api.infer_batch([args.local_image], [args.context_image])[0]
    print(result["leaf_label"])
