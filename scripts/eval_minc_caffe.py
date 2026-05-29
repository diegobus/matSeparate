#!/usr/bin/env python3
"""Evaluate MINC Caffe GoogLeNet model on MINC-2500 dataset."""

import sys
sys.path.insert(0, '/tmp')  # for caffe_pb2

import os
import struct
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Caffe-faithful GoogLeNet (no BatchNorm, with LRN)
# ---------------------------------------------------------------------------

class LRN(nn.Module):
    def __init__(self, local_size=5, alpha=1e-4, beta=0.75, k=1.0):
        super().__init__()
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        return F.local_response_norm(x, self.local_size, self.alpha, self.beta, self.k)


class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, pad=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=True)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class Inception(nn.Module):
    def __init__(self, in_ch, n1x1, n3x3r, n3x3, n5x5r, n5x5, pool_proj):
        super().__init__()
        self.branch1 = ConvReLU(in_ch, n1x1, 1)
        self.branch2 = nn.Sequential(ConvReLU(in_ch, n3x3r, 1), ConvReLU(n3x3r, n3x3, 3, pad=1))
        self.branch3 = nn.Sequential(ConvReLU(in_ch, n5x5r, 1), ConvReLU(n5x5r, n5x5, 5, pad=2))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            ConvReLU(in_ch, pool_proj, 1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class CaffeGoogLeNet(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()
        self.conv1 = ConvReLU(3, 64, 7, stride=2, pad=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.lrn1  = LRN(local_size=5, alpha=0.0001, beta=0.75)

        self.conv2r = ConvReLU(64, 64, 1)
        self.conv2  = ConvReLU(64, 192, 3, pad=1)
        self.lrn2   = LRN(local_size=5, alpha=0.0001, beta=0.75)
        self.pool2  = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192,  64,  96, 128, 16,  32,  32)
        self.inception3b = Inception(256, 128, 128, 192, 32,  96,  64)
        self.pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192,  96, 208, 16,  48,  64)
        self.inception4b = Inception(512, 160, 112, 224, 24,  64,  64)
        self.inception4c = Inception(512, 128, 128, 256, 24,  64,  64)
        self.inception4d = Inception(512, 112, 144, 288, 32,  64,  64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.lrn1(x)
        x = self.conv2r(x)
        x = self.conv2(x)
        x = self.lrn2(x)
        x = self.pool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Caffemodel weight loader
# ---------------------------------------------------------------------------

def load_caffemodel_weights(model: CaffeGoogLeNet, caffemodel_path: str):
    import caffe_pb2
    net = caffe_pb2.NetParameter()
    with open(caffemodel_path, 'rb') as f:
        net.ParseFromString(f.read())

    # Build name -> blobs dict (supports both old `layers` and new `layer` format)
    blobs = {}
    layers = list(net.layers) or list(net.layer)
    for layer in layers:
        if layer.blobs:
            blobs[layer.name] = [np.array(list(b.data), dtype=np.float32).reshape(
                list(b.shape.dim) if b.shape.dim else
                [b.num, b.channels, b.height, b.width] if b.num else [-1]
            ) for b in layer.blobs]

    def load_conv(module: ConvReLU, caffe_name: str):
        w, b = blobs[caffe_name]
        module.conv.weight.data.copy_(torch.from_numpy(w))
        module.conv.bias.data.copy_(torch.from_numpy(b.reshape(-1)))

    def load_inception(module: Inception, prefix: str):
        load_conv(module.branch1,    f'{prefix}/1x1')
        load_conv(module.branch2[0], f'{prefix}/3x3_reduce')
        load_conv(module.branch2[1], f'{prefix}/3x3')
        load_conv(module.branch3[0], f'{prefix}/5x5_reduce')
        load_conv(module.branch3[1], f'{prefix}/5x5')
        load_conv(module.branch4[1], f'{prefix}/pool_proj')

    load_conv(model.conv1,  'conv1/7x7_s2')
    load_conv(model.conv2r, 'conv2/3x3_reduce')
    load_conv(model.conv2,  'conv2/3x3')

    for tag in ['3a','3b','4a','4b','4c','4d','4e','5a','5b']:
        load_inception(getattr(model, f'inception{tag}'), f'inception_{tag}')

    w, b = blobs['fc8-20']
    model.fc.weight.data.copy_(torch.from_numpy(w.reshape(model.fc.weight.shape)))
    model.fc.bias.data.copy_(torch.from_numpy(b.reshape(-1)))

    print(f"Loaded weights from {caffemodel_path}")
    return model


# ---------------------------------------------------------------------------
# MINC-2500 dataset
# ---------------------------------------------------------------------------

class MINC2500(Dataset):
    def __init__(self, root: str, split_file: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.categories = [l.strip() for l in open(self.root / 'categories.txt')]
        self.cat2idx = {c: i for i, c in enumerate(self.categories)}
        self.samples = []
        with open(split_file) as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                category = path.split('/')[1]
                self.samples.append((path, self.cat2idx[category]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img = Image.open(self.root / rel_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                per_class_total[l] = per_class_total.get(l, 0) + 1
                if p == l:
                    correct += 1
                    per_class_correct[l] = per_class_correct.get(l, 0) + 1
            total += len(labels)
            if total % 500 == 0:
                print(f"  {total}/{len(loader.dataset)}  acc={correct/total:.4f}")

    return correct / total, per_class_correct, per_class_total


def main():
    minc_root = '/home/ubuntu/matSeparate/data/external/minc/minc-2500'
    model_dir  = '/home/ubuntu/matSeparate/data/external/minc/minc-model'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Caffe preprocessing: BGR mean subtraction, resize to 256 then center-crop 224
    # torchvision works in RGB, so we convert mean to RGB order
    mean_bgr = np.array([104, 117, 124], dtype=np.float32)
    mean_rgb = mean_bgr[[2, 1, 0]] / 255.0

    # Caffe expects pixel values in [0,255] minus mean.
    # ToTensor divides by 255, so to undo that scaling use std=1/255.
    # Result: (x/255 - mean/255) / (1/255) = x - mean  (in [0,255] space)
    std_scale = [1.0 / 255.0] * 3

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_rgb, std=std_scale),
    ])

    test_dataset = MINC2500(
        minc_root,
        os.path.join(minc_root, 'labels', 'test1.txt'),
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=True)

    print(f"Test samples: {len(test_dataset)}")
    categories = test_dataset.categories

    model = CaffeGoogLeNet(num_classes=23).to(device)
    load_caffemodel_weights(model, os.path.join(model_dir, 'minc-googlenet.caffemodel'))
    model = model.to(device)

    print("\nEvaluating GoogLeNet on MINC-2500 test split 1...")
    acc, per_class_correct, per_class_total = evaluate(model, test_loader, device)

    print(f"\nOverall accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nPer-class accuracy:")
    for i, cat in enumerate(categories):
        c = per_class_correct.get(i, 0)
        t = per_class_total.get(i, 0)
        print(f"  {cat:15s}: {c/t:.4f}  ({c}/{t})")


if __name__ == '__main__':
    main()
