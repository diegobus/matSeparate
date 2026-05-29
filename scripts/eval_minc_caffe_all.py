#!/usr/bin/env python3
"""Evaluate all three MINC Caffe models (GoogLeNet, VGG16, AlexNet) on MINC-2500."""

import sys
sys.path.insert(0, '/tmp')  # for caffe_pb2

import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ---------------------------------------------------------------------------
# Caffe-faithful AlexNet (with group=2 for conv2/4/5, matching BVLC AlexNet)
# ---------------------------------------------------------------------------

class CaffeAlexNet(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.lrn1  = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2, groups=2)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.lrn2  = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1, groups=2)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1, groups=2)
        self.pool5 = nn.MaxPool2d(3, stride=2)
        self.fc6   = nn.Linear(256 * 6 * 6, 4096)
        self.drop6 = nn.Dropout(p=0.5)
        self.fc7   = nn.Linear(4096, 4096)
        self.drop7 = nn.Dropout(p=0.5)
        self.fc8   = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.lrn1(self.pool1(x))
        x = F.relu(self.conv2(x))
        x = self.lrn2(self.pool2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        x = self.drop6(F.relu(self.fc6(x)))
        x = self.drop7(F.relu(self.fc7(x)))
        return self.fc8(x)


# ---------------------------------------------------------------------------
# Caffemodel weight loader
# ---------------------------------------------------------------------------

def parse_caffemodel(path):
    import caffe_pb2
    net = caffe_pb2.NetParameter()
    with open(path, 'rb') as f:
        net.ParseFromString(f.read())
    blobs = {}
    layers = list(net.layers) or list(net.layer)
    for layer in layers:
        if layer.blobs:
            blobs[layer.name] = []
            for b in layer.blobs:
                shape = list(b.shape.dim) if b.shape.dim else [b.num, b.channels, b.height, b.width]
                arr = np.array(list(b.data), dtype=np.float32).reshape(shape)
                blobs[layer.name].append(arr)
    return blobs


def _load_conv(module, blobs, name):
    w, b = blobs[name]
    module.weight.data.copy_(torch.from_numpy(w.reshape(module.weight.shape)))
    module.bias.data.copy_(torch.from_numpy(b.reshape(-1)))


def _load_fc(module, blobs, name):
    w, b = blobs[name]
    module.weight.data.copy_(torch.from_numpy(w.reshape(module.weight.shape)))
    module.bias.data.copy_(torch.from_numpy(b.reshape(-1)))


def load_alexnet(model: CaffeAlexNet, blobs):
    _load_conv(model.conv1, blobs, 'conv1')
    _load_conv(model.conv2, blobs, 'conv2')
    _load_conv(model.conv3, blobs, 'conv3')
    _load_conv(model.conv4, blobs, 'conv4')
    _load_conv(model.conv5, blobs, 'conv5')
    _load_fc(model.fc6, blobs, 'fc6')
    _load_fc(model.fc7, blobs, 'fc7')
    _load_fc(model.fc8, blobs, 'fc8-20')


def load_vgg16(model: nn.Module, blobs):
    caffe_names = [
        'conv1_1','conv1_2','conv2_1','conv2_2',
        'conv3_1','conv3_2','conv3_3',
        'conv4_1','conv4_2','conv4_3',
        'conv5_1','conv5_2','conv5_3',
    ]
    conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
    for module, name in zip(conv_layers, caffe_names):
        _load_conv(module, blobs, name)
    fc_modules = [m for m in model.classifier if isinstance(m, nn.Linear)]
    _load_fc(fc_modules[0], blobs, 'fc6')
    _load_fc(fc_modules[1], blobs, 'fc7')
    _load_fc(fc_modules[2], blobs, 'fc8-20')


# ---------------------------------------------------------------------------
# MINC-2500 dataset
# ---------------------------------------------------------------------------

class MINC2500(Dataset):
    def __init__(self, root, split_file, transform=None):
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
                self.samples.append((path, self.cat2idx[path.split('/')[1]]))

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

def evaluate(model, loader, device, categories):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                per_class_total[l] = per_class_total.get(l, 0) + 1
                if p == l:
                    correct += 1
                    per_class_correct[l] = per_class_correct.get(l, 0) + 1
            total += len(labels)
    acc = correct / total
    print(f"  Overall accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("  Per-class accuracy:")
    for i, cat in enumerate(categories):
        c = per_class_correct.get(i, 0)
        t = per_class_total.get(i, 1)
        print(f"    {cat:15s}: {c/t:.4f}  ({c}/{t})")
    return acc


def make_loader(minc_root, split_file, input_size, batch_size=64):
    # Caffe preprocessing: subtract BGR mean [104,117,124] from [0,255] pixels.
    # ToTensor gives [0,1], so use std=1/255 to recover [0,255] scale before mean sub.
    mean_rgb = np.array([124, 117, 104], dtype=np.float32) / 255.0
    std = [1.0 / 255.0] * 3
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_rgb, std=std),
    ])
    ds = MINC2500(minc_root, split_file, transform=transform)
    return DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True), ds.categories


def main():
    minc_root  = '/home/ubuntu/matSeparate/data/external/minc/minc-2500'
    model_dir  = '/home/ubuntu/matSeparate/data/external/minc/minc-model'
    split_file = os.path.join(minc_root, 'labels', 'test1.txt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    configs = [
        ('alexnet',   'minc-alexnet.caffemodel',   227, CaffeAlexNet,    load_alexnet),
        ('vgg16',     'minc-vgg16.caffemodel',      224, lambda: models.vgg16(num_classes=23), load_vgg16),
    ]

    for arch, fname, crop, build_fn, load_fn in configs:
        print(f"{'='*50}")
        print(f"Model: {arch}")
        loader, categories = make_loader(minc_root, split_file, crop)
        print(f"Parsing {fname} ...")
        blobs = parse_caffemodel(os.path.join(model_dir, fname))
        model = build_fn().to(device)
        load_fn(model, blobs)
        del blobs
        print(f"Evaluating on {len(loader.dataset)} test samples ...")
        evaluate(model, loader, device, categories)
        print()


if __name__ == '__main__':
    main()
