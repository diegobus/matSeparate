#!/usr/bin/env python3
"""
Train graph-structure ablation variants on MINC-2500 fold-1.

Four variants, all trained with CE loss on leaf logits (loss is not a confound):

  mlp_head     -- ResNet50 + 4-layer MLP, ~430K non-CNN params
                  Ablation 1: is it the graph structure, or just extra parameters?
  hgnn_ce      -- HGNN with the true material taxonomy graph + CE loss
                  Baseline for topology ablations; same architecture as main HGNN
                  but without the hierarchical greedy_loss.
  random_tree  -- HGNN with a random spanning tree over the same 38 nodes
                  Ablation 2: does the specific semantic hierarchy matter?
  full_graph   -- HGNN with fully-connected edges between all 38 nodes
                  Ablation 2: does the graph need to be sparse / tree-structured?

All share: ResNet50 backbone (pretrained, frozen), AdamW 1e-4, cosine LR, 5 epochs.

Usage:
    python scripts/train_ablation_variants.py                    # all four
    python scripts/train_ablation_variants.py --variant mlp_head
    python scripts/train_ablation_variants.py --epochs 10
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from datasets.minc import MINC2500Dataset
from gnn_classifier.hgnn import HGNN, ImageEncoder
from taxonomy.tree import get_taxonomy

VARIANTS = ["mlp_head", "mlp_matched", "hgnn_ce", "random_tree", "full_graph"]


# ── Model definitions ──────────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    """
    ResNet50 backbone + 4-layer MLP head (~430K non-CNN params).

    Output is 23 leaf logits (CE on leaf classes directly).
    Deliberately smaller than HGNN-CE to show the original ablation 1 gap.
    """

    def __init__(self, num_classes: int = 23, backbone: str = "resnet50",
                 pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        self.encoder = ImageEncoder(output_dim=512, backbone=backbone,
                                    pretrained=pretrained, finetune=False)
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class MLPMatchedClassifier(nn.Module):
    """
    Parameter-matched MLP head: same frozen ResNet50 features as HGNN-CE,
    same 38-node output, same CE-on-leaf-logits training — only difference
    is MLP instead of GAT message passing.

    Architecture:  512 → 560 → 560 → 38  (~622K non-CNN params ≈ HGNN-CE's 616K)
    Output:        38 taxonomy-node logits; leaf logits extracted at eval time
                   identically to HGNN-CE.

    Solves the parameter-count confound in mlp_head vs hgnn_ce: if HGNN-CE
    still outperforms this model, the gain is attributable to the structured
    prototype/message-passing mechanism, not extra capacity.
    """

    def __init__(self, num_nodes: int = 38, backbone: str = "resnet50",
                 pretrained: bool = True, dropout: float = 0.1,
                 hidden: int = 560):
        super().__init__()
        self.encoder = ImageEncoder(output_dim=512, backbone=backbone,
                                    pretrained=pretrained, finetune=False)
        self.head = nn.Sequential(
            nn.Linear(512, hidden),  nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, num_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ── Graph topology constructors ───────────────────────────────────────────────

def make_random_tree(nodes: list, root: str = "root", seed: int = 0) -> nx.DiGraph:
    """Random spanning tree over the same node set — valid rooted DAG."""
    rng = random.Random(seed)
    non_root = [n for n in nodes if n != root]
    rng.shuffle(non_root)
    g = nx.DiGraph()
    g.add_node(root)
    for i, node in enumerate(non_root):
        g.add_node(node)
        g.add_edge(rng.choice([root] + non_root[:i]), node)
    return g


def make_full_graph(nodes: list, root: str = "root") -> nx.DiGraph:
    """
    Fully-connected directed graph (each node → every other node).

    Structured as root → all, plus all-to-all among leaf/internal nodes.
    Passing this to HGNN tests whether the sparse tree topology is necessary
    or if message passing between any pair of nodes achieves the same effect.
    """
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n)
    non_root = [n for n in nodes if n != root]
    for n in non_root:
        g.add_edge(root, n)
    for i, u in enumerate(non_root):
        for v in non_root[i + 1:]:
            g.add_edge(u, v)
            g.add_edge(v, u)
    return g


def canonicalize(g: nx.DiGraph) -> nx.DiGraph:
    order = list(nx.topological_sort(g)) if nx.is_directed_acyclic_graph(g) else list(g.nodes)
    g2 = nx.DiGraph()
    for n in order:
        g2.add_node(n, **g.nodes[n])
    for u, v in g.edges:
        g2.add_edge(u, v, **g.edges[u, v])
    return g2


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, leaf_indices, optimizer, device,
              is_hgnn: bool, train: bool, use_leaf_indices: bool = False):
    """
    use_leaf_indices: if True (mlp_matched), extract leaf logits via leaf_indices
                      even though the model is not a GNN.
    """
    model.train() if train else model.eval()
    total_loss = correct = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)

            if is_hgnn or use_leaf_indices:
                logits = model(imgs)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                leaf_logits = logits[:, leaf_indices]
            else:
                leaf_logits = model(imgs)

            loss = F.cross_entropy(leaf_logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (leaf_logits.argmax(1) == labels).sum().item()
            n          += imgs.size(0)

    return total_loss / n, correct / n


# ── Train one variant ─────────────────────────────────────────────────────────

def train_variant(variant: str, args, cfg: dict, taxonomy_g: nx.DiGraph,
                  leaf_indices: torch.Tensor, node_to_idx: dict, device):
    run_dir = repo_root / args.runs_dir / variant
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build model and graph
    if variant == "mlp_head":
        model = MLPClassifier(num_classes=23, backbone=cfg["backbone"],
                               pretrained=True, dropout=cfg["dropout"])
        is_hgnn = False
        cnn_p   = sum(p.numel() for p in model.encoder.parameters())
        non_cnn = (sum(p.numel() for p in model.parameters()) - cnn_p) / 1e3
        print(f"  MLP non-backbone params: {non_cnn:.0f}K")
    elif variant == "mlp_matched":
        # Parameter-matched MLP: 512 → 560 → 560 → 38, same output dim as HGNN-CE
        num_nodes = taxonomy_g.number_of_nodes()
        model = MLPMatchedClassifier(num_nodes=num_nodes, backbone=cfg["backbone"],
                                      pretrained=True, dropout=cfg["dropout"])
        is_hgnn = False
        cnn_p   = sum(p.numel() for p in model.encoder.parameters())
        non_cnn = (sum(p.numel() for p in model.parameters()) - cnn_p) / 1e3
        print(f"  MLP-matched non-backbone params: {non_cnn:.0f}K")
    else:
        if variant == "hgnn_ce":
            g = taxonomy_g
        elif variant == "random_tree":
            g = canonicalize(make_random_tree(list(taxonomy_g.nodes), seed=args.seed))
            for n in g.nodes:
                g.nodes[n].update(taxonomy_g.nodes.get(n, {}))
        elif variant == "full_graph":
            g = make_full_graph(list(taxonomy_g.nodes))
            for n in g.nodes:
                g.nodes[n].update(taxonomy_g.nodes.get(n, {}))
        model = HGNN(
            graph=g,
            cnn_kwargs=dict(backbone=cfg["backbone"], pretrained=True,
                            output_dim=cfg["cnn_output_dim"], finetune=False),
            gnn_kwargs=dict(input_dim=cfg["cnn_output_dim"], hidden_dim=cfg["gnn_hidden_dim"],
                            output_dim=cfg["gnn_output_dim"], num_layers=cfg["gnn_layers"],
                            num_heads=1, skip_connection=True, dropout=cfg["dropout"]),
            dropout_prob=cfg["dropout"],
        )
        is_hgnn = True
        cnn_p   = sum(p.numel() for p in model.cnn.parameters())
        non_cnn = (sum(p.numel() for p in model.parameters()) - cnn_p) / 1e3
        print(f"  {variant} non-backbone params: {non_cnn:.0f}K")

    model = model.to(device)

    # Data loaders
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    aug = T.Compose([
        T.RandomResizedCrop(cfg["image_size"], scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(), T.ColorJitter(0.3, 0.3, 0.2, 0.05),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(256), T.CenterCrop(cfg["image_size"]),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    root = repo_root / args.data_root
    train_ds = MINC2500Dataset(root, root / "labels" / f"train{args.fold}.txt", transform=aug)
    val_ds   = MINC2500Dataset(root, root / "labels" / f"validate{args.fold}.txt", transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # mlp_matched outputs 38 logits (like HGNN variants) — needs leaf_indices
    use_leaf_indices = (variant == "mlp_matched")

    # Differential LR: backbone 0.1×, head 1×
    if is_hgnn:
        backbone_params = list(model.cnn.parameters())
        head_params     = [p for p in model.parameters()
                           if not any(p is q for q in backbone_params)]
    else:
        backbone_params = list(model.encoder.parameters())
        head_params     = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    metrics_all = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, leaf_indices, optimizer,
                                    device, is_hgnn, train=True,
                                    use_leaf_indices=use_leaf_indices)
        vl_loss, vl_acc = run_epoch(model, val_loader,   leaf_indices, None,
                                    device, is_hgnn, train=False,
                                    use_leaf_indices=use_leaf_indices)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{args.epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f} | {elapsed:.0f}s")

        metrics_all.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                             "val_loss": vl_loss, "val_acc": vl_acc, "time_sec": elapsed})
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_all, f, indent=2)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            save = {"model_state_dict": model.state_dict(), "variant": variant,
                    "val_acc": vl_acc, "epoch": epoch, "node_to_idx": node_to_idx}
            torch.save(save, run_dir / "checkpoint_best.pt")

    # config.json written last — acts as a completion sentinel for the runner
    with open(run_dir / "config.json", "w") as f:
        json.dump({**cfg, "variant": variant, "best_val_acc": best_val_acc,
                   "epochs": args.epochs, "is_hgnn": is_hgnn}, f, indent=2)

    print(f"  Best val acc: {best_val_acc:.4f}  →  {run_dir}")
    return best_val_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",        default="data/external/minc/minc-2500")
    p.add_argument("--taxonomy",         default="taxonomy/assets/minc-taxonomy.json")
    p.add_argument("--fold",             type=int,   default=1)
    p.add_argument("--epochs",           type=int,   default=5)
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--weight-decay",     type=float, default=5e-4)
    p.add_argument("--image-size",       type=int,   default=224)
    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--backbone",         default="resnet50")
    p.add_argument("--cnn-output-dim",   type=int,   default=512)
    p.add_argument("--gnn-hidden-dim",   type=int,   default=256)
    p.add_argument("--gnn-output-dim",   type=int,   default=128)
    p.add_argument("--gnn-layers",       type=int,   default=2)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--runs-dir",         default="runs/graph_ablations")
    p.add_argument("--hgnn-ref-run",     default="runs/minc_hgnn",
                   help="Directory containing the reference HGNN run; used to "
                        "borrow node_to_idx so all variants share the same "
                        "leaf ordering.")
    p.add_argument("--variant", choices=VARIANTS + ["all"], default="all")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load taxonomy
    tax_path = repo_root / args.taxonomy
    g_raw    = get_taxonomy(str(tax_path))
    taxonomy_g = canonicalize(g_raw)

    # Leaf indices — borrow node_to_idx from the reference HGNN run so all
    # ablation variants share the same canonical leaf ordering.
    ref_ckpts = sorted((repo_root / args.hgnn_ref_run).glob("*/checkpoint_best.pt"))
    if not ref_ckpts:
        raise FileNotFoundError(
            f"No HGNN checkpoint found under {args.hgnn_ref_run}. "
            "Train the main HGNN first with train_hgnn.py."
        )
    ref_ckpt = torch.load(ref_ckpts[-1], map_location="cpu")
    node_to_idx = ref_ckpt["node_to_idx"]
    leaves = [n for n in taxonomy_g.nodes if taxonomy_g.out_degree(n) == 0]
    leaf_indices = torch.tensor(
        sorted([node_to_idx[n] for n in leaves]), dtype=torch.long, device=device
    )
    print(f"Taxonomy: {taxonomy_g.number_of_nodes()} nodes, {len(leaves)} leaves")

    cfg = {
        "backbone": args.backbone, "cnn_output_dim": args.cnn_output_dim,
        "gnn_hidden_dim": args.gnn_hidden_dim, "gnn_output_dim": args.gnn_output_dim,
        "gnn_layers": args.gnn_layers, "dropout": args.dropout,
        "image_size": args.image_size,
    }

    variants = VARIANTS if args.variant == "all" else [args.variant]
    results  = {}
    for variant in variants:
        print(f"\n{'='*60}\nTraining: {variant}\n{'='*60}")
        results[variant] = train_variant(
            variant, args, cfg, taxonomy_g, leaf_indices, node_to_idx, device
        )

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(f"{'Variant':<20} {'Best val acc':>14}")
    print("-" * 36)
    for v, acc in results.items():
        print(f"{v:<20} {acc:>14.4f}")


if __name__ == "__main__":
    main()
