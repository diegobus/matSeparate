"""
Evaluate SAM + HGNN reconciled segmentation maps against MINC-S GT annotations.

For each photo:
  - Build GT pixel label map from MINC-S segment masks
  - Run reconciliation (SAM proposals + HGNN) to get predicted label map
  - Evaluate pixel accuracy and per-class IoU on GT-covered pixels only

GT pixels not covered by any annotation are excluded (MINC-S segments are
sparse — they annotate specific material regions, not the full image).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_classifier.hgnn import HGNN
from taxonomy.tree import get_taxonomy
import networkx as nx

CATEGORIES = [
    'brick','carpet','ceramic','fabric','foliage','food','glass','hair',
    'leather','metal','mirror','other','painted','paper','plastic',
    'polishedstone','skin','sky','stone','tile','wallpaper','water','wood',
]
NUM_CLASSES = len(CATEGORIES)
OTHER_IDX = CATEGORIES.index('other')
IMAGENET_MEAN = (123, 116, 103)


def load_hgnn(run_dir, taxonomy_path, device):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / 'checkpoint_best.pt', map_location=device)
    node_to_idx = ckpt['node_to_idx']
    with open(run_dir / 'config.json') as f:
        cfg = json.load(f)
    g = get_taxonomy(taxonomy_path)
    node_order = list(nx.topological_sort(g))
    g_canon = nx.DiGraph()
    for n in node_order:
        g_canon.add_node(n, **g.nodes[n])
    for u, v in g.edges:
        g_canon.add_edge(u, v, **g.edges[u, v])
    model = HGNN(
        graph=g_canon,
        cnn_kwargs=dict(backbone=cfg['backbone'], pretrained=False,
                        output_dim=cfg['cnn_output_dim'], finetune=False),
        gnn_kwargs=dict(input_dim=cfg['cnn_output_dim'], hidden_dim=cfg['gnn_hidden_dim'],
                        output_dim=cfg['gnn_output_dim'], num_layers=cfg['gnn_layers'],
                        num_heads=1, skip_connection=True, dropout=cfg['dropout']),
        dropout_prob=cfg['dropout'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    leaves = [n for n in g_canon.nodes if g_canon.out_degree(n) == 0]
    leaf_indices = torch.tensor(sorted([node_to_idx[n] for n in leaves]),
                                dtype=torch.long, device=device)
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    leaf_to_cat = torch.zeros(len(leaf_indices), dtype=torch.long, device=device)
    for pos, idx in enumerate(leaf_indices.tolist()):
        leaf_to_cat[pos] = CATEGORIES.index(idx_to_node[idx])
    return model, leaf_indices, leaf_to_cat


def crop_mask(photo_np, mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    crop = photo_np[r0:r1, c0:c1].copy()
    m = mask[r0:r1, c0:c1]
    for ch, mv in enumerate(IMAGENET_MEAN):
        crop[:, :, ch][~m] = mv
    return Image.fromarray(crop)


def reconcile(sam_masks, softmax_scores, H, W, conf_threshold=0.3):
    label_map = np.full((H, W), OTHER_IDX, dtype=np.int32)
    conf_map  = np.zeros((H, W), dtype=np.float32)
    for mask, scores in zip(sam_masks, softmax_scores):
        conf = float(scores.max())
        pred = int(scores.argmax()) if conf >= conf_threshold else OTHER_IDX
        update = mask & (conf > conf_map)
        label_map[update] = pred
        conf_map[update]  = conf
    return label_map, conf_map


def build_gt_label_map(group, repo_root, H, W):
    """Build a per-pixel GT label map from MINC-S segment annotations."""
    gt_map = np.full((H, W), -1, dtype=np.int32)  # -1 = unannotated
    for _, row in group.iterrows():
        p = Path(row['mask_path'])
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            continue
        mask = np.array(Image.open(p))
        if mask.shape != (H, W):
            mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255)
                            .resize((W, H), Image.NEAREST)) > 127
        else:
            mask = mask.astype(bool)
        gt_map[mask] = int(row['label_index'])
    return gt_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sam-dir', default='out/sam_eval_top200_matador_auto_vit_b')
    p.add_argument('--hgnn-run', default='runs/minc_hgnn/20260603_092440')
    p.add_argument('--taxonomy', default='taxonomy/assets/minc-taxonomy.json')
    p.add_argument('--conf-threshold', type=float, default=0.3)
    p.add_argument('--n-photos', type=int, default=None, help='Limit photos (default: all)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repo_root = Path(__file__).parent.parent
    sam_dir = repo_root / args.sam_dir
    cache_dir = sam_dir / 'cache' / 'auto_masks'

    print("Loading HGNN...")
    model, leaf_indices, leaf_to_cat = load_hgnn(
        repo_root / args.hgnn_run, repo_root / args.taxonomy, device
    )

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])

    df = pd.read_csv(sam_dir / 'per_segment_metrics.csv')
    by_photo = {pid: grp for pid, grp in df.groupby('photo_id')}

    pkl_files = sorted(cache_dir.glob('*.pkl'))
    if args.n_photos:
        pkl_files = pkl_files[:args.n_photos]

    # Accumulators for pixel-level metrics
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)  # [gt, pred]
    total_annotated = 0
    total_correct = 0
    photos_done = 0

    for pkl_path in pkl_files:
        photo_id_str = pkl_path.stem.split('__')[0]
        photo_id_int = int(photo_id_str)

        if photo_id_int not in by_photo:
            continue

        group = by_photo[photo_id_int]
        photo_path = group['photo_path'].iloc[0]
        photo_path = Path(photo_path) if Path(photo_path).is_absolute() else repo_root / photo_path
        if not photo_path.exists():
            continue

        photo_np = np.array(Image.open(photo_path).convert('RGB'))
        H, W = photo_np.shape[:2]

        with open(pkl_path, 'rb') as f:
            cached = pickle.load(f)

        # Resize SAM masks
        sam_masks = []
        for entry in cached:
            seg = entry['segmentation']
            if seg.shape != (H, W):
                seg = np.array(Image.fromarray(seg.astype(np.uint8) * 255)
                               .resize((W, H), Image.NEAREST)) > 127
            sam_masks.append(seg.astype(bool))

        if not sam_masks:
            continue

        # Classify masks
        crops = [crop_mask(photo_np, m) for m in sam_masks]
        valid_idx = [i for i, c in enumerate(crops) if c is not None]
        if not valid_idx:
            continue

        imgs = torch.stack([transform(crops[i]) for i in valid_idx]).to(device)
        with torch.no_grad():
            logits = model(imgs)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            leaf_logits = logits[:, leaf_indices]
            softmax_scores = F.softmax(leaf_logits, dim=1).cpu().numpy()

        all_scores = [None] * len(sam_masks)
        for pos, i in enumerate(valid_idx):
            all_scores[i] = softmax_scores[pos]

        valid_masks  = [sam_masks[i]  for i in valid_idx]
        valid_scores = [all_scores[i] for i in valid_idx]

        label_map, _ = reconcile(valid_masks, valid_scores, H, W, args.conf_threshold)

        # Build GT pixel label map
        gt_map = build_gt_label_map(group, repo_root, H, W)

        # Evaluate on annotated pixels only
        annotated = gt_map >= 0
        n_annotated = annotated.sum()
        if n_annotated == 0:
            continue

        gt_pixels   = gt_map[annotated]
        pred_pixels = label_map[annotated]

        correct = (gt_pixels == pred_pixels).sum()
        total_annotated += n_annotated
        total_correct   += correct

        # Fill confusion matrix
        for gt_c in range(NUM_CLASSES):
            gt_mask_c = annotated & (gt_map == gt_c)
            if not gt_mask_c.any():
                continue
            preds_for_c = label_map[gt_mask_c]
            for pred_c in range(NUM_CLASSES):
                confusion[gt_c, pred_c] += (preds_for_c == pred_c).sum()

        photos_done += 1
        if photos_done % 20 == 0:
            running_acc = total_correct / max(total_annotated, 1)
            print(f"  {photos_done} photos | running pixel acc: {running_acc:.4f}")

    # ── Summary metrics ────────────────────────────────────────────────────────
    pixel_acc = total_correct / max(total_annotated, 1)

    # Per-class IoU
    iou_per_class = []
    acc_per_class = []
    class_names_present = []
    for c in range(NUM_CLASSES):
        tp = confusion[c, c]
        fn = confusion[c, :].sum() - tp   # GT=c, pred!=c
        fp = confusion[:, c].sum() - tp   # pred=c, GT!=c
        n_gt = confusion[c, :].sum()
        if n_gt == 0:
            continue
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        acc = tp / n_gt
        iou_per_class.append(iou)
        acc_per_class.append(acc)
        class_names_present.append(CATEGORIES[c])

    miou = np.mean(iou_per_class)
    mean_class_acc = np.mean(acc_per_class)

    print(f"\n{'='*55}")
    print(f"Photos evaluated:    {photos_done}")
    print(f"Annotated pixels:    {total_annotated:,}")
    print(f"Pixel accuracy:      {pixel_acc:.4f}  ({pixel_acc*100:.2f}%)")
    print(f"Mean class accuracy: {mean_class_acc:.4f}  ({mean_class_acc*100:.2f}%)")
    print(f"mIoU:                {miou:.4f}  ({miou*100:.2f}%)")
    print(f"\n{'Class':<16} {'IoU':>8} {'ClassAcc':>10} {'GT pixels':>12}")
    print("-" * 50)
    rows = sorted(zip(iou_per_class, acc_per_class, class_names_present,
                      [confusion[CATEGORIES.index(n), :].sum() for n in class_names_present]),
                  key=lambda x: -x[3])
    for iou, acc, name, n_gt in rows:
        print(f"{name:<16} {iou:>8.4f} {acc:>10.4f} {n_gt:>12,}")


if __name__ == '__main__':
    main()
