"""
Idea 1: Hierarchy-guided SAM mask merging.

For each image:
  1. Run HGNN on every SAM auto mask crop.
  2. For each mask get: leaf prediction + parent node.
  3. Find spatially adjacent mask pairs (dilated overlap).
  4. If two adjacent masks agree on parent but differ at leaf → merge (union).
  5. Re-classify merged mask.
  6. Compare recall@IoU≥0.5 against GT before and after merging.

Key hypothesis: merging fixes the case where SAM splits one material region into
multiple proposals. Hierarchy agreement is the signal that two proposals belong
together — a flat classifier's predictions aren't stable enough to use for merging.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
import pandas as pd
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from gnn_classifier.hgnn import HGNN
from taxonomy.tree import get_taxonomy
import networkx as nx

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES = [
    'brick','carpet','ceramic','fabric','foliage','food','glass','hair',
    'leather','metal','mirror','other','painted','paper','plastic',
    'polishedstone','skin','sky','stone','tile','wallpaper','water','wood',
]
IMAGENET_MEAN = (123, 116, 103)


# ── Build parent lookup ────────────────────────────────────────────────────────

def build_parent_lookup(taxonomy_path):
    """Returns dict: leaf_category_name → parent_node_name"""
    g = get_taxonomy(taxonomy_path)
    parent_of = {}
    for node in g.nodes:
        if g.out_degree(node) == 0:  # leaf
            preds = list(g.predecessors(node))
            parent_of[node] = preds[0] if preds else 'root'
    return parent_of


# ── HGNN loader (same as eval_sam_classification) ────────────────────────────

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


# ── Crop a SAM mask from a photo ─────────────────────────────────────────────

def crop_mask(photo_np, mask, image_size=224):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    crop = photo_np[r0:r1, c0:c1].copy()
    m = mask[r0:r1, c0:c1]
    # Gray out pixels outside mask
    for ch, mv in enumerate(IMAGENET_MEAN):
        crop[:, :, ch][~m] = mv
    return Image.fromarray(crop)


# ── Classify a batch of crops ─────────────────────────────────────────────────

def classify_crops(crops, model, leaf_indices, leaf_to_cat, transform, device):
    if not crops:
        return []
    imgs = torch.stack([transform(c) for c in crops]).to(device)
    with torch.no_grad():
        logits = model(imgs)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        leaf_logits = logits[:, leaf_indices]
        pred_pos = leaf_logits.argmax(1)
        preds = leaf_to_cat[pred_pos].cpu().tolist()
    return preds


# ── IoU between two binary masks ──────────────────────────────────────────────

def compute_iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union > 0 else 0.0


# ── Find adjacent mask pairs via dilation ─────────────────────────────────────

def find_adjacent_pairs(masks, dilation_px=15):
    """
    O(N²) bounding-box adjacency — no pixel-level ops, no dilation.
    Two masks are adjacent if their bounding boxes are within dilation_px pixels.
    Some false positives are OK: the parent-agreement filter handles over-merging.
    """
    bboxes = []
    for m in masks:
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        if len(rows) == 0:
            bboxes.append(None)
        else:
            bboxes.append((rows[0], rows[-1], cols[0], cols[-1]))

    adjacent = set()
    pad = dilation_px
    for i in range(len(masks)):
        if bboxes[i] is None:
            continue
        r0i, r1i, c0i, c1i = bboxes[i]
        for j in range(i + 1, len(masks)):
            if bboxes[j] is None:
                continue
            r0j, r1j, c0j, c1j = bboxes[j]
            if (r0i - pad <= r1j and r1i + pad >= r0j and
                    c0i - pad <= c1j and c1i + pad >= c0j):
                adjacent.add((i, j))
    return adjacent


# ── Main evaluation ───────────────────────────────────────────────────────────

def load_cached_masks(cache_dir, photo_id, H, W, model_type='vit_b'):
    """Load cached SAM masks, trying photo_id with zero-padding and dimension suffix."""
    pid = f"{int(photo_id):09d}"
    pattern = f"{pid}__{model_type}__{H}x{W}.pkl"
    path = cache_dir / pattern
    if not path.exists():
        matches = list(cache_dir.glob(f"{pid}__{model_type}__*.pkl"))
        if not matches:
            return None
        path = matches[0]
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_gt_mask(mask_path, repo_root, H, W):
    """Load a GT segment mask, resolving relative paths against repo_root."""
    p = Path(mask_path)
    if not p.is_absolute():
        p = repo_root / p
    mask = np.array(Image.open(p))
    if mask.shape != (H, W):
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255)
                        .resize((W, H), Image.NEAREST)) > 127
    return mask.astype(bool)


def evaluate(sam_dir, hgnn_run, taxonomy_path, iou_threshold, dilation_px, device):
    sam_dir = Path(sam_dir)
    repo_root = Path(__file__).parent.parent
    parent_of = build_parent_lookup(taxonomy_path)

    # Load ALL GT segment data (not pre-filtered) so total_gt is correct
    df_all = pd.read_csv(sam_dir / 'per_segment_metrics.csv')
    total_gt = len(df_all)
    # Matched subset: GT segments that SAM found at all (pre-computed best_iou)
    df = df_all[df_all['best_iou'] >= iou_threshold].copy()
    print(f"Total GT segments: {total_gt} | SAM-matched (IoU≥{iou_threshold}): {len(df)}")

    # Load HGNN
    print("Loading HGNN...")
    model, leaf_indices, leaf_to_cat = load_hgnn(hgnn_run, taxonomy_path, device)

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)
    ])

    cache_dir = sam_dir / 'cache' / 'auto_masks'

    # Metrics
    before_recall = before_correct = 0
    after_recall  = after_correct  = 0
    merge_events  = 0

    by_photo = df.groupby('photo_id')
    n_photos = len(by_photo)

    for photo_num, (photo_id, group) in enumerate(by_photo, 1):
        if photo_num % 20 == 0:
            print(f"  Progress: {photo_num}/{n_photos} photos | merges so far: {merge_events}")
        photo_path = group['photo_path'].iloc[0]
        if not Path(photo_path).is_absolute():
            photo_path = repo_root / photo_path
        photo_np = np.array(Image.open(photo_path).convert('RGB'))
        H, W = photo_np.shape[:2]

        cached = load_cached_masks(cache_dir, photo_id, H, W)
        if cached is None:
            continue

        # Load and resize all SAM masks to photo dimensions
        sam_masks = []
        for entry in cached:
            seg = entry['segmentation']
            if seg.shape != (H, W):
                seg = np.array(Image.fromarray(seg.astype(np.uint8) * 255)
                               .resize((W, H), Image.NEAREST)) > 127
            sam_masks.append(seg.astype(bool))

        if not sam_masks:
            continue

        # Step 1: Classify ALL SAM masks with HGNN
        crops = [crop_mask(photo_np, m) for m in sam_masks]
        valid = [i for i, c in enumerate(crops) if c is not None]
        preds_valid = classify_crops([crops[i] for i in valid],
                                     model, leaf_indices, leaf_to_cat, transform, device)
        preds = [None] * len(sam_masks)
        for i, pred in zip(valid, preds_valid):
            preds[i] = pred

        parents = [parent_of.get(CATEGORIES[p], 'root') if p is not None else None
                   for p in preds]

        # Step 2: BEFORE — use pre-computed best_mask_index + best_iou from CSV
        for _, row in group.iterrows():
            best_i = int(row['best_mask_index'])
            best_iou = float(row['best_iou'])
            if best_iou >= iou_threshold and best_i < len(sam_masks) and preds[best_i] is not None:
                before_recall += 1
                if preds[best_i] == row['label_index']:
                    before_correct += 1

        # Step 3: Build merged mask set.
        # Merge adjacent masks that agree on parent node prediction — these likely
        # represent SAM splitting one material region into multiple proposals.
        adj_pairs = find_adjacent_pairs(sam_masks, dilation_px)
        merged_masks = list(sam_masks)
        merged_preds = list(preds)
        merged_set = set()  # track which originals were merged

        for i, j in adj_pairs:
            if merged_preds[i] is None or merged_preds[j] is None:
                continue
            # Same parent prediction → likely the same material region, merge
            if parents[i] == parents[j]:
                merged = merged_masks[i] | merged_masks[j]
                crop = crop_mask(photo_np, merged)
                if crop is None:
                    continue
                new_pred = classify_crops([crop], model, leaf_indices, leaf_to_cat, transform, device)
                if new_pred:
                    merged_masks.append(merged)
                    merged_preds.append(new_pred[0])
                    merged_set.add(i)
                    merged_set.add(j)
                    merge_events += 1

        # Step 4: AFTER — evaluate ALL GT segments (including the 30% SAM missed)
        # Load all GT masks for this photo from MINC-S (they share the same photo)
        # We must iterate df_all by photo to get all GT rows, not just the matched ones
        all_photo_rows = df_all[df_all['photo_id'] == photo_id]
        for _, row in all_photo_rows.iterrows():
            gt_mask = load_gt_mask(row['mask_path'], repo_root, H, W)
            best_iou_after, best_i_after = 0.0, -1
            for i, sm in enumerate(merged_masks):
                iou = compute_iou(gt_mask, sm)
                if iou > best_iou_after:
                    best_iou_after, best_i_after = iou, i
            if best_iou_after >= iou_threshold and best_i_after >= 0 \
                    and merged_preds[best_i_after] is not None:
                after_recall += 1
                if merged_preds[best_i_after] == row['label_index']:
                    after_correct += 1

    print(f"\nDilation: {dilation_px}px | IoU threshold: {iou_threshold}")
    print(f"Total GT: {total_gt} | SAM-matched before: {before_recall} | Merge events: {merge_events}")
    print()
    print(f"{'Metric':<25} {'Before':>10} {'After':>10} {'Delta':>10}")
    print("-" * 55)
    r_before = before_recall / total_gt
    r_after  = after_recall  / total_gt
    a_before = before_correct / max(before_recall, 1)
    a_after  = after_correct  / max(after_recall,  1)
    e2e_before = before_correct / total_gt
    e2e_after  = after_correct  / total_gt
    print(f"{'Recall@IoU':<25} {r_before:>10.4f} {r_after:>10.4f} {r_after-r_before:>+10.4f}")
    print(f"{'Classifier accuracy':<25} {a_before:>10.4f} {a_after:>10.4f} {a_after-a_before:>+10.4f}")
    print(f"{'End-to-end (R×A)':<25} {e2e_before:>10.4f} {e2e_after:>10.4f} {e2e_after-e2e_before:>+10.4f}")

    return {
        'before': {'recall': r_before, 'accuracy': a_before, 'e2e': e2e_before},
        'after':  {'recall': r_after,  'accuracy': a_after,  'e2e': e2e_after},
        'merge_events': merge_events,
        'total_gt': total_gt,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sam-dir', default='out/sam_eval_top200_matador_auto_vit_b')
    p.add_argument('--hgnn-run', default='runs/minc_hgnn/20260603_092440')
    p.add_argument('--taxonomy', default='taxonomy/assets/minc-taxonomy.json')
    p.add_argument('--iou-threshold', type=float, default=0.5)
    p.add_argument('--dilation-px', type=int, default=15,
                   help='Pixel dilation for adjacency detection')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = evaluate(
        args.sam_dir, args.hgnn_run, args.taxonomy,
        args.iou_threshold, args.dilation_px, device,
    )
