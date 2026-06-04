"""
Reconcile SAM proposals + HGNN classifications into coherent pixel-level segmentation maps.

For each pixel:
  - Collect all SAM masks covering it
  - Pick the one where HGNN softmax confidence is highest
  - Assign that label

Uncovered pixels (no SAM mask): filled via nearest-neighbor from labeled pixels.

Outputs per photo:
  - side-by-side PNG: original | SAM coverage | reconciled segmentation | GT overlay
  - summary stats: coverage, per-class pixel counts
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from torchvision import transforms as T
import pandas as pd
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
NUM_CLASSES = len(CATEGORIES)
IMAGENET_MEAN = (123, 116, 103)

# Distinct color palette for 23 categories (RGB)
PALETTE = [
    (165, 42,  42),   # brick       - brown
    (255, 0,   0),    # carpet      - red
    (255, 165, 0),    # ceramic     - orange
    (255, 20,  147),  # fabric      - deep pink
    (0,   128, 0),    # foliage     - green
    (255, 215, 0),    # food        - gold
    (135, 206, 235),  # glass       - sky blue
    (255, 192, 203),  # hair        - pink
    (139, 69,  19),   # leather     - saddle brown
    (192, 192, 192),  # metal       - silver
    (200, 200, 255),  # mirror      - light blue
    (128, 128, 128),  # other       - gray
    (144, 238, 144),  # painted     - light green
    (255, 255, 224),  # paper       - light yellow
    (0,   191, 255),  # plastic     - deep sky blue
    (169, 169, 169),  # polishedstone - dark gray
    (255, 228, 196),  # skin        - bisque
    (70,  130, 180),  # sky         - steel blue
    (112, 128, 144),  # stone       - slate gray
    (210, 180, 140),  # tile        - tan
    (255, 182, 193),  # wallpaper   - light pink
    (0,   0,   255),  # water       - blue
    (139, 90,  43),   # wood        - dark wood
]


# ── HGNN loader ───────────────────────────────────────────────────────────────

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


# ── Crop a SAM mask ────────────────────────────────────────────────────────────

def crop_mask(photo_np, mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return None
    r0, r1, c0, c1 = rows[0], rows[-1]+1, cols[0], cols[-1]+1
    crop = photo_np[r0:r1, c0:c1].copy()
    m = mask[r0:r1, c0:c1]
    for ch, mv in enumerate(IMAGENET_MEAN):
        crop[:, :, ch][~m] = mv
    return Image.fromarray(crop)


# ── Reconcile masks into a pixel map ──────────────────────────────────────────

OTHER_IDX = CATEGORIES.index('other')  # 11


def reconcile(sam_masks, softmax_scores, H, W, conf_threshold=0.3):
    """
    sam_masks:      list of [H, W] bool arrays
    softmax_scores: list of [NUM_CLASSES] float arrays (HGNN softmax per mask)
    conf_threshold: masks below this max-softmax are labeled 'other'

    Uncovered pixels (no SAM mask) are labeled 'other' — SAM is unlikely
    to produce proposals for genuinely ambiguous 'other' regions.

    Returns:
      label_map: [H, W] int array
      conf_map:  [H, W] float array, max confidence at each pixel
    """
    label_map = np.full((H, W), OTHER_IDX, dtype=np.int32)
    conf_map  = np.zeros((H, W), dtype=np.float32)

    for mask, scores in zip(sam_masks, softmax_scores):
        conf = float(scores.max())
        pred = int(scores.argmax()) if conf >= conf_threshold else OTHER_IDX
        # Update pixels where this mask is more confident than current assignment
        update = mask & (conf > conf_map)
        label_map[update] = pred
        conf_map[update]  = conf

    return label_map, conf_map


# ── Render colored segmentation ────────────────────────────────────────────────

def colorize(label_map, alpha=0.6):
    H, W = label_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cat_idx, color in enumerate(PALETTE):
        rgb[label_map == cat_idx] = color
    return Image.fromarray(rgb)


def overlay_on_photo(photo_np, label_map, alpha=0.5):
    seg_rgb = np.array(colorize(label_map), dtype=np.float32)
    photo_f = photo_np.astype(np.float32)
    blended = (alpha * seg_rgb + (1 - alpha) * photo_f).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def draw_gt_overlay(photo_np, gt_masks, gt_labels, H, W, repo_root):
    """Draw GT segment outlines on the photo."""
    img = Image.fromarray(photo_np.copy())
    draw = ImageDraw.Draw(img)
    for mask_path, label_idx in zip(gt_masks, gt_labels):
        p = Path(mask_path)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            continue
        mask = np.array(Image.open(p))
        if mask.shape != (H, W):
            mask = np.array(Image.fromarray(mask.astype(np.uint8)*255)
                            .resize((W, H), Image.NEAREST)) > 127
        # Draw boundary of GT mask
        color = PALETTE[label_idx % len(PALETTE)]
        # Find boundary pixels
        from scipy.ndimage import binary_erosion
        boundary = mask & ~binary_erosion(mask)
        ys, xs = np.where(boundary)
        for y, x in zip(ys[::3], xs[::3]):  # subsample for speed
            draw.point((x, y), fill=color)
    return img


def make_legend(categories, palette, cell_size=20):
    cols = 4
    rows = (len(categories) + cols - 1) // cols
    W = cols * 120
    H = rows * cell_size + 10
    img = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, (cat, color) in enumerate(zip(categories, palette)):
        row, col = divmod(i, cols)
        x = col * 120 + 5
        y = row * cell_size + 5
        draw.rectangle([x, y, x+cell_size-2, y+cell_size-2], fill=color)
        draw.text((x+cell_size+2, y+2), cat, fill=(0,0,0))
    return img


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sam-dir', default='out/sam_eval_top200_matador_auto_vit_b')
    p.add_argument('--hgnn-run', default='runs/minc_hgnn/20260603_092440')
    p.add_argument('--taxonomy', default='taxonomy/assets/minc-taxonomy.json')
    p.add_argument('--out-dir', default='runs/reconciled_segmentation')
    p.add_argument('--n-photos', type=int, default=20, help='Number of photos to visualize')
    p.add_argument('--conf-threshold', type=float, default=0.3,
                   help='Min max-softmax to assign a class; below → other')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    repo_root = Path(__file__).parent.parent
    sam_dir = repo_root / args.sam_dir
    out_dir = repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HGNN...")
    model, leaf_indices, leaf_to_cat = load_hgnn(
        repo_root / args.hgnn_run, repo_root / args.taxonomy, device
    )

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])

    cache_dir = sam_dir / 'cache' / 'auto_masks'

    # Load GT data
    df = pd.read_csv(sam_dir / 'per_segment_metrics.csv')
    by_photo = {pid: grp for pid, grp in df.groupby('photo_id')}

    # Get photo list from cache
    pkl_files = sorted(cache_dir.glob('*.pkl'))[:args.n_photos]

    stats = []

    for pkl_path in pkl_files:
        # Parse photo_id from filename: 000000110__vit_b__825x550.pkl
        photo_id_str = pkl_path.stem.split('__')[0]
        photo_id_int = int(photo_id_str)

        with open(pkl_path, 'rb') as f:
            cached = pickle.load(f)

        # Find photo path from GT data or infer
        photo_path = None
        if photo_id_int in by_photo:
            pp = by_photo[photo_id_int]['photo_path'].iloc[0]
            photo_path = Path(pp) if Path(pp).is_absolute() else repo_root / pp
        if photo_path is None or not photo_path.exists():
            continue

        photo_np = np.array(Image.open(photo_path).convert('RGB'))
        H, W = photo_np.shape[:2]

        # Resize SAM masks to photo dimensions
        sam_masks = []
        for entry in cached:
            seg = entry['segmentation']
            if seg.shape != (H, W):
                seg = np.array(Image.fromarray(seg.astype(np.uint8)*255)
                               .resize((W, H), Image.NEAREST)) > 127
            sam_masks.append(seg.astype(bool))

        # Classify all masks with HGNN in one batch
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
            softmax_scores = F.softmax(leaf_logits, dim=1).cpu().numpy()  # [N, NUM_CLASSES]

        # Map back to full mask list
        all_scores = [None] * len(sam_masks)
        for pos, i in enumerate(valid_idx):
            all_scores[i] = softmax_scores[pos]

        valid_masks  = [sam_masks[i]  for i in valid_idx]
        valid_scores = [all_scores[i] for i in valid_idx]

        # Reconcile into pixel map
        label_map, conf_map = reconcile(valid_masks, valid_scores, H, W, args.conf_threshold)

        # Coverage stats
        coverage = (conf_map > 0).mean()
        pred_labels, counts = np.unique(label_map, return_counts=True)
        top_cats = sorted(zip(counts, pred_labels), reverse=True)[:5]

        stats.append({
            'photo_id': photo_id_int,
            'n_masks': len(valid_masks),
            'coverage_pct': float(coverage * 100),
            'top_categories': [(CATEGORIES[c], int(n)) for n, c in top_cats if c >= 0],
        })

        # ── Visualize ──────────────────────────────────────────────────────────
        seg_overlay = overlay_on_photo(photo_np, label_map, alpha=0.55)
        seg_clean   = colorize(label_map)
        conf_img    = Image.fromarray((conf_map * 255).astype(np.uint8).repeat(3).reshape(H, W, 3))

        # GT overlay if available
        if photo_id_int in by_photo:
            grp = by_photo[photo_id_int]
            gt_overlay = draw_gt_overlay(
                photo_np,
                grp['mask_path'].tolist(),
                grp['label_index'].tolist(),
                H, W, repo_root,
            )
        else:
            gt_overlay = Image.fromarray(photo_np)

        # Assemble 2×2 grid: original | segmentation overlay | clean seg | GT outlines
        cell_w, cell_h = W, H
        grid = Image.new('RGB', (cell_w * 2, cell_h * 2 + 5))
        grid.paste(Image.fromarray(photo_np), (0, 0))
        grid.paste(seg_overlay, (cell_w, 0))
        grid.paste(seg_clean, (0, cell_h + 5))
        grid.paste(gt_overlay, (cell_w, cell_h + 5))

        # Add simple labels
        draw = ImageDraw.Draw(grid)
        for text, pos in [
            ("Original", (5, 5)),
            ("HGNN Segmentation", (cell_w + 5, 5)),
            ("Label Map", (5, cell_h + 10)),
            ("GT Outlines", (cell_w + 5, cell_h + 10)),
        ]:
            draw.text((pos[0]+1, pos[1]+1), text, fill=(0,0,0))
            draw.text(pos, text, fill=(255,255,255))

        grid.save(out_dir / f'{photo_id_str}.png')
        print(f"  {photo_id_str}: {len(valid_masks)} masks, {coverage*100:.1f}% covered, "
              f"top: {', '.join(CATEGORIES[int(c)] for _,c in top_cats[:3] if c >= 0)}")

    # Save legend
    make_legend(CATEGORIES, PALETTE).save(out_dir / 'legend.png')

    # Save stats
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    avg_coverage = np.mean([s['coverage_pct'] for s in stats])
    print(f"\nDone. {len(stats)} photos processed.")
    print(f"Average SAM pixel coverage: {avg_coverage:.1f}%")
    print(f"Visualizations saved to: {out_dir}")


if __name__ == '__main__':
    main()
