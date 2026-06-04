# Current State

Captured: 2026-06-03 America/Los_Angeles / 2026-06-04 UTC

## Local repo

- Path: `/Users/diegobustamante/github/matSeparate`
- Current branch: `matador-c1-pipeline`
- Upstream: `origin/matador-c1-pipeline`
- `git pull --ff-only`: already up to date; fetched new remote branch `origin/sam`
- Untracked local files/directories:
  - `docs/`
  - `t4run`

Remote branches:

- `origin/main` at `b479fd4` - patch class inference api
- `origin/patch_classifier` at `b21285f` - patch classifier training updates
- `origin/matador-c1-pipeline` at `5fe8498` - MINC-2500 evaluation and HGNN fine-tuning pipeline
- `origin/sam` at `d743770` - reproducible SAM MINC-S baseline pipeline

Branch relationship:

- `sam` branches from `main`, not from `matador-c1-pipeline`.
- `matador-c1-pipeline` contains the MINC/HGNN/segmentation stack and tracked MINC processed data.
- `sam` contains the SAM baseline scripts and does not include the `matador-c1-pipeline` additions.
- A direct merge/rebase needs care because `sam` vs `matador-c1-pipeline` is a real feature-branch divergence, not a linear continuation.

## EC2 box

- SSH alias: `cs231n`
- Short wrapper available locally: `t4run`
- Hostname: `ip-172-31-42-16`
- User: `ubuntu`
- Repo path: `/home/ubuntu/matSeparate`
- Current branch on box: `sam`
- Upstream on box: `origin/sam`
- Box remote URL currently embeds a GitHub token; remove/rotate it before sharing logs or committing remote config.

Hardware/runtime:

- GPU: Tesla T4, 15 GiB
- Disk: `/dev/root` 123G total, 64G used, 60G free
- RAM: 15 GiB total, about 10 GiB available
- Active GPU job:
  - `scripts/train_hierseg.py --lam-hier 0.5 --epochs 30 --batch-size 16 --patch-size 112 --runs-dir runs/hierseg_lam05`
  - Run dir: `runs/hierseg_lam05/20260604_003114`
  - At inspection: around epoch 14/30

EC2 repo state:

- No tracked modifications reported.
- Large untracked research workspace, including:
  - `RESEARCH_LOG.md`
  - `models/hierseg.py`
  - `gnn_classifier/dense_hgnn.py`
  - `datasets/minc.py`
  - many new training/eval scripts under `scripts/`
  - `configs/experiments/dense_hgnn_minc.yaml`
  - `taxonomy/assets/minc-taxonomy.json`
  - outputs under `runs/`, `out/`, `checkpoints/`
  - `.claude/` worktrees and research status files

## Key results observed on box

SAM baseline on targeted 200-image MINC-S subset:

- SAM auto proposals:
  - 200 images, 1067 segments
  - mean best IoU 0.661
  - recall@0.50 0.704
  - 63.67 masks/image
  - 4.46 sec/image
- SAM oracle point:
  - mean IoU 0.834
  - mean Dice 0.896
  - 0.010 sec/segment

MINC-S GT segment classification, masked-crop protocol (`n=6917`):

- flat: accuracy 0.5682, CHD 2.1081, Hier@d2 0.6075
- hier: accuracy 0.5554, CHD 2.1574, Hier@d2 0.6014
- hgnn: accuracy 0.6140, CHD 1.8549, Hier@d2 0.6553
- maskedaug: accuracy 0.5729, CHD 2.0197, Hier@d2 0.6228

MINC-S bbox/nomask protocol (`n=6917`):

- flat: accuracy 0.6363, CHD 1.7055, Hier@d2 0.6798
- hier: accuracy 0.6365, CHD 1.7174, Hier@d2 0.6751
- hgnn: accuracy 0.6403, CHD 1.6817, Hier@d2 0.6811

SAM matched segment classification at IoU>=0.50 (`n=751`):

- flat: accuracy 0.5792, CHD 2.1158, Hier@d2 0.6152
- hier: accuracy 0.5806, CHD 2.1145, Hier@d2 0.6218
- maskedaug: accuracy 0.5979, CHD 1.9800, Hier@d2 0.6391
- hgnn: accuracy 0.6591, CHD 1.6858, Hier@d2 0.6844

Research log conclusions:

- HGNN is currently the strongest classifier on SAM-matched segments.
- SAM auto recall is the main end-to-end bottleneck.
- Hierarchy-guided SAM mask merging was negative overall: recall improved, but classification accuracy dropped enough to hurt end-to-end performance.
- Dense HierSeg is the current active direction. The CE-only baseline reached about 51.4% MINC-S accuracy at epoch 30; the hierarchy-loss run is in progress.

## What needs to be done

1. Let the active `hierseg_lam05` run finish, then compare its final MINC-S metrics against the CE-only HierSeg baseline.
2. Preserve the EC2 research work before changing branches:
   - commit the useful code on a new branch, or
   - copy/sync the untracked scripts/logs/results locally, or
   - at minimum create a patch/archive of the untracked files.
3. Remove the embedded GitHub token from the EC2 repo remote and rotate it if it is still valid.
4. Decide integration direction:
   - merge `sam` into `matador-c1-pipeline`, keeping the segmentation/MINC/HGNN stack, or
   - create a new integration branch from `matador-c1-pipeline` and cherry-pick the SAM work plus selected EC2 research files.
5. Add ignored patterns for generated artifacts:
   - `runs/`
   - `out/`
   - `checkpoints/`
   - caches and large model/data artifacts
6. Promote reusable EC2 research code into tracked files with tests or minimal smoke checks.
7. Turn the current results into a concise report table for the project writeup:
   - SAM proposal recall
   - classifier metrics on GT segments
   - classifier metrics on SAM-matched segments
   - end-to-end estimate: SAM recall times classifier accuracy
