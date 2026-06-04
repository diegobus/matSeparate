# Context Dataset Study Plan

## Infrastructure

S3 bucket:

- `s3://matseparate-context-dataset-251995574236-us-west-2`
- Region: `us-west-2`
- Public access: blocked
- Server-side encryption: AES256
- Prefixes:
  - `raw/`
  - `manifests/`
  - `processed/`
  - `checkpoints/`

EC2 worker preparation:

- Target instance: `g4dn.xlarge`
  - 1 NVIDIA T4
  - 4 vCPU
  - 16 GiB RAM
- Launch template: `matseparate-context-t4`
- Launch template id: `lt-047f7c591c4445fc3`
- AMI: AWS Deep Learning PyTorch GPU Ubuntu 22.04, `ami-0ca70308d230e8a6e`
- Root disk: 1 TB encrypted gp3
- SSH key: `us-west-2-rcatullo`
- Security group: `sg-06d21a4190c197a33`
- IAM instance profile: `matseparate-gpu-worker-profile`
  - scoped to the context dataset bucket

Current blocker:

- The AWS account has `0` vCPU quota for On-Demand G/VT instances in `us-west-2`.
- Quota increase requested:
  - quota: `Running On-Demand G and VT instances`
  - quota code: `L-DB2E81BA`
  - desired value: `4`
  - status at capture: `CASE_OPENED`

Launch command after quota approval:

```bash
aws ec2 run-instances \
  --region us-west-2 \
  --launch-template LaunchTemplateName=matseparate-context-t4,Version='$Latest' \
  --count 1
```

## Dataset Ingestion

Once the dataset URL/source is known:

1. Upload the original archive or mirrored files to:

```bash
aws s3 sync /path/to/dataset s3://matseparate-context-dataset-251995574236-us-west-2/raw/
```

or, for a single archive:

```bash
aws s3 cp dataset.tar s3://matseparate-context-dataset-251995574236-us-west-2/raw/
```

2. Generate a manifest under `manifests/` with at least:

- image id
- image path
- material label
- patch crop coordinates, if available
- larger-context crop coordinates, if available
- split id
- source scene/photo id

3. Keep all splits grouped by source scene/photo id to avoid leakage between patch and context examples from the same original image.

4. Materialize training-ready derivatives under `processed/`, but keep `raw/` immutable.

## Study Design

Core claim:

- A material classifier trained/evaluated with larger visual context performs better than one using only small material swatches/patches.

The important part is to make the comparison controlled:

- Same labels.
- Same train/val/test split.
- Same backbone.
- Same optimizer, augmentations, epochs, and resolution policy where possible.
- Same number of samples.
- Only the visible input context changes.

Recommended conditions:

1. Patch-only baseline:
   - tight material patch/swatch crop
   - equivalent to the existing MINC/Matador patch-style classifier setup

2. Context crop:
   - larger crop centered on the same material region
   - includes surrounding object/scene cues

3. Context plus mask, optional:
   - larger crop with the target material mask/channel or outside-mask attenuation
   - tests whether context helps without making the classifier ignore the target material

4. Full image plus region prompt, optional:
   - image-level context with target bbox/mask metadata
   - useful if the downstream pipeline will classify proposed regions from segmentation

Metrics:

- top-1 accuracy
- macro accuracy / balanced accuracy
- CHD
- Hier@d2
- per-class deltas, especially visually ambiguous materials
- calibration metrics if using probabilities downstream

Analysis tables:

- patch-only vs context on the same test examples
- per-class improvement/degradation
- confusion matrix changes
- performance by patch size / context ratio
- performance by label hierarchy depth or superclass

Expected risk:

- Context may improve accuracy by exploiting object/scene priors rather than material texture.
- That is acceptable if stated clearly, but we should include one ablation that masks or attenuates non-target pixels to separate material appearance from contextual priors.

## Immediate Next Steps

1. Provide the dataset URL/source.
2. Upload or mirror the dataset into `raw/`.
3. Wait for G/VT quota approval, or provide an AWS profile/account that already has T4 quota.
4. Launch the prepared `matseparate-context-t4` worker.
5. Build the manifest and leakage-safe splits.
6. Run the patch-only baseline and context-crop baseline with identical training settings.
7. Add the context-vs-patch comparison table to the project report.
