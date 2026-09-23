# What the RSNA 2024 Lumbar Spine winners did, and what it means for the knee

Studied 2026-09-22 before writing any knee code. Covers places 1–9 of
[RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)
(private LB 0.389–0.404, metric = weighted log loss over 25 targets × 3 severities).
Code is in `code/`, notebooks are also converted to `*.nb.py` for grepping.

## The one difference that changes everything

**Every top-9 lumbar solution was "localise, then crop, then classify"** — and they could do that because
the lumbar data shipped `train_label_coordinates.csv` (x, y, slice for every finding). Stage 1 predicted
those points; stage 2 classified small crops around them.

**The knee data has no coordinates.** Only study-level 0/1 labels, a free-text report, and per-series
plane / fluid-sensitive / fat-suppressed flags. So the dominant lumbar recipe does not port directly.
The transferable ideas are the *crop-free* ones:

- **8th (K_mat):** no cropping — a detector heatmap is used as attention weights over the feature map.
- **7th (heng, one-stage):** "point-masking and pool" — predict a heatmap, multiply it into per-pixel
  grade probabilities, sum. Beat his team's two-stage baseline for NFN.
- **Attention/MIL pooling over whole slices** — every team used it on crops; for us it runs on full
  (or joint-centred) slices and the attention weights *become* the weak localiser.

Cheap ways to get some localisation back (try only after a crop-free baseline works):
1. Centre-crop to the joint (knee is roughly centred in-plane; a fixed central crop at ~60–70% FOV is a strong prior).
2. Crop in **millimetres, not pixels**, using `PixelSpacing` (9th: fixed 50 mm ROIs) — scanners vary a lot.
3. Hand-annotate ~100 studies with a couple of points (joint centre, medial/lateral compartment) and train
   a tiny keypoint model. 3rd/4th/5th all say a small high-quality coordinate set carried their stage 1.

## Patterns shared by nearly all winners (port these)

| Pattern | Who | Evidence |
|---|---|---|
| **2D backbone per slice → sequence head → attention pooling.** 3D CNNs lost. | all | 3rd lists 3D-CNN under "didn't work"; 5th: 2D+sequence beat 3D U-Net on z accuracy |
| **Small/medium backbones.** Bigger did not help. | 1st: `convnext_small` > base > large; ViTs worse. 3rd: resnet18 → convnext-tiny; 9th: resnet18/convnext-nano/swin-tiny | 1st, 3rd, 5th ("bigger backbone didn't work") |
| **Short training.** 3–20 epochs. Long runs overfit. | 1st: 7 ep; 9th: 3–5 ep; 3rd: 10–20 ep | 1st and 3rd both list "long epochs" as failed |
| **Per-series percentile normalisation** (clip 1st–99th pct, scale to 0–1 / uint8) | 1st, 2nd (Ian, Bartley), 9th | ubiquitous |
| **Sort slices by `ImagePositionPatient`, not InstanceNumber**; merge split series; resize mismatched slices | 2nd (Ian, Bartley), 4th | Ian: some series are split across folders |
| **Cache preprocessed volumes** (uint8 PNG / npy / HDF5 at 256²) and train from cache | 2nd, 4th, 6th | 1st read raw DICOM on the fly on Colab — slow but it still won |
| **Treat every position as a separate sample** (each level / side through one shared model) | 3rd (5–10× data), 9th, 2nd | 3rd credits much of its gain to this |
| **Auxiliary losses** | 1st: slice-position aux head, val loss 0.262 → 0.252; 4th: aux head on pre-transformer features "key technique"; 3rd: other conditions + level | strongest single trick in the write-ups |
| **Multi-plane fusion by concatenating slice tokens → transformer** (per-plane encoder or shared encoder + per-plane neck, per-plane positional encoding) | 3rd, 4th, 6th | 3rd: sag T1 + sag T2 + ax T2 together beat every single-view model |
| **Missing plane = zeros + mask, and randomly drop a view (10%) in training** | 6th | robustness to studies without a plane |
| **Slice-position jitter** (shift the centre slice ±1–2 in training) | 1st ("crucial"), 2nd (Ian: 27 crops/target) | makes stage 2 tolerant of stage 1 errors |
| **Mixing augmentations** | 5th: CutMix p=1.0 was their biggest AUC gain; 2nd: manifold mixup on embeddings + cutmix on sequences; 4th: random token masking | overfitting is fast |
| **Mild geometric + brightness/contrast aug**; no heavy tricks | all | ShiftScaleRotate(±10–30°), RandomBrightnessContrast, blur, CoarseDropout |
| **Label-noise handling** | 2nd: drop samples where OOF ensemble disagrees with label by ≥0.8 → +1% LB; 3rd: hand-corrected coords; 8th: built a relabel tool | label noise was real in lumbar; ours will be worse (report-derived) |
| **5-fold CV grouped by study, multilabel-stratified** (`iterstrat.MultilabelStratifiedKFold`) | 1st, 3rd, 9th | 6th: "we trusted CV more than public LB" |
| **Ensembles of diverse small models** | 3rd: 30 models; 4th/6th: MLP/LightGBM stacking | but see efficiency track below |

## Specific code worth reusing (all MIT/Apache unless noted)

- `code/01_wadakoki/colab/12_*.nb.py` — **1st place** `LSTMMIL` (bi-LSTM → attention pool + aux
  attention head) and per-plane `convnext_small` encoders with `in_chans=1`. The aux target is "which
  of the 5 slices is the annotated one". *(No license file — read, don't copy.)*
- `code/02_bartley/src/models/attn.py` — clean `Attention` / `AttentionLSTM` with **padding masks**
  (needed: our series have variable slice counts). `cnn25d_sagittal_1head.py` shows sequence cutmix +
  manifold mixup. *(No license file.)*
- `code/02_bartley/src/preprocess/rsna.py` and `code/02_ianpan/skp/etl/0000_convert_dicoms_to_pngs.py` —
  DICOM stack loading: sort by IPP, pad mismatched shapes, percentile → uint8. **Ian's is MIT — start here.**
- `code/02_ianpan/skp/etl/26_*`, `28_*` + `skp/models/dist_transformer_cls.py` — Ian Pan **extracts
  CNN features once, then trains a transformer on the cached features**. This is exactly our
  "cache features to save GPU hours" plan, done by a radiologist on a 2nd-place team.
- `code/04_tattaka/src/stage2/exp107/train.py` — multi-plane transformer with shared encoder,
  per-plane necks, sinusoidal positional encoding, padding + random token masking, `AttentionPooling`
  heads and the aux head. **Closest architecture to what the knee needs.**
- `code/04_tattaka/input/save_volume_256x256_v2.py` — `standardize_pixel_array` (handles
  PixelRepresentation, RescaleSlope/Intercept, window) and caching volumes to HDF5.
- `code/03_moyashii/src/stage2/moyashii/source/models/rsna2024_attention_net.py` — per-plane
  backbones + GeM pooling + one transformer block + multi-sample dropout.
- `code/09_narai/rsna2024/utils/utils.py` — `dcm_affine` and cross-plane projection (sagittal point →
  nearest axial slice) from IPP/IOP/PixelSpacing. Small and correct; reuse for any cross-plane mapping.
- `code/05_siwooyong/notebook/stage2_train_step2.nb.py` — CutMix implementation for slice stacks.
  *(No license file.)*

## Warnings the winners learned the hard way

- **Left/right flip bug (7th place, heng):** flip augmentation without swapping left/right labels —
  "50% of the time the left is mistakenly labeled as right." It cost them score and they couldn't
  retrain in time. **For the knee this is medial/lateral: never flip coronal/axial without swapping
  Medial↔Lateral labels (Meniscus, OA), or standardise laterality first.**
- The competing-losses problem (heng): localisation and grading losses fought each other in one-stage
  nets. Weight the aux/heatmap loss down or warm it up separately.
- Metric-specific post-processing (2nd: ×1.25 on max severe; 3rd: temperature 0.91; 5th: head-only
  fine-tune for weighted loss) **does not transfer** — our metric is per-label AUC, which ignores any
  monotonic rescaling. Skip all of it. Rank-average ensembles instead.
- Hardware: 1st place trained everything on **Colab T4 high-mem**. 2nd/3rd/7th used A100/A6000 48 GB
  and noted some models need ≥34–48 GB. A small-model plan on Kaggle GPUs is realistic.

## Where the knee is harder than lumbar

- **Runtime:** lumbar test was small; knee test is ~1,300 studies in ≤9 h and is scored on runtime
  for the efficiency track. None of the lumbar inference notebooks did anything special for speed
  (joblib `Parallel` DICOM reads, `num_workers`, `autocast`). We will need to: read only the series we
  use, subsample slices, decode in parallel, and run one small model.
- **Labels:** lumbar had radiologist labels for every study (noisy, but present). We have gold labels
  for a *subset* and report-derived labels for the rest → label cleaning (2nd place's OOF filter) matters more.
- **Series selection:** lumbar had 3 fixed series types. Knee studies have a variable set, described
  by `Anatomical_Plane` × `Fluid_Sensitive` × `Fat_Suppression` flags. We must pick "best available
  per slot" and mask the rest.

## Resulting baseline blueprint (v0)

1. Preprocess once on Kaggle: per study, pick up to 4 slots — sagittal, coronal and axial fluid-sensitive
   (prefer fat-sat), plus one non-fluid-sensitive T1 series for Fracture/osteophytes (see docs/labels.md). Sort by IPP, percentile-normalise per series, resize to
   256², sample a fixed number of slices (e.g. 24), store uint8. Missing slot → flag.
2. Model: shared `efficientnet_b0`/`convnext_nano` (`in_chans=1`, or 3 neighbouring slices as channels),
   per-plane linear neck, per-plane positional encoding, 2-layer transformer (d=256) over all tokens with
   a padding mask, 12 attention-pooling heads (one per finding) + an aux head on pre-transformer features.
3. Train 5 folds, ~8–10 epochs, AdamW, cosine with warmup, BCE, CutMix/token masking, mild aug, no
   uncontrolled L/R flips. Validate on gold labels only, macro AUC + per-label AUC.
4. Then: feature caching for fast head experiments, OOF label cleaning of report labels, joint-centred mm crops.
