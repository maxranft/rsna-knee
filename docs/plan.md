# Plan and decisions so far (2026-09-22, strategy added 2026-09-23)

Carried over from the planning conversation. The research behind it is in
[../reference/STUDY.md](../reference/STUDY.md), [labels.md](labels.md) and [landscape.md](landscape.md).

## Goals
1. **Primary:** a medal on the main leaderboard (macro AUC).
2. **Secondary:** a top-3 finish on the **efficiency track** ($7k / $6k / $5k). Few big teams optimise for it.
3. **Always:** a valid, early submission. Never be without a working notebook.
4. **Learning:** a reusable report → label pipeline and a multi-series MRI pipeline for the portfolio.

Non-goal: out-computing large teams. We have no big-GPU budget, so every choice below trades compute for cleverness.

## Strategy: win on labels, iteration speed and efficiency, not compute

The binding constraint in this competition is **labels, not GPUs**. Most training studies have only a
report. The strategy follows from that. Each item has a check that tells us whether it's working.

| # | Bet | What we do | Success check |
|---|---|---|---|
| **S1** | **Best report-derived labels** | Qwen2.5-7B on Kaggle plus explicit rules for negation, hedging, post-op and severity thresholds, written per label and **tuned against gold labels**. Output soft probabilities. | Per-label agreement with gold (AUC of report-label vs gold ≥ 0.95 on the easy labels). Every rule change is re-scored on gold. |
| **S2** | **Richer supervision from reports** | Extract about 20–40 extra targets per report: tear location / type, acute vs chronic, bruise location, OA severity, post-op status. Use them as **auxiliary heads**. Later: align image embeddings to report-text embeddings (privileged information at training time only). | CV macro AUC with aux heads > without, same folds and seed. |
| **S3** | **Frozen features, fast iteration** | Run a strong frozen encoder (DINOv2-small / RadImageNet / BiomedCLIP; **check licences**) once over the preprocessed slices and cache the features. All aggregator / head / loss experiments run on the cache, in minutes, on the Mac or CPU. Full fine-tuning only at the end, for proven configs. | Experiments per day. Aim for ≥10 a day in weeks 2–3. |
| **S4** | **Efficiency track by design** | One small distilled model. Decode only the 4 slots, subsample slices, parallel decode. Budget: **≤ 30 min total runtime**. Keep a separate "accurate" submission for the main leaderboard. | Runtime logged on every submission. Efficiency score estimated as `AUC/(0.5−maxAUC) + sec/32400`. |

Smaller edges (cheap, adopt when relevant):
- **Anatomy over learned localisation:** per-label series choice and mm-based crops of fixed regions (see labels.md). No keypoint model unless the crop-free baseline stalls.
- **Resolution by label:** 128 px for Effusion / Baker's, higher only for ACL / menisci / Fracture.
- **Metadata second stage:** a LightGBM or small MLP over the 12 image outputs plus free scan metadata (pixel spacing, field strength, manufacturer, series-flag combination). **Keep it only if CV improves**, and first check which tags survived the 86-tag allowlist.
- **Use label correlations:** the same second stage learns the injury and degenerative clusters.
- **Spend effort where the score moves:** Synovitis, Fracture and Lateral Meniscus get extra attention. Effusion doesn't.
- **External data:** fastMRI+ pathology boxes for "where to look" pretraining, *only after* the licence and competition ruling are checked.
- **Experiment discipline:** fixed folds, one change at a time, a 10% subset for smoke tests, and a log line per run in `experiments.md`.
- **Teaming option:** decide by **2026-10-12** (merger deadline 10-15) whether to offer S1/S2 to a stronger team. Criteria: our CV is stalled, and a team with GPUs needs labels.

## Compute: all-cloud
- The laptop is an M3 MacBook with 16 GB unified memory (not upgradeable) and about 197 GB free. It can't hold 570 GB and is slow for MRI training. Use it for code, text-only work and experiments on cached features.
- **Preprocess on Kaggle**, where the data already lives. Nothing gets downloaded. Save compact uint8 stacks as private Kaggle datasets (about 20 GB per notebook output).
- **Train** on Kaggle GPUs: about 30 h a week on P100 or 2×T4, with 9 h sessions. Rent a RunPod or Vast.ai RTX 4090 ($0.30–0.70 an hour) only if the quota runs out. Upload only the shrunk data, and **shut down rented instances**.
- **Label the reports** with an open-weights LLM (Qwen2.5-7B or Llama-3.1-8B) running *on Kaggle*. That keeps the report text in-house. Sending it to external APIs may break the data rules, so check the Rules tab first.
- Budget: $0 on Kaggle only, or about $20–50 with some rented GPU time.

## Model
- About **6–15 M parameters** to start, with a 30 M ceiling.
- Per-slice 2D backbone: EfficientNet-B0 (5.3 M) → B2/B3, or ConvNeXt-Nano/Tiny. Try RadImageNet weights against ImageNet weights.
- Slice aggregator: attention pooling or a 1–2 layer transformer with d = 256, about 0.5–2 M parameters.
- One shared trunk with 12 heads. The labels are strongly correlated (see labels.md).
- The report LLM is separate. It isn't part of the submission.

## Training approach (what changes vs the DR/APTOS project)
- **Loss:** 12 sigmoid outputs with BCE. Not softmax, not QWK.
- **Validate on both, gold decides.** Gold macro AUC (58 studies, pooled OOF) is the final arbiter; macro AUC against the report labels on the 4,349 OOF report-only studies is the steadier signal for model selection. Check the report labeller against gold per label first.
- Train on gold labels plus soft report-derived labels, then fine-tune on gold. Clean noisy report labels with out-of-fold predictions (lumbar 2nd place's confident-learning filter).
- **Architecture:** slice → series → study.
  - 2.5D input: neighbouring slices as channels.
  - Attention over slices.
  - Fuse the series slots with a padding mask and a missing-slot mask.
- **Feature caching:** run a frozen backbone once, then train the aggregator cheaply (even on the Mac).
- **MRI specifics:**
  - Percentile normalisation per series.
  - Sort slices by `ImagePositionPatient`.
  - **No uncontrolled left/right flips** (medial/lateral labels).
  - Mild augmentation, CutMix / token masking.
- **CV:** 5 folds grouped by study, multilabel-stratified. Track the macro and per-label AUC. Trust CV over the public LB.
- **Inference budget:** about 25 s per study. DICOM decoding (JPEG 2000) is the likely bottleneck, so benchmark it in week 1.

## Order of work and milestones

| When | Milestone | Strategy |
|---|---|---|
| **Wk 1 (09-23 → 09-29)** | Accept the rules. Run EDA: label prevalence, report languages, series-flag combinations, surviving DICOM tags (laterality? manufacturer? field strength?), JPEG 2000 decode speed. | — |
| Wk 1 | Report labeller v1 (12 labels) scored against gold per label. Rules written in `labels/rules.md`. | S1 |
| Wk 1 | Kaggle preprocessing notebook → 4-slot uint8 stacks as private datasets. | S3, S4 |
| **Wk 2 (09-30 → 10-06)** | Cache frozen features for 2–3 encoders. Aggregator baseline on the cache. **First submission.** | S3 |
| Wk 2 | Aux-target extraction (20–40 targets) → aux heads, A/B tested on CV. | S2 |
| **Wk 3 (10-07 → 10-13)** | Best config fine-tuned end to end. OOF label cleaning. Metadata / label-correlation second stage. Hard-label work (Synovitis, Fracture, Lateral Meniscus). | S1–S3 |
| 10-12 | **Team-up decision.** | — |
| **10-15** | **Entry and merger deadline.** | — |
| **Wk 4 (10-14 → 10-22)** | Distil into one fast model, ≤ 30 min runtime. Small rank-averaged ensemble for the accurate submission. Select 2 final submissions: accurate + efficient. | S4 |
| **10-22** | **Final submission deadline.** | — |

## Working agreement and scope cut (2026-09-23)
- **Time:** Max has **under 5 h a week** until 10-22. Claude writes the code and explains the key choices; Max reviews.
- **Kaggle runs:** Claude pushes and runs kernels through the Kaggle CLI (`kaggle kernels push`), then polls them and pulls the outputs. Before any GPU run, Claude says roughly how many GPU hours it will use (the quota is about 30 h a week).
- **Code:** private GitHub repo. Keep it private until the competition ends. Sharing code outside a team during the competition breaks the rules.
- **Scope, given the time:** keep **S1** (report labels, tuned on gold), **S3** (feature cache and aggregator baseline) and **S4** (one efficient submission). The **efficient model is also the accurate one** unless time allows a small ensemble.
- **Dropped unless ahead of schedule:** S2 aux heads and text alignment, the metadata / label-correlation second stage, OOF label cleaning, fastMRI+ pretraining.
