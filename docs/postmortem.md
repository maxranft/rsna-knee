# Post-mortem: RSNA Knee Abnormality Detection (2026)

Written 2026-10-25, three days after the competition closed. Sources: [experiments.md](../experiments.md) (214 runs),
[labeller.md](labeller.md), the Kaggle leaderboard and the top public notebooks.

## Result

| | Macro AUC |
|---|---|
| CV, 58 gold studies (pooled OOF) | 0.771 |
| CV, 4,349 report-labelled studies | 0.803 |
| **Public LB** | **0.796** (rank 5,271 of 6,425) |
| **Private LB** | **0.783** |
| Median team, public LB | 0.942 |
| Bronze cut-off (top 10 %), public LB | 0.958 |
| Winning private score | 0.960 |

No medal, and the efficiency track was never entered. CV landed within 0.03 of both leaderboards, so the
validation was honest. The model itself was weak: 0.15 below the median team.

## What we built

- **Labels (S1):** Qwen2.5-7B answering yes/no questions about each report, scored by log-probability, run on Kaggle
  so the report text never left it. Gold agreement 0.861 macro.
- **Features (S3):** a frozen DINOv2-S over four series slots: 256 px crops at 0.7 mm/px, 224 px into the encoder,
  up to 40 slices. Each slice was reduced to the CLS token plus a 4×4 average-pooled patch grid, then PCA to 128 dimensions.
- **Head:** a one-layer transformer over slices and slots, trained on the Mac in about 30 minutes per 5-fold run.
- **Submission:** 32 seeds × 5 folds = 160 heads, rank-averaged, with leave-one-slot-out TTA and
  Synovitis replaced by rank-mean(Synovitis, Effusion).

## Timeline

| Date | |
|---|---|
| 09-22 | Started. The competition had been open since at least August. |
| 09-23 – 09-25 | EDA, fixed folds, DICOM survey. Kaggle kernels for preprocessing, DINOv2-S features, the labeller and translation. |
| 09-29 – 09-30 | First 5-fold results on the Mac; head sweep of 20 changes. |
| 10-04 | Heads uploaded to Kaggle. From here on `kaggle kernels push` was blocked by Claude Code's safety check, so every GPU run waited on Max. DINOv2-B features and a Qwen2.5-14B labeller were built and tested locally but **never run**. |
| 10-05 – 10-06 | Synovitis mix, TTA, distillation: the final model (gold 0.771). |
| 10-06 – 10-13 | Dashboards and statistical diagnostics; more seeds of the same config. |
| 10-14 | First and only submission: public 0.796. |
| 10-15 – 10-22 | 56 more seeds (104–159) of the same config. None submitted. |
| 10-22 | Efficiency package ready about 17:00 CDT. The push was asked for at 19:59 CDT, an hour after the 18:59 CDT (23:59 UTC) deadline. |

## What worked (on our own CV)

| Change | Effect |
|---|---|
| PCA-128 token cache instead of the full 384-d tokens | report +0.018, gold +0.005 |
| 23-seed ensemble vs one seed of the same config | about +0.015 on both gauges |
| Synovitis := rank-mean(Synovitis, Effusion) | gold +0.011 macro (Synovitis 0.53 → 0.66). Gold Synovitis is really *effusion-synovitis*: 47 % positive in gold vs 20 % in the reports. |
| Born-again distillation from a fold-wise teacher | single models: report +0.011, gold +0.010 |
| Leave-one-slot-out TTA | gold +0.003 |
| Dropout 0.3, slot dropout 0.3, 16 → 28 epochs | small, each under 0.005 |

**Rejected:** canonicalising left knees, a second transformer layer, a higher learning rate, bagging, repeating
gold studies 10×, EMA, label smoothing, weight-decay and batch-size changes, auxiliary heads (raw Effusion, knee
side), per-plane projection, neighbour-slice context. d 384 tied d 256 and was slower.

## Why we finished 0.15 behind the median

1. **The frozen, pooled encoder capped the model.** A 4×4 pooled grid means one token summarised about 45 × 45 mm
   of knee, while meniscal and ACL tears are a few millimetres. That detail was gone before any training, and no
   head, ensemble or TTA could bring it back. On 09-30 the log already said ensembles had saturated and that the next
   gain needed better image features. The public notebooks at 0.95–0.96 fine-tuned their encoders. One ran CoAtNet on
   384 px adjacent-slice triplets. Another ran DINOv2-S, the same encoder we used, with its last six blocks unfrozen,
   at 224 and 336 px.
2. **The labels were weaker than planned.** S1's own success check was gold agreement of at least 0.95 on the easy
   labels. The labeller reached 0.861 macro, with only Medial OA above 0.95, Synovitis at 0.71 and Lateral OA at 0.80.
   The 14B labeller that might have closed part of the gap was never run. Public LLM label sets existed
   (e.g. `pilkwang/rsna-knee-llm-labels`) and could have been scored against our 58 gold studies.
3. **The GPU path stalled while the Mac kept busy.** Both fixes needed Kaggle GPU runs, and from 10-04 those waited
   on Max. The Mac, which could run unattended, kept training seeds of a config that had stopped improving:
   89 more runs and about 59 hours of training after 10-06, with flat CV.
4. **No leaderboard feedback until day 22.** The plan put the first submission in week 2. It came on 10-14, the day
   before the entry deadline. A submission around 10-01 would have shown a gap of more than 0.1 to the public
   notebooks, with three weeks left to change direction.

## What went right

- **Honest validation.** We used two gauges with gold as the arbiter, and measured seed noise (gold ±0.010, report ±0.004). CV
  predicted both leaderboards within 0.03.
- **Data handling.** Report text never left Kaggle, and no report text or competition images entered a Claude session.
- **One code path for training and submission.** `scripts/build_kernels.py` bundles `src/knee` into the kernels,
  and the first submission push scored without errors.
- **Label semantics beat model changes.** Reading what the gold Synovitis label actually means gave +0.011 from one
  pre-registered rule, the largest single gain after ensembling.
- **Experiment discipline.** Every one of the 214 runs is logged with one change and a verdict.

## Lessons for the next competition

1. **Submit in week 1**, even a trivial baseline, and compare our score with the public leaderboard every week.
2. **Read the top public notebooks and the discussion forum before designing.** In a mature competition they set
   the floor. The median team here scored 0.942, mostly by building on public work.
3. **Use frozen features for fast experiments, not for the final model.** Plan a fine-tuned, full-resolution
   baseline early, and keep spatial detail when the findings are measured in millimetres.
4. **Stop when a direction saturates.** For example: three runs without a gain on either gauge, then move to the next bet.
5. **Clear blockers on the critical path on day one.** Here that meant a permission rule for `kaggle kernels push`
   and a weekly GPU-quota plan.
6. **Convert every deadline to local time and set a reminder the day before.** The efficiency submission
   missed by one hour.

## If we did it again (four weeks)

- **Week 1:** fork the strongest public notebook and submit it; read the discussion; score our labeller and the
  public label sets on the 58 gold studies.
- **Week 2:** fine-tune one encoder at 336–384 px on Kaggle GPUs with the best labels. Labels are still the edge (S1).
- **Week 3:** label-semantics fixes like the Synovitis one, then a small ensemble.
- **Week 4:** distil for the efficiency track and submit both entries by the day before the deadline.

## Where things are

- [experiments.md](../experiments.md): every run. [labeller.md](labeller.md): labeller results.
  [plan.md](plan.md): the original plan. [runbook.txt](../runbook.txt): how to run things.
- Final model: Kaggle dataset `maxranft/rsna-knee-heads` (private) and kernel `maxranft/rsna-knee-submit`. Local
  copies are in `build/rsna-knee-heads` and `build/rsna-knee-heads-eff`, which are git-ignored.
- Local `data/` was deleted on 2026-10-25. Everything in it can be rebuilt from Max's Kaggle kernels (features,
  labeller, stacks, survey) or from the competition download. `scripts/pca_cache.py` rebuilds the PCA caches.
