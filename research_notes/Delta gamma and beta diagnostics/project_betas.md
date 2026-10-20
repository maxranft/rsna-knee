# The project's beta diagnostics: what they say, how much is noise, and what to do with them (2026-10-16)

Scope: `src/knee/dashboard.py` beta, transfer, plane and Cliff code, run on all 155 finished single runs in `runs/`. These form 35 configs. The analysis read `metrics.json`, `oof.csv`, `args.json` and `history.jsonl`, plus the label columns of `data/raw/train.csv` (read without `Report`) and `data/labeller/all-v2/report_scores_all.csv` (scores only). No report text and no images were read, and no training was run.

Reproduction check: my re-run of `betas_section`/`plane_section` matches the payload in `runs/dashboard.html` (built 2026-10-15T19:16:17) to every digit. AUCs recomputed from `oof.csv` match `metrics.json` to within 5e-5 (rounding).

Scratch scripts are in `/private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/` (s1_repro.py … s8_wls.py). These are temporary files; the key code is quoted below.

## Q1. How the beta code forms configs and regresses, and whether it has bugs or statistical problems

### Takeaway
The code does what its docstrings say, and there is no arithmetic bug. Five statistical and interpretive problems matter, in this order:
1. **The transfer intervals bootstrap over configs only, with the 58 gold studies held fixed.** This makes per-label transfer look far more certain than it is (see Q2 and Q3).
2. **The beta and plane fits include 1- and 2-seed configs.** MIN_SEEDS is not applied there. Those configs give high-leverage points that swing several per-label betas.
3. **The template wording treats β≈1 as "real transfer" and "the interval reaches 0" as "the rule needs work".** Under label noise, real transfer gives β>1, and an interval that reaches 0 is mostly a lack of power.
4. **Attenuation from noisy report AUC is negligible here**, so that concern can be dropped.
5. **The plane's accuracy and β_time carry no information** (Q5).

### Cited Findings
**How configs are formed**
- A config is the `args.json` settings minus `seed`, `log`, `no_open` and `workers`. Switches left at their off value are also dropped, so older runs still match later ones. `distill` and `aux_weight` count only when `teacher` or `aux` is set — [dashboard.py L312–324](/Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py).
- `config_groups` keeps finished runs only. It drops ensembles and mixes (ids starting `ens-` or `mix-`) and runs without gold or report AUC — [dashboard.py L327–336](/Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py).
- `load_runs` drops runs with fewer than 5 folds and runs on synthetic features — [results.py](/Users/maxranft/Projects/rsna-knee/src/knee/results.py).
- `same_labels` keeps configs whose `labels` and `label_map` equal the base config's — [dashboard.py L381–383](/Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py). Today all 35 configs use `all-v2/report_scores_all.csv` with `{"Effusion": "Effusion@moderate"}`, so this filter removes nothing (s1_repro.py).
- **No minimum seed count applies to the fits.** `MIN_SEEDS = 3` is used only in `configs_section` (L366). `betas_section` needs only `MIN_CONFIGS = 8` configs (L316–317, L393), and `plane_section` reuses the same configs.
  - Seeds per config today: 25, 23, 11, 9, 9, 5, 5, seventeen configs with 3, six with 2 and five with 1.
  - All 11 configs with 1 or 2 seeds are from 2026-09-29/30. They sit at the low end of report AUC (0.7615–0.7814), where they have the most leverage (s2_configs.py).
- **The base config** is the auto-improve base (`runs/auto/state.json` `base_runs` = 20261006-174748…), 25 seeds. Its settings: d 384, 28 epochs, drop 0.3, slot-drop 0.3, PCA-128 features, distillation 0.5 from `runs/teacher-ep20x9`. Mean report 0.7993, mean gold 0.7461 (s2_configs.py; [dashboard.py L339–343](/Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py)).
- The teacher used for distillation is fold-safe: the fold-k teacher heads never saw fold k — [teacher_preds.py docstring](/Users/maxranft/Projects/rsna-knee/scripts/teacher_preds.py); [train_head.py L159–164](/Users/maxranft/Projects/rsna-knee/scripts/train_head.py).
- Every config trains the gold studies in its training folds on gold labels. `gold_repeat` multiplies them — [train_head.py L124–126, L154](/Users/maxranft/Projects/rsna-knee/scripts/train_head.py).

**What is regressed on what**
- There is one point per config: the unweighted mean over its seeds. A 1-seed config weighs the same as the 25-seed base — [dashboard.py L386–412](/Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py).
- **Transfer beta** for label j: the slope of mean gold AUC_j on mean report AUC_j, using `beta(y, x) = cov(y, x)/var(x)` (L95–98). The macro version uses the runs' `gold_macro` and `report_macro` and adds α and r for the "characteristic line".
- **Model beta** for label j: the slope of report AUC_j on the mean report AUC of the other 11 labels (L410).
- **Intervals:** `boot_beta` gives a 90% percentile bootstrap that resamples configs, 2,000 draws (L101–108). Gold and report studies are never resampled.
- **Display wording:** "Transfer beta near 1: gains on the report labels are real; an interval that reaches 0: they have not shown up on gold, so that label's report rule needs work rather than the model" — [template.html L948](/Users/maxranft/Projects/rsna-knee/scripts/dashboard_ui/template.html).
- **Reference run:** `collect()` takes the last run that experiments.md tags "Current reference". The built dashboard uses **20260929-140030** (report 0.7797, gold 0.7242), so the "reference model" panel (per-label CIs, ROC curves, labeller ceiling) and the `vs_ref` columns describe a 2026-09-29 run, not the current best config — [dashboard.py L633–636](/Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py); `runs/dashboard.html` payload `"reference":"20260929-140030"`.

**Statistical checks (s3_noise.py, s6_plane.py)**
- **Attenuation:** the between-config SD of report macro (0.0091) is 6× the pooled seed SD (0.0015). The reliability of config-mean report AUC is λ_x = 0.99 for macro and 0.87–0.97 per label, so correcting for errors in x changes the slopes by ≤ 0.3 (Q2).
- **Correlated errors:** the within-config correlation between a seed's report and gold deviations is r = +0.09 for macro and −0.24 to +0.19 per label. Seed luck on report does not carry over to gold, so there is no correlated-error bias.
- **Shared seeds:** 24 configs all use seeds 42/43/44. The mean pairwise correlation of seed deviations across configs is −0.03 for report (permutation p = 0.73) and −0.03 for gold (p = 0.81). Sharing seeds induces no detectable common noise.
- **Missing study-sampling noise:** resampling the 58 gold studies widens the macro transfer interval from [0.92, 1.34] to [0.57, 1.64]. Per-label intervals become 1.5–2.5× wider (Q3).
- **Equation error ("τ"):** for macro, the scatter of config means around the line beyond seed noise has SD 0.0018 (s8_wls.py). Configs fall close to the macro line once seed noise is removed. Visible off-line points:
  - gold-repeat 10: Δreport −0.0164, Δgold −0.0057 vs base. Above the line, as expected, since it trains harder on gold labels.
  - ema 0.999: −0.0115 / −0.0006 (above the line).
  - slice-drop 0.2: −0.0078 / −0.0177 (below the line). (s5_boot.py)

#### Code run (core of s3_noise.py / s5_boot.py / s8_wls.py)
```python
# config means mx (report), my (gold); n = seeds per config; pooled within-config (seed) moments wxx, wyy, wxy
Sxx, Syy, Sxy = var(mx, ddof=1), var(my, ddof=1), cov(mx, my)
Exx, Exy = wxx*mean(1/n), wxy*mean(1/n)                    # expected seed-noise share of the config-mean moments
lam_x   = (Sxx - Exx)/Sxx                                   # reliability ratio of config-mean report AUC
beta_mom = (Sxy - Exy)/(Sxx - Exx)                          # moment (reliability) correction, allows correlated errors
delta = wyy/wxx; beta_deming = (Syy-delta*Sxx + sqrt((Syy-delta*Sxx)**2 + 4*delta*Sxy**2))/(2*Sxy)
# split-half IV: random halves A/B of each config's seeds; x_A instruments x_B (median of 200 splits)
beta_iv = (cov(yB, xA) + cov(yA, xB)) / (cov(xB, xA) + cov(xA, xB))
# two-level bootstrap: gold studies AND configs
for b in range(1000):
    s = rng.integers(0, 58, 58)                     # resample gold studies
    G = config_means(auc(PG[:, s], YG[s]))          # every run's gold AUC per label on the resample (rankdata, ties averaged)
    c = rng.choice(cfgs, len(cfgs)); slope(XR[c], G[c])
# WLS: weights 1/(tau^2 + s_seed^2/n), with tau^2 = residual variance minus mean seed variance
```

### Inferences
- **The intervals ask the wrong question.** The config-only bootstrap answers "would another set of configs give the same slope on these 58 studies?". The decision needs "will report gains show up on the hidden test set?". Only the two-level (configs × gold studies) interval addresses that. The per-run gold OOF needed for it is already on disk. Three config sets × 1,000 two-level draws over 155 runs finished well inside a 10-minute CPU call here, so it is affordable if cached the way `dash_cache.json` is.
- **Apply MIN_SEEDS (≥3) to the fits, or weight configs by 1/(τ² + s²/n).** Macro barely moves either way: OLS 1.10 → 1.09 with ≥3 seeds, and 1.16 under WLS. Per-label betas move a lot:
  - Lateral Meniscus 1.49 → 0.75
  - Synovitis −1.05 → −0.40
  - PF OA 1.39 → 0.65
  - Lateral OA 0.84 → 0.27
  - Medial Meniscus −0.75 → +0.49
- **The transfer note in the template needs new wording** (see Q4 for the theory). A β above 1 is what real transfer through noisy labels looks like. An interval that reaches 0 usually means too few gold positives to tell, not that transfer has failed.
- **The configs are a project trajectory, not a designed experiment.** Early configs differ from the base in 4–6 settings at once. The slope therefore mostly summarises "the changes kept so far moved report and gold together", with most of its leverage from the 0.76→0.80 climb.
  - Configs were selected on report AUC, i.e. on x. That does not bias E[gold | report], but the slope says nothing about a new kind of change (for example a new labeller) that moves report through a different mechanism.

### Gaps
- I did not resample the 4,349 report-only studies. Their sampling noise is common to all configs and should mostly cancel in slopes, but it is unquantified.
- Whether 20260929-140030 is still meant to be the "Current reference" in experiments.md is a decision for Max. It is not a statistical question.

## Q2. Current transfer and model betas, noise levels, reliability ratios and attenuation-corrected slopes

### Takeaway
**Macro transfer is solid at about 1.1.** Every estimator agrees (OLS 1.10, moment-corrected 1.11, Deming 1.12, split-half IV 1.10–1.26, Theil–Sen 1.13–1.20, Spearman 0.78–0.89), because report AUC barely suffers from attenuation (λ_x = 0.99).

**Per label, the uncertainty is almost all in gold, not report.** Seed SD of gold per label is 0.018–0.039 against 0.002–0.006 for report. With only 9–35 gold positives, per-label slopes are fragile and depend on which configs are included.

Pooled seed noise is 0.0015 for report macro and 0.0085 for gold macro. The project's "≈0.004 / ≈0.010" were ranges over 3 seeds, so report noise was overstated and gold noise slightly understated.

### Cited Findings
**Dashboard values reproduced** (k = 35 configs, 155 runs, config bootstrap 90%): macro transfer β = 1.105 [0.923, 1.335], r = 0.87, α = −0.136. The characteristic line is gold = −0.136 + 1.105 × report. Source: s1_repro.py, matching the [runs/dashboard.html](/Users/maxranft/Projects/rsna-knee/runs/dashboard.html) payload.

**Seed noise.** The project's statement is "report macro varies by ~0.004, gold by ~0.010" (3 seeds, 2026-09-29) — [experiments.md L5, L14](/Users/maxranft/Projects/rsna-knee/experiments.md). The pooled within-config SD over all configs with ≥2 seeds is 0.0015 for report and 0.0085 for gold.
- The expected range of 3 normal draws is about 1.69σ, which gives 0.0025 for report and 0.014 for gold.
- The single-seed rule "a report difference under ~0.004 is not a result" is therefore conservative for report. Gold seed noise is larger than stated. (s3_noise.py)

**Macro transfer estimators** (s3_noise.py, s8_wls.py)

| Configs | k | SD between configs, report / gold | Seed SD, report / gold | λ_x | λ_y | OLS [cfg-boot 90%] | Moment-corrected | Deming (δ) | IV split-half | Theil–Sen [95%] | Spearman ρ | Pearson r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all (dashboard) | 35 | 0.0091 / 0.0116 | 0.0015 / 0.0085 | 0.99 | 0.78 | 1.10 [0.92, 1.33] | 1.11 | 1.12 (31.8) | 1.26 [1.12, 1.44] | 1.20 [0.96, 1.42] | 0.89 | 0.87 |
| ≥3 seeds | 24 | 0.0078 / 0.0100 | 0.0015 / 0.0084 | 0.99 | 0.81 | 1.09 [0.90, 1.23] | 1.10 | 1.11 (33.2) | 1.10 [0.92, 1.26] | 1.13 [0.77, 1.41] | 0.78 | 0.86 |

WLS over all configs, weights 1/(τ²+s²/n): 1.16 [1.04, 1.32]. The equation-error SD is 0.0018.

**Per-label transfer, all 35 configs** (the dashboard's set). Gold positives and labeller AUC come from `gold_eval_all.json`; the two-level interval resamples both configs and gold studies.

| Label | Gold pos | Labeller AUC | λ_x | Seed SD rep / gold | Dashboard OLS [cfg-boot 90%] | Moment-corr. | IV | Theil–Sen | Spearman ρ (p) | **Two-level 90%** | P(β>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Macro | 58 studies | 0.861 | 0.99 | 0.0015 / 0.0085 | +1.11 [0.92, 1.34] | 1.11 | 1.26 | 1.20 | 0.89 (<.001) | **[0.57, 1.64]** | 1.00 |
| ACL | 24 | 0.862 | 0.95 | 0.0041 / 0.025 | +0.11 [−0.45, 0.76] | 0.12 | 0.48 | 0.10 | 0.04 (.81) | [−1.21, 1.23] | 0.59 |
| MCL | 9 | 0.927 | 0.92 | 0.0044 / 0.039 | +1.09 [−0.12, 2.18] | 1.15 | 1.67 | 1.62 | 0.31 (.07) | [−1.33, 3.19] | 0.78 |
| Medial Meniscus | 26 | 0.900 | 0.94 | 0.0034 / 0.034 | −0.75 [−2.33, 0.50] | −0.87 | 0.15 | −0.59 | −0.15 (.38) | [−3.52, 1.78] | 0.29 |
| Lateral Meniscus | 23 | 0.865 | 0.87 | 0.0055 / 0.038 | +1.49 [0.96, 1.97] | 1.75 | 1.64 | 1.51 | 0.47 (.005) | [−0.12, 3.23] | 0.94 |
| Medial OA | 15 | 0.949 | 0.97 | 0.0024 / 0.018 | +1.14 [0.64, 1.77] | 1.18 | 1.44 | 1.12 | 0.58 (<.001) | [−0.01, 2.44] | 0.95 |
| Lateral OA | 11 | 0.795 | 0.94 | 0.0035 / 0.032 | +0.84 [0.21, 1.52] | 0.88 | 0.63 | 0.97 | 0.32 (.06) | [−0.95, 2.71] | 0.79 |
| PF OA | 21 | 0.870 | 0.96 | 0.0031 / 0.022 | +1.39 [0.58, 2.17] | 1.40 | 0.94 | 1.54 | 0.49 (.003) | [−0.03, 3.01] | 0.94 |
| Effusion (@moderate) | 35 | 0.820 | 0.96 | 0.0021 / 0.024 | +2.75 [2.04, 3.50] | 2.86 | 2.64 | 2.77 | 0.73 (<.001) | **[1.09, 4.63]** | 1.00 |
| Synovitis | 27 | 0.714 | 0.96 | 0.0030 / 0.031 | −1.05 [−2.38, −0.37] | −1.11 | −0.92 | −1.76 | −0.49 (.003) | [−3.44, 0.97] | 0.19 |
| Baker's | 12 | 0.889 | 0.95 | 0.0040 / 0.023 | +0.86 [0.45, 1.30] | 0.92 | 1.06 | 0.86 | 0.45 (.007) | [−0.32, 2.25] | 0.89 |
| Contusion | 19 | 0.835 | 0.94 | 0.0049 / 0.022 | +0.64 [0.00, 1.08] | 0.74 | 0.62 | 0.32 | 0.21 (.22) | [−0.43, 1.76] | 0.84 |
| Fracture | 18 | 0.906 | 0.91 | 0.0059 / 0.026 | +1.83 [1.21, 2.35] | 2.04 | 2.03 | 1.59 | 0.67 (<.001) | **[0.70, 2.99]** | 0.99 |

Sources: s1_repro.py, s3_noise.py, s5_boot.py, [gold_eval_all.json via dashboard.py L484–488](/Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py), [docs/labeller.md](/Users/maxranft/Projects/rsna-knee/docs/labeller.md).

**Sensitivity of the transfer slopes.** Columns: configs with ≥3 seeds (k = 24); the "top" regime of configs with report ≥ 0.785 (k = 17, range 0.7855–0.7999, which is where the final-week decisions are made); WLS over all configs; and λ_y, the reliability of gold config means with ≥3 seeds. Sources: s5_boot.py, s3_noise.py, s8_wls.py.

| Label | ≥3 seeds: OLS [two-level 90%] | Top regime: OLS [two-level 90%] | WLS all [cfg-boot] | λ_y (≥3 seeds) |
|---|---|---|---|---|
| Macro | 1.09 [0.44, 1.62] | 1.09 [0.36, 1.95] | 1.16 [1.04, 1.32] | 0.81 |
| ACL | 0.20 [−1.06, 1.47] | 0.71 [−1.13, 2.51] | 0.32 [−0.23, 0.83] | 0.62 |
| MCL | 1.16 [−1.55, 3.54] | **2.39 [0.39, 4.42]** | 1.29 [0.11, 2.28] | 0.62 |
| Medial Meniscus | 0.49 [−1.57, 2.23] | 0.79 [−2.08, 3.62] | −0.38 [−1.74, 0.56] | 0.46 |
| Lateral Meniscus | 0.75 [−1.37, 2.68] | −0.29 [−2.58, 2.06] | 1.23 [0.78, 1.68] | 0.41 |
| Medial OA | **1.36 [0.09, 3.15]** | **3.80 [1.31, 6.57]** | 1.26 [0.77, 1.83] | 0.67 |
| Lateral OA | 0.27 [−1.81, 2.19] | 1.09 [−2.70, 4.82] | 0.56 [0.07, 1.28] | **−0.02** |
| PF OA | 0.65 [−1.01, 2.56] | 2.96 [−0.70, 6.49] | 1.31 [0.51, 2.08] | 0.75 |
| Effusion | **2.65 [0.55, 4.87]** | 1.10 [−1.54, 5.25] | 2.72 [2.07, 3.31] | 0.68 |
| Synovitis | −0.40 [−2.93, 1.24] | −2.47 [−5.70, 0.73] | −0.99 [−2.08, −0.32] | **−0.07** |
| Baker's | 0.55 [−0.80, 1.88] | 0.09 [−1.15, 1.59] | 0.84 [0.50, 1.26] | 0.52 |
| Contusion | 0.27 [−0.98, 1.62] | 0.53 [−1.17, 2.37] | 0.64 [0.13, 1.06] | 0.62 |
| Fracture | **1.32 [0.36, 2.36]** | 1.15 [−0.24, 2.66] | 1.67 [1.22, 2.13] | 0.56 |

**Model beta.** Report AUC_j on the mean report AUC of the other 11 labels. λ_x = 0.99 throughout, so the moment-corrected values equal OLS to within ±0.01 (s3_noise.py).

| Label | Dashboard (35 configs) [90%] | ≥3 seeds, moment-corrected [90%] | Seed corr. with other labels |
|---|---|---|---|
| ACL | 1.16 [1.04, 1.26] | 1.30 [1.11, 1.50] | +0.03 |
| MCL | 1.00 [0.87, 1.11] | 1.05 [0.84, 1.36] | +0.06 |
| Medial Meniscus | 0.89 [0.74, 1.06] | 1.06 [0.93, 1.19] | +0.25 |
| Lateral Meniscus | 0.96 [0.83, 1.12] | 0.90 [0.78, 1.05] | +0.28 |
| Medial OA | 0.77 [0.67, 0.95] | 0.73 [0.53, 0.97] | +0.19 |
| Lateral OA | 0.85 [0.74, 1.02] | 0.92 [0.67, 1.21] | +0.40 |
| PF OA | 1.05 [0.95, 1.19] | 0.98 [0.75, 1.27] | +0.07 |
| Effusion | 0.65 [0.57, 0.77] | 0.74 [0.61, 0.90] | +0.10 |
| Synovitis | 0.89 [0.66, 1.06] | 0.98 [0.77, 1.13] | +0.01 |
| Baker's | 0.83 [0.49, 1.07] | 0.74 [0.15, 1.23] | +0.28 |
| Contusion | 1.34 [1.18, 1.48] | 1.16 [0.92, 1.34] | +0.25 |
| Fracture | 1.34 [1.17, 1.46] | 1.20 [1.01, 1.33] | +0.14 |

**Method background**
- Measurement error in x biases an OLS slope toward zero, while error in y adds variance but no bias. Correcting needs the x noise level, typically from repeated measurements — [Regression dilution, Wikipedia](https://en.wikipedia.org/wiki/Regression_dilution).
- Deming regression needs a known error-variance ratio δ and assumes no equation error. As δ → ∞ its slope tends to OLS (a limit derived from its formula, not stated in the article) — [Deming regression, Wikipedia](https://en.wikipedia.org/wiki/Deming_regression).

### Inferences
- **For macro, attenuation is a non-issue, so the dashboard's macro β is trustworthy as a point estimate.** Report config means are 99% signal. Deming with δ ≈ 32 (gold noise ≫ report noise) collapses to OLS. Here Deming would also be the wrong tool, because there is real equation error (τ ≈ 0.0018 macro, 0.007–0.033 per label).
  - The split-half IV of 1.26 for all configs falls to 1.10 with ≥3 seeds. The gap comes from which configs are included, not from attenuation.
- **For labels, the binding constraint is gold noise, not report noise.** For Lateral OA (λ_y = −0.02) and Synovitis (λ_y = −0.07), the 24 well-replicated configs differ in gold AUC by no more than seed noise. No config change tried so far has moved gold for those two labels detectably. Their dashboard betas (+0.84, −1.05) come from 1–2-seed early configs and from noise.
- **Model betas are the stable diagnostic.** They are precise (CI half-width about 0.1–0.2) because they are measured entirely on 4,349 report studies.
  - ACL, Contusion and Fracture respond more than the average label to whatever improved the model (β 1.2–1.3).
  - Effusion and Medial OA respond less (β 0.65–0.77). These two are near their ceiling: gold AUC 0.90 and 0.94 for the base config.
  - Because of the transfer results, a high model beta (ACL) does not imply gold gains.

### Gaps
- λ_y and the per-label equation error are estimated from about 24 configs × 3 seeds, so they have wide uncertainty of their own. I did not bootstrap them.
- Theil–Sen is robust to outlying configs but does not correct for attenuation. It is included only as an outlier check (agreement with OLS within ±0.5 for most labels).

## Q3. Lower-noise estimates of transfer from paired comparisons on the same studies: does a report gain predict a gold gain?

### Takeaway
**For macro, the answer is yes and in proportion: about 1.1 gold per unit of report.**
- Gold agrees in sign 90% of the time when two configs differ by 0.005–0.01 in report, and 99–100% of the time when they differ by more than 0.01.
- Below 0.005, agreement is 60–62%. That is close to what perfect transfer would produce given gold's own noise (56–71% expected), so gold simply cannot verify such small gains.

**Per label,** paired comparisons with |Δreport| > 0.01 show transfer roughly as expected for Medial OA, Effusion, Fracture and MCL, and clearly short of it for Synovitis, Lateral OA, ACL, PF OA, Contusion and Baker's.

### Cited Findings
**All 276 pairs of the 24 configs with ≥3 seeds** (each config's mean over seeds; gold on the same 58 studies, report on the same 4,349). "Expected" is the sign agreement if gold changed by 1.1×Δreport with only seed noise (3 v 3 seeds, gold seed SD 0.0085). Sources: s5_boot.py, s6_plane.py.

| \|Δreport macro\| | Pairs | Gold same sign | Expected if β = 1.1 | Mean signed Δgold / mean \|Δreport\| | Welch-significant report pairs: agreement |
|---|---|---|---|---|---|
| ≤ 0.002 | 64 | 0.62 | ≈0.56 (at 0.001) | 1.85 | 5 pairs, 0.60 |
| 0.002–0.005 | 55 | 0.60 | ≈0.71 (at 0.0035) | 0.51 | 43 pairs, 0.58 |
| 0.005–0.01 | 49 | 0.90 | ≈0.88 (at 0.0075) | 1.29 | 49 pairs, 0.90 |
| 0.01–0.02 | 87 | 0.99 | ≈0.99 (at 0.015) | 1.06 | 87 pairs, 0.99 |
| > 0.02 | 21 | 1.00 | ≈1.00 | 1.13 | 21 pairs, 1.00 |

**The final-week candidates against the base config** (Welch z on seeds; s5_boot.py):

| Config vs base (25 seeds) | Seeds | Δreport | z (report) | Δgold | z (gold, seed noise only) |
|---|---|---|---|---|---|
| d 256 | 11 | +0.0006 | +1.37 | +0.0054 | +2.05 |
| d 256 · distill 0.7 · epochs 20 | 3 | −0.0004 | −0.74 | −0.0016 | −0.59 |
| d 256 · epochs 20 · teacher-dist9 | 3 | −0.0012 | −1.20 | +0.0041 | +1.26 |
| d 256 · epochs 20 | 9 | −0.0021 | −4.87 | −0.0002 | −0.10 |

**Two-level bootstrap for the gold term.** All configs share the same 58 studies, so study sampling is common to every config. Resampling those studies alone ("study-boot") already widens the macro interval from [0.92, 1.33] to [0.70, 1.47]. Adding config resampling gives [0.57, 1.64] (Q2 table; s5_boot.py).

**Per-label sign agreement for pairs with |Δreport_label| > 0.01.** "Expected" assumes β = 1 and gold seed noise only (s7_labels.py).

| Label | Pairs | Gold same sign | Expected (β = 1) | Shortfall |
|---|---|---|---|---|
| Macro | 108 | 0.99 | 0.98 | none |
| Medial OA | 88 | 0.91 | 0.88 | none |
| Effusion | 67 | 0.94 | 0.81 | none (exceeds; β > 1) |
| Fracture | 126 | 0.83 | 0.83 | none |
| MCL | 136 | 0.72 | 0.73 | none |
| Medial Meniscus | 117 | 0.70 | 0.76 | small |
| Lateral Meniscus | 109 | 0.62 | 0.73 | moderate |
| Baker's | 146 | 0.69 | 0.86 | clear |
| Contusion | 110 | 0.60 | 0.86 | clear |
| PF OA | 90 | 0.63 | 0.87 | clear |
| ACL | 138 | 0.62 | 0.85 | clear |
| Lateral OA | 105 | 0.50 | 0.77 | none to chance |
| Synovitis | 90 | 0.39 | 0.76 | reversed |

### Inferences
- **Which report differences to act on.** Between configs whose seed means differ in report macro by ≥ 0.005, the gold direction follows report about 90% of the time, and at ≥ 0.01 it essentially always does. Below 0.005, gold cannot tell the two apart even when transfer is perfect.
  - For those small differences the decision has to rest on report AUC (seed SD 0.0015, so 3 seeds a side resolve about 0.003) plus the well-supported prior that macro transfer ≈ 1.1.
  - It should not rest on a gold difference smaller than about 0.015–0.02.
- **Two of the four final candidates.**
  - The d 256 config has the best gold (+0.0054, z ≈ 2 on seed noise only) with report equal to base (+0.0006). A ≈0.005 gold edge is within gold's study and equation noise, so this is weak evidence.
  - Under a report-first rule it is a tie that costs nothing: 23.6 vs 43.6 training minutes in `train_minutes`, and a smaller head.
- **Pairs are not independent.** Each config appears in 23 pairs, so the agreement rates are descriptive. The per-config comparisons against base and the two-level bootstrap carry the inferential weight.

### Gaps
- I did not analyse rank-averaged ensembles. The final submission is an ensemble, and ensemble-vs-ensemble transfer may differ, likely being less noisy.
- The paired comparisons use config means over seeds. A study-level paired DeLong test per pair was not run. The two-level bootstrap covers the same source of variance.

## Q4. Per label: which labels show transfer clearly above zero, and how does that relate to labeller agreement and gold positives?

### Takeaway
**Clearly positive under the two-level bootstrap:** only Effusion (≈2.7) and Fracture (≈1.3–1.8), plus Medial OA when restricted to well-replicated or top configs. MCL is positive in the top regime.

**Indistinguishable from zero:** ACL, Medial Meniscus, Lateral OA, Contusion and Baker's. Synovitis leans negative.

The pattern follows **labeller–gold agreement and definitional fit more than gold-positive counts**. Synovitis, the worst labeller (0.714), transfers negatively. ACL transfers about zero despite labeller AUC 0.862 and 24 positives. Effusion transfers best after its labeller fix (0.711 → 0.820 with @moderate).

Under the label-noise model, real transfer should give β ≥ 1, roughly 1/(PPV+NPV−1), so the dashboard's "near 1 = real" benchmark is too strict.

### Cited Findings
**Theory.** Under class-conditional label corruption, AUC against corrupted labels = (1−α−β)·AUC against clean labels + (α+β)/2, i.e. AUC_corr − ½ = (1−α−β)(AUC − ½). It follows that optimising AUC on corrupted labels optimises clean AUC — [Menon et al., ICML 2015, Corollary 3](http://proceedings.mlr.press/v37/menon15.pdf).
- So Δgold = Δreport/(1−α−β) when the noise is independent of the image given the true label, and the expected transfer β* is ≥ 1.

**Calibrating β* from the 58 gold studies** (labeller sensitivity and specificity of the thresholded label vs gold; PPV/NPV re-derived at the report-only labelled prevalence). Source: s5_boot.py.
- β* = 1.5 for Medial Meniscus, Medial OA and Baker's; 1.9 for PF OA; 2.3 for Effusion; 2.6 for Lateral Meniscus; 5.0 for Synovitis; 6.0 for ACL; 13.7 for Lateral OA.
- **It cannot be estimated for MCL, Contusion and Fracture.** Their implied true prevalence comes out negative, because labeller specificity on the gold set (0.82–0.84) is too low to explain the report-only positive rates of 6.8–15%.

**Level check (base config).** Pure noise on the same population would need |gold − ½| ≥ |report − ½|. Measured gold vs report AUC:

| Label | Gold | Report |
|---|---|---|
| ACL | 0.718 | 0.815 |
| MCL | 0.654 | 0.744 |
| Medial Meniscus | 0.592 | 0.770 |
| Lateral Meniscus | 0.692 | 0.719 |
| Lateral OA | 0.700 | 0.808 |
| PF OA | 0.738 | 0.792 |
| Synovitis | 0.521 | 0.843 |
| Medial OA | 0.943 | 0.858 |
| Effusion | 0.904 | 0.859 |
| Baker's | 0.860 | 0.801 |
| Contusion | 0.856 | 0.802 |
| Fracture | 0.778 | 0.782 |

Seven labels have gold < report, which pure noise cannot explain (s7_labels.py). The gold set is enriched: "positives per study range from 1 to 9, and none has 0" — [docs/data.md L15](/Users/maxranft/Projects/rsna-knee/docs/data.md).

**Labeller history.** Effusion@moderate raised labeller–gold AUC from 0.711 to 0.820. Synovitis (0.714) and Lateral OA (0.795) are the weakest labeller labels, and "gold uses strict definitions" (broader wordings lost) — [docs/labeller.md](/Users/maxranft/Projects/rsna-knee/docs/labeller.md).

**Per-label synthesis.** Columns: two-level verdict from Q2; sign agreement from Q3; gold positives; labeller AUC and β* from above. Sources: s5_boot.py, s7_labels.py.

| Label | Gold pos | Labeller AUC | β* (noise model) | Measured β (all / ≥3 / top) | Two-level verdict | Pair agreement vs expected |
|---|---|---|---|---|---|---|
| Effusion | 35 | 0.820 | 2.3 | 2.75 / 2.65 / 1.10 | **> 0** (all, ≥3) | 0.94 vs 0.81 |
| Fracture | 18 | 0.906 | n/e | 1.83 / 1.32 / 1.15 | **> 0** (all, ≥3) | 0.83 vs 0.83 |
| Medial OA | 15 | 0.949 | 1.5 | 1.14 / 1.36 / 3.80 | **> 0** (≥3, top) | 0.91 vs 0.88 |
| MCL | 9 | 0.927 | n/e | 1.09 / 1.16 / 2.39 | > 0 only in top | 0.72 vs 0.73 |
| Lateral Meniscus | 23 | 0.865 | 2.6 | 1.49 / 0.75 / −0.29 | ≈ 0 (P > 0 0.94 for all) | 0.62 vs 0.73 |
| PF OA | 21 | 0.870 | 1.9 | 1.39 / 0.65 / 2.96 | ≈ 0 (P > 0 0.94 for all) | 0.63 vs 0.87 |
| Baker's | 12 | 0.889 | 1.6 | 0.86 / 0.55 / 0.09 | ≈ 0 | 0.69 vs 0.86 |
| Contusion | 19 | 0.835 | n/e | 0.64 / 0.27 / 0.53 | ≈ 0 | 0.60 vs 0.86 |
| Lateral OA | 11 | 0.795 | 13.7 | 0.84 / 0.27 / 1.09 | ≈ 0; λ_y ≈ 0 | 0.50 vs 0.77 |
| ACL | 24 | 0.862 | 6.0 | 0.11 / 0.20 / 0.71 | ≈ 0 | 0.62 vs 0.85 |
| Medial Meniscus | 26 | 0.900 | 1.5 | −0.75 / 0.49 / 0.79 | ≈ 0 | 0.70 vs 0.76 |
| Synovitis | 27 | 0.714 | 5.0 | −1.05 / −0.40 / −2.47 | ≤ 0 (P > 0 0.10–0.34) | **0.39** vs 0.76 |

### Inferences
- **Gold-positive count is not what separates the labels.** MCL has the fewest positives (9) yet transfers in the top regime. ACL (24), Medial Meniscus (26) and Synovitis (27) have many positives and do not transfer.
- **What separates them is whether the labeller's label means the same thing as gold's label on this enriched gold set.**
  - For Synovitis, report-label gains anti-transfer, with gold AUC at chance (0.52, 90% CI 0.41–0.64). This fits a labeller "synovitis" that tracks something else the model learns well (report AUC 0.843) but that gold's strict definition does not reward. This is an inference; I did not inspect any reports.
  - For ACL, the model's report gains are the largest of any label (model β 1.16–1.30) yet show no gold gain. A definitional or partial-tear mismatch is plausible (cf. the ACL@abnormal variant, +0.031 labeller AUC, not adopted).
- **Effusion's β ≈ 2.7 is what the noise model predicts (β* ≈ 2.3), not an anomaly.** Medial OA (1.1–1.4 against 1.5) and Fracture are also consistent with full transfer.
- **Labels with gold AUC below report AUC have extra problems.** These are ACL, MCL, both menisci, Lateral OA, PF OA and Synovitis. Either the gold population is harder (enriched with multi-pathology knees), or labeller errors are themselves image-predictable, or both. Either way, report gains on those labels are only partly about the gold-defined finding.

### Gaps
- The β* calibration rests on 9–35 positives and on an enriched gold set whose labeller specificity does not carry over to the report-only population, as the negative prevalences show. Treat β* as qualitative.
- I could not separate the two explanations for gold < report (population difficulty vs systematic labeller error) without reading reports, which is off-limits. Max could check a few Synovitis and ACL disagreements in the Kaggle viewer.

## Q5. The 3D plane (accuracy ~ AUC + training minutes): is accuracy at 0.5 meaningful, and is β_time identifiable?

### Takeaway
**No on both counts.** Macro accuracy at 0.5 is a near-deterministic, prevalence-dominated transform of report AUC: R² = 0.917 from AUC alone, the same as with minutes added. It sits only 0.025–0.037 above an all-negative classifier, and for three labels the model's 0.5-threshold accuracy is at or below all-negative.

**β_time is zero and unidentifiable.** It is −0.0002 accuracy per hour (SE 0.0015, t = −0.15). Training minutes vary 28–65 across the 25 seeds of the *same* config, so they mostly measure machine load. Training time is also not what the efficiency prize scores.

### Cited Findings
**Plane fit** (35 configs): accuracy = 0.541 + 0.360 × AUC − 3.9e-6 × minutes. 90% CI for β_auc [0.330, 0.408], for β_time [−6.9e-5, 3.5e-5] per minute; R² = 0.917 (s1_repro.py, matching the dashboard).
- corr(accuracy, AUC) = 0.957; corr(AUC, minutes) = 0.30; the classical t for β_time is −0.15 (s6_plane.py).
- Across runs, accuracy ~ AUC + mean predicted probability still gives R² = 0.91 (s6_plane.py).

**Accuracy against an all-negative baseline.** Config accuracy spans 0.8167–0.8295. Predicting all-negative on the report-only studies scores 0.792 (s6_plane.py). Base config per-label accuracy at 0.5 vs all-negative:

| Label | Model | All-negative |
|---|---|---|
| MCL | 0.846 | 0.848 |
| Lateral Meniscus | 0.863 | 0.865 |
| Fracture | 0.929 | 0.932 |
| ACL | 0.884 | 0.862 |

**Under-calling at 0.5.** Predicted-positive rate vs prevalence: MCL 0.039 vs 0.152, Lateral Meniscus 0.028 vs 0.135, ACL 0.077 vs 0.138, Fracture 0.015 vs 0.068 (s6_plane.py).

**Minutes are noise.** `train_minutes` for the 25 seeds of the base config: mean 43.6, SD 9.4, range 28.1–64.9 (s7_labels.py). Across configs, minutes span 9.4–43.6. `train_minutes` removes pauses but not slow-downs from other jobs — [dashboard.py L210–215](/Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py).

**What the competition scores.** The main metric is macro AUC, "only ranking matters". The efficiency prize is `AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400`, about 0.044 AUC per hour of submission runtime — [docs/competition.md L16–22](/Users/maxranft/Projects/rsna-knee/docs/competition.md).
- The dashboard itself estimates head inference at 0.1 min of a 17.0 min pipeline (preprocess 12, encode 3.4, setup 1.5) — [dashboard.py L551–561](/Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py).
- The code already calls accuracy "a display metric … AUC, which is rank-based, decides" — [dashboard.py L148–154](/Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py).

### Inferences
- **The plane adds no decision-relevant information.** β_auc = 0.36 is mechanical: higher-AUC models clear more studies past a fixed 0.5 cut on imbalanced labels.
- **β_time cannot be identified.** The minutes regressor is mostly within-config noise, and its between-config part is confounded with epochs, d and teacher use.
- **If the efficiency axis matters,** the relevant x is measured inference runtime per study on Kaggle. Head choice barely moves it (0.1 of 17 min); the encoder and preprocessing dominate. A better y is gold or report AUC, not accuracy.
- **Delta and gamma** (learning-curve sensitivities) were not in scope beyond noting that they use the same validation AUC.

### Gaps
- Kaggle inference timings per head were not measured. The dashboard flags its efficiency numbers as `"measured": False`.

## Q6. Bottom line for the final week (submission 2026-10-22): when to trust report-AUC gains, and which labels have headroom

### Takeaway
**Use report macro AUC to choose between configs and ensembles.**
- Its gains transfer to gold at about 1.1:1. The two-level 90% CI is [0.57, 1.64] over all configs and [0.36, 1.95] within the current top cluster.
- Gold reverses report only when report differences are below about 0.005, and there gold itself cannot resolve anything.
- Do not let a gold difference under about 0.015–0.02 override a replicated report difference.

**Do not use per-label report gains as gold proxies,** except for Effusion, Fracture, Medial OA and (in the top regime) MCL. For Synovitis, ACL and Lateral OA they carry no gold signal or a negative one.

**Headroom against the labeller ceiling is largest for:**

| Label | Gold AUC | Labeller ceiling | Gap |
|---|---|---|---|
| Medial Meniscus | 0.59 | 0.90 | 0.31 |
| MCL | 0.65 | 0.93 | 0.27 |
| Synovitis | 0.52 | 0.71 | 0.19 |
| Lateral Meniscus | 0.69 | 0.87 | 0.17 |
| ACL | 0.72 | 0.86 | 0.14 |
| PF OA | 0.74 | 0.87 | 0.13 |
| Fracture | 0.78 | 0.91 | 0.13 |

Medial OA, Effusion, Contusion and Baker's are at or above their labeller ceiling.

### Cited Findings
**Headroom table** (base config, mean of 25 seeds, gold AUC with a 90% gold-study bootstrap; s7_labels.py; labeller AUC from `data/labeller/all-v2/gold_eval_all.json`). Base macro: gold 0.746 [0.714, 0.778], report 0.799.

| Label | Gold pos | Labeller vs gold AUC | Base report AUC | Base gold AUC [90%] | Labeller − gold | Transfer evidence (Q4) | Model β |
|---|---|---|---|---|---|---|---|
| Medial Meniscus | 26 | 0.900 | 0.770 | 0.592 [0.474, 0.701] | **0.308** | ≈ 0 | 0.89–1.06 |
| MCL | 9 | 0.927 | 0.744 | 0.654 [0.483, 0.803] | **0.273** | > 0 in top regime | 1.00–1.05 |
| Synovitis | 27 | 0.714 | 0.843 | 0.521 [0.408, 0.639] | **0.193** | ≤ 0 | 0.89–0.98 |
| Lateral Meniscus | 23 | 0.865 | 0.719 | 0.692 [0.587, 0.793] | 0.173 | ≈ 0 to + | 0.90–0.96 |
| ACL | 24 | 0.862 | 0.815 | 0.718 [0.610, 0.822] | 0.144 | ≈ 0 | 1.16–1.30 |
| PF OA | 21 | 0.870 | 0.792 | 0.738 [0.629, 0.840] | 0.132 | ≈ 0 to + | 0.98–1.05 |
| Fracture | 18 | 0.906 | 0.782 | 0.778 [0.658, 0.884] | 0.128 | **> 0** | 1.20–1.34 |
| Lateral OA | 11 | 0.795 | 0.808 | 0.700 [0.553, 0.839] | 0.095 | ≈ 0 | 0.85–0.92 |
| Baker's | 12 | 0.889 | 0.801 | 0.860 [0.767, 0.937] | 0.029 | ≈ 0 to + | 0.74–0.83 |
| Medial OA | 15 | 0.949 | 0.858 | 0.943 [0.897, 0.980] | 0.006 | **> 0** | 0.73–0.77 |
| Contusion | 19 | 0.835 | 0.802 | 0.856 [0.760, 0.935] | −0.021 | ≈ 0 | 1.16–1.34 |
| Effusion | 35 | 0.820 | 0.859 | 0.904 [0.844, 0.954] | −0.084 | **> 0** | 0.65–0.74 |

**Supporting evidence**
- Macro transfer and pair agreement come from Q2 and Q3 (s5_boot.py).
- The project ground rules already say "When they disagree, gold wins. Gold differences below about 0.02 are noise" — [CLAUDE.md](/Users/maxranft/Projects/rsna-knee/CLAUDE.md); [experiments.md L5](/Users/maxranft/Projects/rsna-knee/experiments.md).
- The labeller log says "Rewording against 58 studies has run its course" for Synovitis, Lateral OA and Contusion; the next lever is a stronger LLM — [docs/labeller.md](/Users/maxranft/Projects/rsna-knee/docs/labeller.md).

### Inferences
**Selection rule for the final week (S3/S4 model choice):**
1. Rank candidates by report macro, using seed means (≥3 seeds; seed SD 0.0015).
2. Treat |Δreport| < 0.003 as a tie. Break ties on gold only if |Δgold| ≥ about 0.02, otherwise on cost.
3. Never pick by per-label gold. The "gold wins" rule is sound as an arbiter for large gaps. In the regime the project is now in (top configs within 0.0015 report of each other), it should be read as "gold cannot overrule report", not "pick the gold-best".
- On this rule, d 256 (report +0.0006, gold +0.0054, half the training time) and the base config are a tie. Either is defensible, and d 256 is cheaper.

**Where to invest, by strategy:**
- **S1/S2, labels: Synovitis, then ACL.** Report-label gains do not reach gold for either, so model work aimed at them is wasted until the labels match gold's definitions. Synovitis is at chance on gold. Raising it to about 0.70 would be worth about +0.015 macro on its own.
- **S3, model and features: Medial Meniscus, MCL, Lateral Meniscus.**
  - Medial Meniscus: the labeller ceiling is high (0.90), so the labels are not the bottleneck, yet gold is 0.59 and transfer is about zero. That points to a missing image signal (meniscus-relevant slots or series, resolution) or a shortcut that fails on the enriched gold set, not to label noise. This is an inference.
  - MCL: transfers in the top regime and has the second-largest gap, but only 9 gold positives, so any gold read on it is very noisy.
- **Low priority: Medial OA, Effusion, Contusion, Baker's.** They are at or above the labeller ceiling, and the model betas show Medial OA and Effusion are already saturating (0.65–0.77).
- **The macro rule is safe despite the weak labels.** Macro transfer works because the labels that transfer well (Effusion, Fracture, Medial OA, MCL) carry it. A report macro gain that comes mainly from ACL, Contusion or Synovitis is less likely to show on gold than one spread across labels. A cheap guard is to check which labels a candidate's report gain comes from.

**Dashboard fixes (small):**
1. Apply MIN_SEEDS to the betas and the plane, or use WLS.
2. Add the two-level (gold-study × config) bootstrap for transfer intervals.
3. Reword the transfer note: real transfer means β ≳ 1, and an interval through 0 means "not enough gold positives to tell".
4. Drop the plane's accuracy and β_time, or replace them with inference runtime and AUC.
5. Re-point "Current reference" to the current best config, if that is intended.

### Gaps
- **Population shift.** The hidden test set is about 1,300 studies. Whether its label distribution looks like the enriched gold set or the report-only population is unknown, and that decides whether the gold-specific failures (Synovitis, Medial Meniscus) cost as much on the leaderboard as on the 58 gold studies.
- **Ensembles.** Transfer for rank ensembles of many seeds or configs (the actual "accurate" submission) was not measured. It is likely the same direction with less noise, but that is unverified.
- **Labeller-vs-gold as a ceiling.** The labeller–gold AUC is not a hard ceiling for an image model: Effusion and Contusion already exceed it. The gaps are a guide to headroom, not a bound.
