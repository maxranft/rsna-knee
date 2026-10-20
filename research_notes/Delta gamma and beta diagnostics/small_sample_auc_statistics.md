# Small-sample ROC AUC statistics: 58 gold studies, 12 labels, few seeds

Scope: how much uncertainty a 58-study gold set really carries for macro ROC AUC over 12 binary labels, which comparison methods suit it (paired bootstrap, Welch on seed means, Cliff's delta, individuals control chart), and the pitfalls (selection bias, leaderboard noise). Worked numbers come from synthetic simulations. No project data was read. The scripts are in the session scratchpad (`sim_single_auc.py`, `sim_macro.py`, `sim_seeds_cc.py`), and the setup is described below so the runs can be repeated.

**Simulation setup used throughout (synthetic only).** Binormal scores: negatives ~ N(0,1), positives ~ N(a,1), true AUC = Φ(a/√2).
- **Macro simulations:** 12 labels, expected positives at n = 58 of [5, 6, 8, 10, 12, 14, 16, 18, 21, 24, 27, 30], drawn at random in each replicate from those prevalences.
- **True per-label AUCs:** 0.68 to 0.90, macro 0.793.
- **Label correlation:** a latent liability with exchangeable correlation ρ_y.
- **Score-noise correlation across labels:** ρ_e.
- **ρ_m:** the within-class correlation between the two compared models' scores. It is the key nuisance parameter for paired comparisons. Two configs fine-tuned from the same backbone on the same data plausibly sit at ρ_m ≈ 0.8–0.95; different architectures or teams sit lower.
- **Replicates and resamples:** 600 replicates per scenario at n = 58, with 1,000 paired bootstrap resamples over studies each.

---

## 1. Variance of a single AUC with small n (Hanley & McNeil, DeLong, CI width at 8 positives / 50 negatives)

### Takeaway
With 8 positives and 50 negatives, a single-label AUC has a sampling SD of about 0.06–0.11, so its 90% CI spans ±0.10 (AUC 0.9) to ±0.18 (AUC 0.6). Per-label gold AUCs are close to uninformative for model comparison. Standard Wald and bootstrap intervals under-cover at small n and high AUC: they give 73–85% actual coverage for a nominal 90% when positives are 5–8 and AUC is 0.8–0.9.

### Cited Findings
- Hanley & McNeil (1982) showed that the ROC area equals the probability that a randomly chosen diseased case is ranked above a randomly chosen non-diseased case, which is the Wilcoxon/Mann-Whitney statistic. They used this link to derive closed-form expressions for its sampling variability and for sample-size planning — [Hanley & McNeil 1982 PDF](https://math.ucdavis.edu/~saito/data/roc/hanley-mcneil-auc.pdf); [Harvard catalogue entry](https://hcp.hms.harvard.edu/publication/meaning-and-use-area-under-receiver-operating-characteristic-roc-curve)
- The Hanley-McNeil SE is SE(A) = sqrt([A(1−A) + (n_pos−1)(Q1−A²) + (n_neg−1)(Q2−A²)] / (n_pos·n_neg)), with Q1 = A/(2−A) and Q2 = 2A²/(1+A). Q1 and Q2 come from an exponential-distribution approximation — [sample-size calculator reproducing HM 1982](https://nivedi.res.in/Nadres_v2/sample_size_calculator/roc_single.php)
- A 1994 simulation study reportedly found that the Hanley-McNeil method can underestimate the minimum sample size, and it proposed a binormal alternative. This comes from a search-result abstract only, not checked against full text — [BVS/Medline record](https://pesquisa.bvsalud.org/sms/resource/pt/mdl-8169102)
- DeLong, DeLong & Clarke-Pearson (1988), Biometrics 44(3):837–845, give a distribution-free variance and covariance for correlated AUCs, based on generalized U-statistics. The method is meant for comparing AUCs measured on the same individuals — [Duke Scholars record](https://scholars.duke.edu/publication/712661); [MedCalc ROC comparison](https://www.medcalc.org/en/book/roc-comparison.php)
- Stata's `roctab` uses the DeLong variance by default because it performed better than the alternatives in the scenarios the author examined — [Stata roctab manual](https://www.stata.com/manuals/rroctab.pdf)
- Feng, Cortese & Baumgartner (2017, Stat Methods Med Res 26(6):2603–2621) compared 29 parametric and non-parametric CI methods for AUC at small sample sizes. They note that systematic small-sample study had been "by and large missing". Their best-method conclusions were not retrieved — [Padova repository](https://www.research.unipd.it/handle/11577/3172976); [auRoc R package](https://cran.r-project.org/web/packages/auRoc/refman/auRoc.html)
- A 2026 large-scale study of CIs in medical image analysis found BCa coverage significantly higher than percentile coverage for micro AUC at every one of nine sample sizes. The coverage numbers were not retrieved — [arXiv 2601.17103](https://arxiv.org/pdf/2601.17103)

### Simulation results (own, synthetic; `sim_single_auc.py`)
- **True sampling SD of one AUC at n = 58**, from 20,000 Monte Carlo draws, binormal. The 90% half-width is 1.645 × SD.

  | positives / negatives | AUC 0.6 | AUC 0.7 | AUC 0.8 | AUC 0.9 |
  |---|---|---|---|---|
  | 5 / 53 | 0.134 (±0.22) | 0.121 (±0.20) | 0.102 (±0.17) | 0.071 (±0.12) |
  | 8 / 50 | 0.108 (±0.18) | 0.100 (±0.16) | 0.084 (±0.14) | 0.058 (±0.095) |
  | 15 / 43 | 0.085 (±0.14) | 0.078 (±0.13) | 0.066 (±0.11) | 0.046 (±0.075) |
  | 30 / 28 | 0.074 (±0.12) | 0.068 (±0.11) | 0.058 (±0.095) | 0.040 (±0.065) |

- **Hanley-McNeil vs DeLong:** the mean DeLong SE matches the true SD within about 2% in every cell. The HM formula overstates the SD by 15–30% when positives are few and AUC is high under binormal data: 0.098 vs 0.084 at 8/50, AUC 0.8, and 0.075 vs 0.058 at AUC 0.9. At 30/28 it is within about 6%.
- **Actual coverage of nominal 90% CIs at 8 positives / 50 negatives** (2,000 replicates, 2,000 resamples):

  | true AUC | HM Wald | DeLong Wald | DeLong logit | percentile, unstratified | percentile, stratified | BCa, stratified |
  |---|---|---|---|---|---|---|
  | 0.7 | 0.91 | 0.87 | 0.90 | 0.88 | 0.87 | 0.90 |
  | 0.8 | 0.91 | 0.84 | 0.88 | 0.86 | 0.85 | 0.88 |
  | 0.9 | 0.90 | 0.78 | 0.84 | 0.81 | 0.80 | 0.85 |

  HM covers well only because its SE is inflated: its intervals are 15–30% wider.
- **Coverage at 5 positives / 53 negatives, AUC 0.9:** DeLong Wald 0.73, percentile 0.74–0.76, BCa 0.80, logit-DeLong 0.82.
- **Coverage at 15/43 and 30/28:** all methods reach 0.85–0.92. BCa and logit-DeLong are closest to nominal (0.90–0.92).
- **Larger n (HM formula, AUC 0.8):** at n = 1,300 with 112 positives (8.6%), SE 0.026 (±0.042). With 390 positives (30%), SE 0.015 (±0.024). At n = 4,349 with 374 positives, SE 0.014 (±0.023).

### Inferences
- Per-label gold AUCs, such as the "weakest 3 labels" logged in experiments.md, carry ±0.10–0.20 of 90% noise on labels with 5–15 positives. Ranking labels by gold AUC, or acting on a single label's change, is unreliable unless the change exceeds about 0.15–0.25.
- If CIs are needed for one gold label, prefer BCa or logit-transformed DeLong over percentile or plain Wald. Even those under-cover at 5–8 positives and high AUC, so treat the 90% interval as roughly an 80–85% interval there.

### Gaps
- Feng et al. (2017) and Obuchowski & Lieber (1998, Acad Radiol, small-sample AUC CIs) are the canonical small-sample references. Neither full text was retrieved, so their recommended methods are not quoted here.
- No source was found that gives coverage tables for exactly 8/50. The numbers above are from simulation only.

---

## 2. Variance of a macro (12-label mean) AUC; validity of a paired bootstrap over studies; percentile vs BCa

### Takeaway
On 58 studies, macro AUC over 12 labels has a sampling SD of about 0.022 if label errors are independent, rising to about 0.028 at modest inter-label correlation and about 0.045 at strong correlation. The **paired** difference between two models is much tighter: SD 0.010–0.031 depending mainly on how correlated the two models' scores are (ρ_m). A paired bootstrap over studies is the right tool. It keeps both the label correlation and the model pairing, and in simulation its SE was within about 7% of the true SD, with 87–92% coverage for a nominal 90% interval on the difference.

### Cited Findings
- Hanley & McNeil (1983) showed that two AUCs from the same cases are correlated through the paired ratings, and that this correlation must be used to shrink the SE of their difference. They gave a binormal lookup table from rating correlation to area correlation — [PubMed 6878708](https://pubmed.ncbi.nlm.nih.gov/6878708/); [MetricGate summary](https://metricgate.com/docs/hanley-mcneil-roc-comparison/)
- DeLong et al. (1988) estimate the full covariance matrix of correlated AUCs non-parametrically, which allows paired tests on the same cases — [Duke Scholars](https://scholars.duke.edu/publication/712661)
- A 2026 methodological evaluation of Obuchowski-McClish vs Hanley-McNeil sample-size methods found that a strong between-test correlation (ρ = 0.8) cut required sample sizes by 49% (discrete) and 68% (continuous) relative to a weak one (ρ = 0.3) — [PMC12924612](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12924612/)
- Bouthillier et al. (2021) recommend pairing because it removes shared sources of variance: SD(A−B) ≤ σ_A + σ_B when paired — [arXiv 2103.03098](https://arxiv.org/abs/2103.03098)
- Agarwal et al. (2021) validated percentile, basic, BC and BCa stratified-bootstrap CIs against "true" coverage. Percentile CIs gave the best coverage at small width for N ≥ 10 runs. With 3 runs, bootstrap CIs under-covered and "might require a larger nominal coverage rate" — [NeurIPS 2021 paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)
- In general bootstrap guidance, all bootstrap CIs tend to under-cover at small n, and BCa usually comes closer to nominal than the plain percentile interval — [MetricGate bootstrap CI notes](https://metricgate.com/docs/bootstrap-confidence-interval); the AUC-specific confirmation is in [arXiv 2601.17103](https://arxiv.org/pdf/2601.17103)

### Simulation results (own, synthetic; `sim_macro.py`)
- **SD of one model's macro AUC at n = 58:**
  - Independent labels: 0.022, which matches 0.074/√12 from the mean per-label SD of 0.074.
  - ρ_y = 0.4, ρ_e = 0.3: 0.027–0.029. The average inter-label correlation of AUC errors is only about 0.06, yet the 11 off-diagonal terms per label inflate the variance by about 1 + 11 × 0.06 ≈ 1.66.
  - ρ_y = 0.8, ρ_e = 0.7: 0.045–0.046, with inter-label error correlation of about 0.32.
  - The 90% CI half-width for absolute macro AUC is therefore ±0.036, ±0.046 or ±0.075.
- **SD of the paired macro difference, B − A, true difference 0, n = 58:**

  | label correlation | ρ_m 0.5 | ρ_m 0.8 | ρ_m 0.9 | ρ_m 0.95 |
  |---|---|---|---|---|
  | none | — | 0.0156 | 0.0121 | — |
  | modest (ρ_y 0.4, ρ_e 0.3) | 0.031 | 0.019 | 0.0136 | 0.010 |
  | strong (ρ_y 0.8, ρ_e 0.7) | — | 0.0305 | 0.0227 | — |

- **Paired bootstrap check (1,000 resamples over studies, recomputing all 12 AUCs per resample):**
  - The mean bootstrap SE was within −7% to +6% of the true SD in all eight null scenarios.
  - Coverage of the 90% percentile interval for the true difference was 0.87–0.91 under the null, 0.87–0.92 at true differences of +0.02 or +0.04, and 0.88 at n = 1,300.
- **Calibration of "P(better)" (share of resamples where B wins) when the true difference is 0:**
  - P > 0.95 in 2.8–7.2% of datasets.
  - P > 0.90 in 6.3–12.8%.
  - P > 0.75 in 20–30%.
  - The median is about 0.50.
  - It therefore behaves roughly like 1 − (one-sided p-value). It is not a posterior probability that B is better on new data.
- **Undefined labels in resamples:** on a fixed gold set, the chance that a resample has zero positives for a label is (1 − k/58)^58. That is 0.54% for k = 5, 0.18% for k = 6 and 0.02% for k = 8, so about 0.7% of resamples lose a label. (In the random-prevalence simulation the rate was about 6%, because some replicates had only 1–3 positives.) Both models drop the same label in the same resample, so pairing is preserved, but the macro in that resample is over 11 labels.

### Inferences
- Bootstrap over **studies**, never over (study, label) cells. Resampling studies keeps the inter-label correlation, which the simulation shows can double the SD of macro AUC. Resampling labels or cells would understate it.
- The percentile interval on the **paired difference** was close to nominal even at n = 58, because the difference distribution is roughly symmetric. For a **single** label's AUC at 5–8 positives, BCa or logit-DeLong is better than percentile (Section 1).
- The main driver of comparison noise is ρ_m, not n alone. Two near-identical configs (ρ_m ≈ 0.95) can be separated at Δ ≈ 0.025. Configs with different inputs or architectures (ρ_m ≈ 0.5–0.8) need Δ ≈ 0.05–0.08 on gold. The project can measure ρ_m directly from paired OOF predictions on the 58 gold studies; it needs only model outputs, not report text.
- **Pooled OOF AUC:** when gold studies are spread across folds, a pooled AUC mixes score scales from different fold models. Ranking across folds then adds a small extra noise or bias term. Forman & Scholz (2010, SIGKDD Explorations, "Apples-to-apples in cross-validation studies") discussed this; that reference is from background knowledge and was not verified here. Rank-normalising within fold before pooling is one mitigation, but it changes the metric, so it would have to be applied consistently.

### Gaps
- No published analytic variance for a *macro* average of correlated AUCs was found. The DeLong covariance matrix gives it in principle as a 12×12 structural-component covariance averaged with weights of 1/12, but no paper was found that tabulates it for multi-label settings.
- The real ρ_m and inter-label error correlation of the project's models are unknown and should be measured from OOF outputs.

---

## 3. Is "gold differences below about 0.02 are noise" reasonable? Minimum detectable difference at n = 58

### Takeaway
"Below 0.02 is noise" is a sound *floor*: 0.02 is about 1–1.5 SD of the paired gold difference for similar models (ρ_m 0.8–0.9). It is **not** a sound *decision threshold*. A true +0.02 gain shows up as P(better) > 0.95 only 28–44% of the time, and an observed +0.02 arises from pure noise 7–13% of the time (one-sided) for a single comparison. The minimum difference detectable with 80% power at n = 58 is about 0.034 (ρ_m 0.9) to 0.047 (ρ_m 0.8), and about 0.077 if ρ_m ≈ 0.5.

### Cited Findings
- A clinical sample-size example (MedCalc, using the Hanley-McNeil paired method) shows that detecting AUC 0.825 vs 0.900 (Δ = 0.075) with a rating correlation of 0.4, α = 0.05 and 80% power needs 133 positives plus 266 negatives = 399 cases — [MedCalc sample size for ROC comparison](https://www.medcalc.org/en/manual/sample-size-ROC-comparison.php)
- Higher between-test correlation sharply reduces the sample size needed for a paired AUC comparison: 49–68% fewer cases at ρ = 0.8 than at ρ = 0.3 — [PMC12924612](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12924612/)
- Bouthillier et al. (2021) found that improvements smaller than the variance seen on a given test set "will not generalize", and that this variance is not negligible compared with typical published improvements — [arXiv 2103.03098](https://arxiv.org/abs/2103.03098)

### Simulation results (own, synthetic; `sim_macro.py`)
- **Minimum detectable difference from the simulated paired SDs**, with modest label correlation. MDD80 means 80% power, one-sided α = 0.05, i.e. the 90% interval excludes 0. z(0.02) is 0.02 divided by the SD.

  | ρ_m | SD of Δ | 90% half-width | MDD80 (one-sided 0.05) | MDD80 (two-sided 0.05) | z(0.02) | P(\|Δ_obs\| > 0.02 under the null) |
  |---|---|---|---|---|---|---|
  | 0.5 | 0.031 | 0.051 | 0.077 | 0.086 | 0.65 | 52% |
  | 0.8 | 0.019 | 0.031 | 0.047 | 0.053 | 1.06 | 29% |
  | 0.9 | 0.0136 | 0.022 | 0.034 | 0.038 | 1.47 | 14% |
  | 0.95 | 0.010 | 0.016 | 0.025 | 0.028 | 2.0 | 4.6% |

- **True gain of +0.02 on every label, n = 58** (direct simulation, 600 replicates):
  - **ρ_m 0.8:** P(better) > 0.95 in 28% of replicates, > 0.90 in 41% and > 0.75 in 65%. The median P(better) is 0.85, and the observed difference has the **wrong sign 14%** of the time.
  - **ρ_m 0.9:** 44%, 59% and 81%, median 0.93, wrong sign 4.8%.
- **True gain of +0.04:** P(better) > 0.95 in 71% (ρ_m 0.8) and 90% (ρ_m 0.9). The wrong sign appears in 1.3% and 0.3%.
- **Report-label set (n = 4,349, noisy labels with sensitivity 0.85 and specificity 0.95, noise independent of the models, ρ_m 0.8):**
  - A true +0.02 gain is measured as +0.0152, attenuated by a factor of 0.76.
  - The SD of the paired difference is 0.0018, so z ≈ 8.4, against z ≈ 1.06 for the same gain on gold.
  - Under the null, the SD of the difference is 0.0019.

### Inferences
- Keep 0.02 as "do not even look below this". Treat 0.02–0.05 on gold as a grey zone to be settled by the report-label OOF signal, which is roughly 8× more sensitive in z terms, or by more seeds. Treat gold as the decider only for large differences, or as a veto when it clearly disagrees, for example when the 90% interval excludes 0 in the opposite direction.
- A better rule than a fixed 0.02 is to scale the threshold by the measured paired SD for each comparison. That is exactly what the paired bootstrap's interval does, since ρ_m varies from about 0.5 to 0.95 across kinds of change.
- **"Gold decides" has a power problem:** for a true +0.02 change, gold says the wrong direction 5–14% of the time and is inconclusive most of the time. Gold should override the report-label signal only when the gold interval excludes zero against it. A disagreement inside the noise band is expected, not evidence.
- **Caveat on the report-label signal:** in the real project, models are trained on the same labelling function used to score them. Report-label errors are therefore *not* independent of model errors, and a model that better reproduces the labeller's mistakes is rewarded. That is a bias, not variance, and the bootstrap cannot reveal it. This is why gold remains necessary as a check.

### Gaps
- No literature source gives a minimum detectable difference specifically for *macro* AUC on about 60 cases. The figures above are simulation-based and depend on the assumed ρ_m, label correlation and per-label AUC range.

---

## 4. Seed variance: how many seeds, Welch with 3 seeds, Cliff's delta with 3×3 pairs, IQM / stratified bootstrap

### Takeaway
Three seeds per config is too few for either test to be informative. Welch with 3 vs 3 has a median 90% CI half-width of 1.76 σ_seed (2.7 σ at the 90th percentile) and only 22% power to detect a 1-σ_seed difference. Cliff's delta on 3×3 pairs can take only 10 values, and its most extreme value (±1) corresponds to an exact two-sided p of 0.10. About 17 seeds per arm are needed for 80% power at a 1-σ difference, and 6 at a 2-σ difference. More importantly, seed tests measure only training noise on a *fixed* gold set. They ignore evaluation-set noise, which on 58 studies is likely the larger term.

### Cited Findings
- **Bouthillier et al. (2021, MLSys)** — [arXiv 2103.03098](https://arxiv.org/abs/2103.03098):
  - **Sources of variance:**
    - Each source was randomized 200 times. Data sampling (bootstrap of train/test) "stands out as the most important source of variance".
    - Weight initialization is generally under 50% of the bootstrap variance, on a par with SGD data order.
    - A binomial test-set model explains the data-sampling variance well. The test sets were Glue-RTE n = 277, SST-2 n = 872 and CIFAR-10 n = 10,000.
  - **Decision rules (k = 50 splits):**
    - Single-point comparisons had about 10% false positives and about 75% false negatives.
    - Comparing averages against a typical-published-improvement threshold had under 5% false positives but about 90% false negatives.
    - Their P(A>B) test with γ = 0.75 had about 5% false positives and about 30% false negatives.
    - They recommend randomizing as many sources of variation as possible (init, data order, augmentation, splits, HPO), pairing, out-of-bootstrap splits, and reporting all procedures within significance bounds rather than only the best.
  - **Runs needed:** about 29 runs to reliably detect P(A>B) > 0.75. Detecting P(A>B) < 0.6 needs more than 500 runs.
- **Agarwal et al. (2021, NeurIPS)** — [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf); [arXiv 2108.13264](https://arxiv.org/abs/2108.13264):
  - 3–10 runs per task are prevalent in deep RL. Folk wisdom says 20–30 runs are enough, but CIs on sample medians suggest 50–100 runs in Atari 100k.
  - Their stratified bootstrap resamples runs with replacement independently within each task, then recomputes the aggregate. Percentile CIs were reliable from N = 10 runs, and with 3 runs bootstrap CIs under-covered.
  - IQM is the 25% trimmed mean: it drops the top and bottom 25% of all runs × tasks. It gives much smaller CIs than the median and is more robust than the mean.
- **Picard (2021)** scanned up to 10⁴ seeds on CIFAR-10 and fewer on ImageNet with pretrained models. Even though seed variance is "not very large", it is "surprisingly easy to find an outlier that performs much better or much worse than the average" — [arXiv 2109.08203](https://arxiv.org/abs/2109.08203). A secondary summary puts the seed-only spread at up to 1.82% accuracy on CIFAR-10 and about 0.5% on ImageNet; this figure was not verified in the primary text — [alphaXiv overview](https://www.alphaxiv.org/overview/2109.08203)
- **Dodge et al. (2020)** fine-tuned BERT on 4 GLUE tasks hundreds of times, varying only the seed, and found "substantially different results" at identical hyperparameters. Weight initialization and data order contribute comparably, and some initializations are good across tasks. They released 2,100 trials and proposed early-stopping unpromising seeds — [arXiv 2002.06305](https://arxiv.org/abs/2002.06305)

### Simulation results (own, synthetic; `sim_seeds_cc.py`, 200,000 replicates, seed scores ~ N(μ, σ²))
- **Welch t, 3 vs 3:**
  - Median Welch df is 3.2, and df < 3 in 42% of replicates.
  - Median 90% CI half-width is 1.76σ; the 90th percentile is 2.73σ.
  - The false-positive rate at nominal two-sided 0.10 is 7.9%, slightly conservative.
  - Power at two-sided α = 0.10 is 12% at Δ = 0.5σ, 22% at 1σ, 58% at 2σ and 87% at 3σ.
  - The sign of the seed-mean difference is wrong 27% of the time at Δ = 0.5σ and 11% at 1σ.
- **Welch, 5 vs 5:** half-width 1.16σ, power at 1σ 40%. **10 vs 10:** half-width 0.77σ, power at 1σ 69%.
- **Seeds per arm for 80% power** (two-sided α = 0.05, t-test): 64 at Δ = 0.5σ, 17 at 1σ, 9 at 1.5σ and 6 at 2σ.
- **Cliff's delta, 3×3 — exact null distribution over the 20 rank arrangements:**
  - Possible values are ±1, ±7/9, ±5/9, ±1/3 and ±1/9, in steps of 2/9.
  - P(δ = +1) = 0.05, so P(|δ| = 1) = 0.10, and P(|δ| ≥ 7/9) = 0.20.
  - Under the null, δ ≥ 0.474 (the usual "large" cut-off) occurs 20% of the time.
  - Power to obtain δ = +1 is 26% at Δ = 1σ, 62% at 2σ and 89% at 3σ.

### Inferences
- **Cliff's delta on 3×3:** report it only as a descriptive ordinal summary. "δ = 1" means only that all three candidate seeds beat all three reference seeds, which has 1-in-20 odds under the null (one-sided). Asymptotic CIs for δ (Cliff / Feng-Cliff type) are not valid at n = 3 per arm, and the bootstrap of δ on 3×3 is degenerate. Use the exact permutation distribution above instead.
- **Welch with 3 seeds:** it is valid in level (slightly conservative), but the interval is so wide (±1.8σ) that it confirms only differences of about 3σ or more. If σ_seed on gold macro AUC is about 0.01, that means only differences of about 0.03 or more.
- **What the seed test leaves out:** both configs are scored on the same 58 studies, so a seed-level test conditions on that particular gold sample. A "significant" Welch result says "on these 58 studies config B is reliably higher", not "B generalizes better".
  - The generalization uncertainty is roughly Var(Δ) ≈ σ²_eval,Δ + 2σ²_seed/k.
  - σ_eval,Δ is the paired-bootstrap SD of the difference: about 0.014–0.019 on gold (Section 3).
  - With k = 3 and σ_seed = 0.01, the seed term contributes only √(2·10⁻⁴/3) ≈ 0.008, so evaluation-set noise dominates. Adding seeds beyond about 5 helps little on gold. It helps much more on the n = 4,349 report-label set, where σ_eval,Δ ≈ 0.002.
- **Practical combined procedure:** a two-level bootstrap. Resample seeds within each config, in the spirit of Agarwal's stratified bootstrap with configs as strata, and resample studies jointly and paired across configs. Alternatively, run the paired study-bootstrap on seed-ensembled predictions and add the seed-mean variance. Report P(better) from that combined distribution.
- **Don't import Bouthillier's γ = 0.75 rule:** their P(A>B) is the share of *paired training runs* (over splits and seeds) in which A wins, and it needs about 29 runs. The project's "P(better)" is a bootstrap share for one fixed pair of models, roughly 1 − one-sided p. Under the null it exceeds 0.75 20–30% of the time (Section 2), so the project's P(better) should not be read on Bouthillier's γ = 0.75 scale.
- **IQM:** with 3 seeds the IQM reduces to about the median and gains nothing. It becomes useful with 8 or more seeds, or when pooling seeds × folds.

### Gaps
- No source was found that quantifies seed SD of macro AUC for small medical-imaging CNNs on about 60-case evaluation sets. The project's own seed logs are the only way to estimate σ_seed.
- Hierarchical (two-level) bootstrap references for ML evaluation were not retrieved in this session.

---

## 5. Multiple comparisons and selection bias: winner's curse on a 58-study gold set; remedies

### Takeaway
Adopting the best of many configs evaluated on the same 58 gold studies produces a large optimistic bias. With a paired-difference SD of about 0.0125, the best of 10 truly-equal configs beats the reference by +0.014 on average and exceeds +0.02 27% of the time. For the best of 50, the figures are +0.020 and 49%. When real effects are small (spread τ = 0.005), the winner's observed gain is about 4× its true gain. Cawley & Talbot (2010) show that this selection overfitting is often as large as the differences between algorithms.

### Cited Findings
- Cawley & Talbot (2010, JMLR 11:2079–2107) argue that low variance in a model-selection criterion is at least as important as unbiasedness. Over-fitting the selection criterion produces effects "of comparable magnitude to differences in performance between learning algorithms". Evaluation practices that reuse the selection data are biased. The findings apply to "any model selection practice involving the optimisation of a model selection criterion evaluated over a finite sample of data" — [JMLR abstract](https://www.jmlr.org/papers/v11/cawley10a.html); [full PDF](https://jmlr2020.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf)
- Bouthillier et al. (2021) recommend highlighting all procedures within the significance bounds, not just the best-performing one — [arXiv 2103.03098](https://arxiv.org/abs/2103.03098)
- Blum & Hardt (2015, "The Ladder") formalize adaptive overfitting to a reused holdout or leaderboard. They propose the Ladder algorithm, which reports a new best score only when it beats the previous one by a margin, with guarantees in a fully adaptive model, and they show high utility on real Kaggle submission files — [arXiv 1502.04585](https://arxiv.org/abs/1502.04585)
- Roelofs et al. (2019, NeurIPS) meta-analysed over 100 Kaggle competitions and found "little evidence of substantial overfitting" from test-set reuse. This holds for large holdouts with many teams and is not evidence that 58 cases are safe to reuse — [NeurIPS abstract](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning)

### Simulation results (own, synthetic; `sim_seeds_cc.py`)
- **All candidates truly equal to the reference.** Each candidate is compared with the same reference, so pairwise differences have correlation 0.5.

  | SD of Δ | K = 5 | K = 10 | K = 20 | K = 50 |
  |---|---|---|---|---|
  | 0.010 (ρ_m ≈ 0.95) | +0.008 (P>0.02: 9%) | +0.011 (13%) | +0.013 (20%) | +0.016 (30%) |
  | 0.0125 | +0.010 (18%) | +0.014 (27%) | +0.017 (36%) | +0.020 (49%) |
  | 0.016 | +0.013 (31%) | +0.017 (42%) | +0.021 (53%) | +0.025 (67%) |

  Each cell gives E[max observed Δ] and, in brackets, P(max > 0.02). The chance that at least one null candidate reaches z > 1.645 (P(better) > 0.95) is 17% at K = 5, 25% at K = 10, 34% at K = 20 and 47% at K = 50.
- **Real but small effects**, true Δ_k ~ N(0, τ²), with SD of Δ = 0.0125:
  - τ = 0.005: the winner's observed gain is +0.016 (K = 10) or +0.021 (K = 30), but its true gain is only +0.004 or +0.005. The empirical-Bayes shrinkage factor is τ²/(τ² + σ²) = 0.14.
  - τ = 0.01: observed +0.021 or +0.027, true +0.012 or +0.015, shrinkage factor 0.39.
  - The winner had a positive true effect 80–98% of the time, so selection still helps; it just over-reports.

### Inferences
- With "dozens" of configs compared against one reference on gold, an observed gold gain of about 0.02 for the adopted winner is roughly what selection alone produces. Expect a re-evaluation to give back most of it. Each adopted change also compounds: the "reference" itself was a past winner.
- **Remedies, from cheapest:**
  1. **Select on the report-label OOF set (n = 4,349) and use gold only to confirm.** Then the 58 gold studies are looked at once per adopted change, not once per candidate.
  2. **Re-test the winner** with fresh seeds, and ideally fresh folds, before adopting it; report the re-test number, not the selection number.
  3. **Shrink reported gains,** for example by multiplying the observed gold Δ by τ²/(τ² + σ²), with τ estimated from the spread of all candidate Δs minus the bootstrap variance.
  4. **Ladder-style margin:** adopt only if Δ exceeds the previous best by a margin of about 1–2 SD of the paired difference.
  5. **Pre-register** a short list of planned comparisons, and count the number of looks; a Bonferroni or Holm adjustment over K looks is the crude bound.
  6. Keep a small untouched **final holdout** (e.g. 15–20 of the 58 gold studies) for the final accurate/efficient submission choice. This cuts gold power further, so it is only worth it if (1) is adopted.

  The "reusable holdout" (Dwork et al. 2015, Science) is the formal version of (4) and (6); this reference is from background knowledge and was not verified in this session.
- experiments.md should record the number of candidates tried per adopted change, so that the selection bias can be estimated afterwards.

### Gaps
- No source quantified the winner's curse specifically for AUC on 50–100-case validation sets. The numbers are from a Gaussian approximation calibrated to the Section 2 simulations.

---

## 6. Control charts for seed-to-seed AUC: is an individuals chart with σ = MR̄/1.128 appropriate?

### Takeaway
An individuals (I-MR) chart is acceptable only as a **run-health monitor within one fixed config**, to flag a broken or diverged run. It is a poor tool for detecting improvements or comparing configs.
- **Too few points:** with σ and the centre estimated from 5–20 seeds, the false-alarm rate per point is 1.1–5.7% instead of the nominal 0.27%, and the median in-control ARL is 123–270 instead of 370.
- **Heavy tails make it worse:** with t₅ tails, the rate is 1.6% even with 100 Phase-I points.
- **Insensitive by design:** a genuinely +1σ better single run crosses the upper limit only 2.3% of the time, even with known parameters.

### Cited Findings
- With estimated parameters, the false-alarm rate and run length vary from practitioner to practitioner, because each has a different Phase-I sample. That variation becomes small only with very large Phase-I samples — [Univ. of Amsterdam, Qual. Reliab. Eng. Int. paper](https://pure.uva.nl/ws/files/54174975/qre.2658.pdf)
- For individuals charts with moving-range estimates, estimated limits make exceedance events dependent, which raises false alarms after short runs. The common rule of thumb of 20–30 Phase-I subgroups may not be enough for estimated limits to behave like true ones — [NCSU repository study](https://repository.lib.ncsu.edu/server/api/core/bitstreams/f446012d-1074-4ff9-bf39-c57c3125a45c/content)
- Quesenberry (1993), Chen (1997) and Chakraborti (2000) are cited as showing that a large amount of Phase-I data is needed for unconditional chart performance to match the known-parameter case. The minimum Phase-I sample for guaranteed in-control performance is "often large and infeasible", which motivates corrected limits — [JSM 2016 proceedings](https://ww2.amstat.org/meetings/proceedings/2016/data/assets/pdf/389612.pdf); [JSM 2012 proceedings](https://ww2.amstat.org/meetings/proceedings/2012/data/assets/pdf/304524_73278.pdf)
- Bootstrap-based adjustment of control limits has been proposed for charts with estimated parameters, since a larger Phase-I sample alone may not remove excess false alarms — [search summary of MEWMA/MAEWMA literature](https://independent.academia.edu/nesmasaleh4)

### Simulation results (own, synthetic; `sim_seeds_cc.py`, 200,000 Phase-I samples)
- **Normal data, limits x̄ ± 3·MR̄/1.128 from m Phase-I points:**

  | m | CV of σ̂ | mean false alarms per new point | median conditional ARL₀ | 10th-percentile ARL₀ |
  |---|---|---|---|---|
  | 5 | 0.44 | 5.7% | 123 | 5 |
  | 10 | 0.30 | 2.3% | 206 | 15 |
  | 20 | 0.21 | 1.1% | 270 | 34 |
  | 50 | 0.13 | 0.54% | 325 | 78 |
  | 100 | 0.09 | 0.39% | 347 | 121 |

  With known parameters, the false-alarm rate is 0.27% and ARL₀ is 370.
- **Heavy-tailed data (t₅):** mean false-alarm rate 6.8% (m = 5), 2.4% (m = 20) and 1.6% (m = 100). The median ARL₀ is 41–68.
- **Detection of a single run shifted up by +1σ:** 2.3% with known limits (Φ(−2)) and 3–11% with estimated limits, the extra being mostly false-alarm inflation. For a +2σ run it is 16–29%.
- **Efficiency of MR̄/d₂ for i.i.d. data:** it is unbiased (mean 1.00), but its variance relative to the sample SD s gives a relative efficiency of only 0.61–0.69 at m = 5–50. In other words, it needs about 1.6× as many seeds for the same precision of σ.

### Inferences
- The moving-range σ exists to separate short-term variation from drift in a time-ordered process. Seeds of one config are exchangeable i.i.d. draws, so the plain sample SD (or a robust MAD) is more efficient. If the chart's x-axis mixes configs or code changes, MR̄ absorbs real config effects, the limits widen, and "special causes" become hard to interpret.
- On gold macro AUC, the seed-to-seed spread includes the training noise as seen through only 58 studies. The chart's σ is therefore not the noise relevant to "will this config generalize better", which is the paired-bootstrap SD (Section 4).
- To detect small *sustained* improvements across successive runs, EWMA or CUSUM charts are the textbook choice (Montgomery, *Introduction to Statistical Quality Control*). This reference is from background knowledge and the chapter was not retrieved. For config comparison, the paired bootstrap plus seed variance (Section 4) is the appropriate tool.
- **If the chart is kept:**
  - Compute limits from at least 20 seeds of one frozen config, or use bootstrap-calibrated or t-based prediction limits: x̄ ± t₀.₉₉₈₇,ₘ₋₁ · s · √(1 + 1/m).
  - Label any out-of-limits point as "investigate", not "better".
  - Avoid Western Electric run rules, which raise false alarms further.

### Gaps
- No source was found that applies Shewhart charts to ML seed variance specifically. Montgomery's text and Quesenberry (1993) were not retrieved in full, so their specific Phase-I recommendations are not quoted.

---

## 7. Public vs private leaderboard shake-up with about 1,300 test studies: expected macro-AUC noise

### Takeaway
At n ≈ 1,300 (same label mix as gold), one submission's macro AUC has a sampling SD of about 0.006, and the paired difference between two similar submissions has an SD of about 0.003–0.004. If the public leaderboard is about one third of the test set (about 433 studies), the public macro AUC carries SD ≈ 0.010. Public and private scores of one model can differ by ±0.020 (90%) from sampling alone. Two submissions that truly differ by 0.005 swap order on the public leaderboard 16–23% of the time and on the private leaderboard 7–15% of the time.

### Cited Findings
- Roelofs et al. (2019) found little evidence of substantial adaptive overfitting across more than 100 Kaggle competitions. Public-to-private differences are thus mostly sampling noise rather than leaderboard overfitting — [NeurIPS 2019](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning)
- Blum & Hardt's Ladder paper studied a Kaggle challenge with 12,000 test samples (8,400 private, 3,600 public). According to a search summary of that paper, statistical deviations from random public/private splitting were large relative to the differences in mean scores among top teams, while top-50 scores were highly correlated across splits. This was not verified against the full text — [arXiv 1502.04585](https://arxiv.org/abs/1502.04585)
- Kaggle splits the hidden test set into public and private parts, the same for all teams. Public shares vary by competition; one BirdCLEF+ 2026 write-up reports about 34% public and 66% private. Shake-up size varies hugely, and in one extreme case the winner was 1,485th on the public leaderboard — [search summary: DataCamp Kaggle course](https://campus.datacamp.com/courses/winning-a-kaggle-competition-in-python/kaggle-competitions-process?ex=8); [CEUR-WS paper](https://ceur-ws.org/Vol-4283/paper369.pdf)
- The project's competition notes say the test set is about 1,300 studies, the metric is macro AUC over 12 labels, and prevalence differs between train, public LB and private LB. On the efficiency track, if maxAUC ≈ 0.90, 1 hour of runtime is worth about 0.044 AUC — project file `docs/competition.md` (internal)

### Simulation results (own, synthetic; `sim_macro.py`)
- **n = 1,300:**
  - SD of one model's macro AUC: 0.0056–0.0058 (90% ± 0.009).
  - SD of the paired difference: 0.0037–0.0038 at ρ_m 0.8 and 0.0028 at ρ_m 0.9. MDD80 is about 0.009 and 0.007.
  - The paired bootstrap SE was 0.0038 against a true SD of 0.0037, and 90% coverage was 0.88.
  - A true +0.01 gain gave P(better) > 0.95 in 83% of replicates (median P(better) 0.996), which is consistent with an MDD80 of about 0.009.
- **Public share 1/3 (n ≈ 433):** macro SD 0.010, difference SD 0.0067 (ρ_m 0.8) or 0.0050 (ρ_m 0.9).
- **Private share 2/3 (n ≈ 867):** macro SD 0.0068, difference SD 0.0048 or 0.0033.
- **Public − private gap:**
  - For one model: SD 0.0122, 90% ± 0.020.
  - For the *difference* between two of your own submissions: SD 0.0082 (ρ_m 0.8) or 0.0060 (ρ_m 0.9), 90% ± 0.010–0.014.
- **Order-flip probability for a true gap of 0.005:**
  - Public: 23% (ρ_m 0.8) or 16% (ρ_m 0.9).
  - Private: 15% or 6.5%.
  - Full 1,300: 9.4% or 3.7%.
- **For a true gap of 0.01:** public 6.8% or 2.3%, private 1.9% or 0.1%.

### Inferences
- **Choosing between your own submissions:** public-LB differences under about 0.01 between two of your submissions are within noise when ρ_m ≈ 0.8. That is about 2.5× more precise than the 58-study gold set, but still not decisive for the 0.002–0.005 tweaks typical late in a competition.
- **Shake-up between teams:** model correlation across teams is lower (ρ_m perhaps 0.5–0.7), so between-team differences carry more noise than the paired numbers above, approaching the one-model SD of about 0.006–0.010. Medal-zone gaps on Kaggle are often of this size, so shake-ups of several ranks are expected from noise alone.
- **Prevalence shift:** the public/private prevalence differences that the competition states mean per-label AUCs on labels with few positives will be the noisiest. This is the same structure as gold, at roughly 20× the size.
- **Efficiency track:** a private-AUC noise of ±0.009 (90%) is worth about 12 minutes of runtime under the efficiency formula in `docs/competition.md`. Runtime differences of a few minutes between candidate efficient models are therefore within AUC noise.
- **Picking the two final submissions:** they should be chosen on the internal signals (report-label OOF plus gold), not on small public-LB differences, consistent with Roelofs et al.'s finding that public-LB reuse is not the main danger while sampling noise is.

### Gaps
- The actual public/private split fraction for this competition is not in the project docs and was not found. The numbers above assume 1/3 public, and the 1/2 case can be interpolated: SD ∝ 1/√n.
- No published analysis of macro-AUC shake-up on a multi-label medical Kaggle competition was found. The Ladder paper's split-noise numbers are for a different metric and were only seen in summary.
- The test-set label prevalences are unknown. The simulation assumes the gold prevalence mix (8.6–52% positives), and rarer labels would raise the SDs.
