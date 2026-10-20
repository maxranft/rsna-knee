# Cross-model metric regressions ("beta" diagnostics): literature, statistics and pitfalls

Scope: what the dashboard's betas correspond to in the literature, and whether they are statistically sound. The betas are the transfer beta (gold AUC on report AUC per label), the characteristic line (gold macro = alpha + beta x report macro), the model beta (one label's report AUC on the mean of the other 11) and the accuracy/AUC/minutes plane. They are fitted across about 10–25 configs (each config is a mean over 2–30 seeds), with 4,349 report-labelled OOF studies and 58 gold studies.

Project facts used below: in `src/knee/dashboard.py`, `beta()` is cov(y, x)/var(x) and `boot_beta()` is a percentile 90% bootstrap over config points. `MIN_CONFIGS = 8`. Each config point is an unweighted mean over its seeds. The plane uses accuracy at the 0.5 threshold. Gold positives out of 58 (from `docs/data.md`): ACL 24, MCL 9, Medial Meniscus 26, Lateral Meniscus 23, Medial OA 15, Lateral OA 11, PF OA 21, Effusion 35, Synovitis 27, Baker's 12, Contusion 19, Fracture 18.

## Q1. Accuracy-on-the-line literature: transforms, number of models, when the line breaks, slope meaning

### Takeaway
The ML term for "gold = alpha + beta x report" across many models is a **linear trend** ("accuracy on the line") between ID and OOD performance, fitted on a **probit (or logit) scale**. Effective robustness, the residual from that line, is the analogue of Jensen's alpha. These papers rely on 39 to more than 1,000 models spanning tens of accuracy points, R² ≈ 0.94–0.997. The line breaks under subpopulation or spurious-correlation shifts, small correlated test sets (Camelyon17: 50 slides) and mixed pretraining regimes. On the transformed scale, a slope above 1 means "OOD gains exceed ID gains"; below 1, they fall short.

### Cited Findings
- **Recht et al. 2019 (ICML), "Do ImageNet Classifiers Generalize to ImageNet?"** Accuracy drops of 3–15% on CIFAR-10.1 and 11–14% on ImageNetV2. Model accuracies "closely follow a linear function with slope greater than 1 (1.7 for CIFAR-10 and 1.1 for ImageNet)": acc_new = 1.69·acc_orig − 72.7% (CIFAR-10) and acc_new = 1.11·acc_orig − 20.2% (ImageNet). 95% bootstrap CIs (100,000 resamples over models) were [1.63, 1.76] and [1.07, 1.19]. They read slope > 1 as "every percentage point of progress on the original test set translates into more than one percentage point on the new test set … model robustness improves with increasing accuracy" — [arXiv 1902.10811](https://arxiv.org/abs/1902.10811)
- Recht et al. find "the relative order of models is almost exactly preserved", which argues against adaptive overfitting to the reused test set. They attribute the drop to distribution shift: the new sets' lower annotator "selection frequency" made them harder — [arXiv 1902.10811](https://arxiv.org/abs/1902.10811)
- Recht et al. (Appendix C.2) report better fits on a **probit scale**: over ImageNet accuracies from 21% to 83%, Φ⁻¹(acc_new) = u·Φ⁻¹(acc_orig) + v, "The probit scale is in a sense more natural than a linear scale as the accuracy numbers are probabilities." They motivate it with a model in which each example has a scalar difficulty — [arXiv 1902.10811](https://arxiv.org/abs/1902.10811)
- **Taori et al. 2020 (NeurIPS), "Measuring Robustness to Natural Distribution Shifts in Image Classification".** 204 ImageNet models in 213 test conditions. **Effective robustness** is ρ(f) = acc₂(f) − β(acc₁(f)), where β(·) is the baseline linear fit; "a model without special robustness properties falls on the linear fit and hence has ρ(f) = 0." They pair it with **relative robustness**, because an intervention can raise ρ while lowering both accuracies — [arXiv 2007.00644](https://arxiv.org/abs/2007.00644)
- Taori et al. fit the baseline **only on "standard models"** (no robustness intervention, no extra data). Accuracies are "transformed according to the logit distribution; this transform assigns greater mass at the tails and experimentally provided the best linear fits". Their 95% confidence regions come from "1,000 linear fits by sampling the models with replacement", the same model-level bootstrap the dashboard uses, but over about 200 points. Points carry 99.5% Clopper–Pearson intervals — [arXiv 2007.00644](https://arxiv.org/abs/2007.00644)
- **Miller et al. 2021 (ICML), "Accuracy on the Line".** Table 1 gives probit-domain R² and model counts: CIFAR-10.1 0.995 (1,060 models), CIFAR-10.2 0.997 (1,060), CINIC-10 0.991 (949), STL-10 0.995 (456), CIFAR-10-C Fog 0.990 (790), Brightness 0.940 (519), ImageNet-V2 0.996 (219), YCB-Objects 0.975 (39), FMoW-WILDS 0.984 (162), iWildCam-WILDS 0.881 pretrained (66) vs 0.536 from scratch (63) — [arXiv 2107.04649](https://arxiv.org/abs/2107.04649)
- Miller et al., appendix B.3, axis-scaling comparison. Fitted slopes depend on the transform: CIFAR-10.2 slope 0.84 linear (R² 0.98), 0.77 probit (R² 1.00), 0.75 logit (R² 1.00); ImageNetV2 0.95 linear, 0.93 probit, 0.94 logit; YCB 0.68 linear (R² 0.94). **The slope value is transform-dependent, so a slope is only meaningful together with its scale** — [arXiv 2107.04649](https://arxiv.org/abs/2107.04649)
- Miller et al.: the same line holds when varying "model hyperparameters, training duration, and training dataset size" (CIFAR-10.2 slope 0.77, R² 1.00 in each panel). So configs that differ only in hyperparameters, epochs or data size are expected to fall on one line — [arXiv 2107.04649](https://arxiv.org/abs/2107.04649)
- Miller et al., where the line is weak or breaks:
  - Some CIFAR-10-C corruptions; isotropic Gaussian noise gives slope 0.29, R² 0.44.
  - Camelyon17-WILDS. They note it has only 50 slides, with the OOD test set from 10 slides of one hospital, and show that "correlated examples can result in high OOD variation".
  - A different iWildCam split.
  - ImageNet-pretrained models on iWildCam "exhibit a different linear trend than models trained from scratch".

  — [arXiv 2107.04649](https://arxiv.org/abs/2107.04649)
- Miller et al., theory (Theorem 1): for linear classifiers under a Gaussian model with a random shift direction independent of θ, the "probit-transformed accuracies on D and D′ have a near-linear relationship with slope α/γ", with deviation O(d^−1/2). They also describe modifications, e.g. a shift that changes covariance anisotropically, that break the trend — [arXiv 2107.04649](https://arxiv.org/abs/2107.04649)
- **Baek et al. 2022 (NeurIPS), "Agreement-on-the-Line".** Uses ≥150 models per shift, probit-scaled. When ID-vs-OOD accuracy is strongly linear (R² ≥ 0.95), pairwise ID-vs-OOD **agreement** is also linear, "with almost the same slope and bias". When accuracy correlation is weak (R² ≤ 0.75), so is agreement. They found this only for neural networks. ALine-D estimates the line from agreement alone (needs ≥3 models) and predicts OOD accuracy with mean absolute error ≤2% where the phenomenon holds — [arXiv 2206.13089](https://arxiv.org/abs/2206.13089)
- **Teney et al. 2023 (NeurIPS), "ID and OOD Performance Are Sometimes Inversely Correlated on Real-world Datasets".** Inverse ID/OOD correlations occur on Camelyon17 and "on 4 out of 5 other datasets" in the appendix, "across models trained with varying numbers of epochs and random seeds". Past studies missed these "due to a biased selection of models", and "studies … that use ID performance for model selection … will necessarily miss the best-performing models" — [arXiv 2209.00613](https://arxiv.org/abs/2209.00613)
- **Liang et al. 2023 (ICML), "Accuracy on the Curve".** Under subpopulation shift, majority-vs-minority subgroup performance follows a nonlinear, parabolic "moon shape" across architectures, hyperparameters and training durations. Stronger spurious correlation in training makes it more nonlinear — [PMLR v202](https://proceedings.mlr.press/v202/liang23d.html), [arXiv 2305.02995](https://arxiv.org/abs/2305.02995)
- **Kornblith et al. 2019 (CVPR), "Do Better ImageNet Models Transfer Better?"** 16 architectures × 12 datasets. ImageNet accuracy vs transfer accuracy: r = 0.99 for fixed features and 0.96 for fine-tuning, on **log-odds** (logit) accuracy, because "a 1% additive increase in accuracy is different if it is relative to a base accuracy of 50% vs. 99%". Regularizers that improve ImageNet accuracy "are highly detrimental" to transfer of penultimate-layer features. So a config change can move a model off the line in the transfer direction — [arXiv 1805.08974](https://arxiv.org/abs/1805.08974)

### Inferences
- The dashboard's "characteristic line" is accuracy-on-the-line with report AUC as "ID" and gold AUC as "OOD". The per-config residual gold − (alpha + beta·report) is Taori's effective robustness, the analogue of Jensen's alpha. Plotting that residual per config, with the line fitted only on "standard" configs, is the literature-standard diagnostic.
- AUC is itself a probability. Under the equal-variance binormal ROC model AUC = Φ(d′/√2), so probit(AUC) is linear in d′ (a standard binormal-ROC identity; not fetched here). Fitting the lines on probit or logit AUC follows Miller, Recht, Taori and Kornblith and removes ceiling compression for labels near AUC 0.95.
- The literature's lines come from **hundreds of models spanning 10–80 accuracy points**, with R² of 0.94–0.997. The dashboard's configs probably span about 0.005–0.03 report macro AUC (to be checked), so its x-range relative to noise is 1–3 orders of magnitude smaller. Those published R² and slope precisions should not be expected here.
- Three features of this project make a break more likely, on the patterns Miller, Teney and Liang identify:
  - Gold vs report is **two shifts at once**: different labels (report → expert) and a different, enriched subpopulation (58 studies with 1–9 positives each, none with 0).
  - The gold set is small and correlated, like Camelyon17's 50 slides.
  - Configs mix pretraining or encoder regimes, which put iWildCam models on separate lines.
- Interpreting the slope on a transformed scale:
  - **β ≈ 1**: report gains carry over to gold one for one.
  - **β > 1**: gold improves faster than report, as in Recht's harder new test sets. This is also the *expected* direction under pure label noise (see Q2).
  - **β < 1**: part of each report gain does not reach gold. Possible causes are fitting report-labeller artefacts, attenuation from x-noise (Q3), or a ceiling on the raw scale.
  - **β ≤ 0**: the Teney regime, or noise.

### Gaps
- No paper found that applies accuracy-on-the-line to AUC rather than accuracy, or to multi-label medical imaging with report-derived vs expert labels.
- Recht et al.'s exact model counts were not extracted.
- Not fetched: Shi et al. 2023 on effective robustness depending on the ID set, and Wenzel et al. 2022 ("Assaying OOD generalization", 31k networks).

## Q2. Proxy metrics and noisy validation labels: when do noisy-label gains fail to transfer to clean labels?

### Takeaway
If report-label noise is **class-conditional and independent of the image**, AUC on report labels is an exact affine function of clean AUC: AUC_noisy − ½ = (1−α−β)(AUC_clean − ½). Ranking is then preserved and the ideal gold-on-report slope is 1/(1−α−β) ≥ 1. Transfer fails when the noise is **instance-dependent and correlated with model errors**. Examples are reports that omit subtle findings the model also misses, and models that learn the labeller's systematic errors. In that case noisy-label metrics inflate apparent performance, and can prefer the wrong (often higher-capacity) model.

### Cited Findings
- **Menon, van Rooyen, Ong & Williamson 2015 (ICML), Corollary 3.** Under the mutually contaminated distributions (MC) framework of Scott et al. 2013 (which includes class-conditional noise), "for any scorer s: AUC_Dcorr(s) = (1−α−β)·AUC_D(s) + (α+β)/2". Hence "optimising the AUC with respect to the corrupted distribution optimises the AUC with respect to the clean one", and the same holds for balanced error. The cost is a regret penalty of (1−α−β)⁻¹: "for high-noise regimes (i.e. α+β ≈ 1), we need more corrupted samples" — [PMLR v37](https://proceedings.mlr.press/v37/menon15.pdf)
- **Chen et al. 2021 (AAAI), "Robustness of Accuracy Metric and its Inspirations in Learning with Noisy Labels".** For multi-class **class-conditional** noise (transition matrix T), Theorem 1 says the maximum accuracy on the noisy distribution is 1−ε and is attained by the clean-optimal classifier. "The accuracy on hold-out noisy samples is an unbiased estimator of the accuracy on noisy distribution. Therefore, a noisy validation set is reliable", formalised as a validation bound in Theorem 3. They claim this is the first theoretical justification of model selection on noisy validation sets, tested on CIFAR-10/100 with synthetic noise and Clothing1M with real noise — [arXiv 2012.04193](https://arxiv.org/abs/2012.04193)
- **Northcutt, Athalye & Mueller 2021 (NeurIPS Datasets & Benchmarks), "Pervasive Label Errors in Test Sets Destabilize ML Benchmarks".**
  - At least 3.3% label errors on average across 10 test sets, and ≥6% of the ImageNet validation set; 51% of algorithmically flagged candidates were confirmed by humans.
  - "On ImageNet with corrected labels: ResNet-18 outperforms ResNet-50 if the prevalence of originally mislabeled test examples increases by just 6%. On CIFAR-10 … VGG-11 outperforms VGG-19 if … increases by just 5%."
  - "Lower capacity models may be practically more useful than higher capacity models in real-world datasets with high proportions of erroneously labeled data." Their noise estimates are "optimistic in favor of higher capacity models".

  — [arXiv 2103.14749](https://arxiv.org/abs/2103.14749)
- **Imperfect reference standards (diagnostic accuracy).** Valenstein (1990) concluded that when index-test and reference-standard errors are **highly correlated**, "apparent" sensitivity and specificity exceed the true values. With slight correlation they may be over- or underestimated. Under conditional independence they are point-identified. (Valenstein is summarised by the later paper; not retrieved first-hand.) — [arXiv 2204.00180](https://arxiv.org/abs/2204.00180)
- A conference simulation of ROC analysis with an imperfect reference found that error correlations below about 0.3 biased AUC **downward** and only about 0.5 biased it **upward** (abstract only; weak evidence) — [SMDM 2010 abstract](https://smdm.confex.com/smdm/2010on/webprogram/Paper5566.html)

### Inferences
- **Predicted transfer slope under ideal noise.** In the MC parameterisation, the corrupted-positive sample is (1−α)P + αQ, so α = P(truly negative | report-positive) = 1 − PPV of the report label. Likewise β = 1 − NPV, which gives 1−α−β = PPV + NPV − 1 ("markedness"). This is my reading of Menon et al. §2; check before relying on it. If gold equals truth on the same distribution, then gold AUC − ½ = (report AUC − ½)/(PPV+NPV−1), so the per-label transfer beta should be ≥ 1. It should be larger for labels whose report extraction is worse.
  - Gold labels and reports both exist for the 58 gold studies, so PPV and NPV of the report labels can be computed there. This needs aggregate counts only, no report text, and gives a **theory-predicted beta per label** to compare with the fitted one.
  - Caveat: PPV and NPV depend on prevalence, and the gold set is enriched, so values from the 58 studies only approximate those in the 4,349.
- **Two shifts, separated.** Compute report-label AUC and gold-label AUC on the *same* 58 studies. The gap between them is pure label noise. The gap between report AUC on the 58 and report AUC on the 4,349 is pure subpopulation shift. The transfer beta mixes the two.
- **Mechanism for β < 1.** Menon's and Chen's guarantees require noise independent of the image given the true class. Report-derived knee labels plausibly violate this: radiologists under-report subtle grade-1 changes and phrasing varies. If models trained on report labels learn the same blind spots, model and label errors are positively correlated. That is Valenstein's inflation case, and part of each report-AUC gain is then illusory on gold. A falling per-label transfer beta (e.g. < 1 on a probit scale) is the expected signature. Northcutt's capacity result gives the same warning for bigger models and heads.
- Rank-based model selection on report AUC is defensible to the extent the Menon/Chen assumptions approximately hold. Rank agreement between report and gold across configs (Q4) is the most direct check.

### Gaps
- Not found: an empirical study, medical or otherwise, that measures how often model rankings on report/NLP-derived labels disagree with rankings on radiologist labels (e.g. CheXpert labeller vs radiologist test set). This would be the closest analogue and is worth a targeted search.
- Not found: a closed-form result for AUC under instance-dependent noise.
- The Valenstein 1990 original was not retrieved.

## Q3. Errors-in-variables and regression dilution when both axes are noisy

### Takeaway
Noise in **x** (report AUC per config) attenuates the OLS slope by the reliability ratio λ = var(true x)/[var(true x) + var(x-noise)]. A larger sample of configs does not fix it. Noise in **y** (gold AUC) does not bias the slope, but widens its CI. With a narrow config spread, attenuation can be large: λ = 0.5 when the true spread equals the per-config noise SD. The configs' seeds are replicates, so the x-noise variance can be estimated directly, which is the textbook prerequisite for correction (reliability ratio, Deming, SIMEX). A further issue is not covered by simple dilution: seed noise is **correlated between x and y**, because the same models produce both scores.

### Cited Findings
- **Hutcheon, Chiolero & Hanley 2010 (BMJ 340:c2289), "Random measurement error and regression dilution bias".** Random error in the exposure (x) biases slopes toward the null, and larger samples do not fix it. Error in the outcome (y) mainly widens standard errors and CIs (summary points via search; full text returned 403) — [BMJ](https://www.bmj.com/content/340/bmj.c2289)
- Regression dilution is "the biasing of the linear regression slope towards zero" from errors in the independent variable. Noise in y "causes uncertainty in the estimated slope, but not bias". Regression is "not symmetric".
  - Correction for random x: the regression dilution ratio (Frost & Thompson 2000, JRSS A 163:173–190), estimated "usually from repeated measurements on the same individuals". Regression calibration (Carroll et al.) treats the dilution ratio as its simplest case.
  - Correction is "neither necessary nor appropriate" for pure prediction, but is needed for inference about coefficients. A poorly executed correction can do more harm than none.
  - Spearman (1904) disattenuated correlation: r_x′y′ = r_xy / √(r_xx′ · r_yy′).

  — [Wikipedia: Regression dilution](https://en.wikipedia.org/wiki/Regression_dilution)
- Correction for regression dilution is seldom applied in epidemiology. Reliability substudies with repeat measurements are the usual route (Berglund 2012) — [PMC3410287](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3410287/)
- **Deming regression** is maximum likelihood for a straight line when x and y both have independent normal errors with a **known** variance ratio δ. Slope: β̂ = [s_yy − δs_xx + √((s_yy − δs_xx)² + 4δs_xy²)]/(2s_xy). δ = 1 gives orthogonal regression. When δ is estimated from other data, the procedure "does not account for error in that estimate". **York regression** extends Deming to **correlated** x/y errors. Origins: Adcock 1878, Kummell 1879, Koopmans 1936, Deming 1943 — [Wikipedia: Deming regression](https://en.wikipedia.org/wiki/Deming_regression)
- **SIMEX** (Cook & Stefanski 1994, JASA 89:1314–1328) adds extra simulated measurement error at increasing variances, tracks the estimate, and extrapolates back to zero error. It assumes additive, independent, mean-zero normal errors with known or well-estimated variance — [EPA method summary](https://archive.epa.gov/nheerl/arm/web/pdf/m11.pdf); [R `simex` docs](https://search.r-project.org/CRAN/refmans/simex/html/simex.html). With non-normal errors, residual bias after parametric SIMEX "can be substantial" — [arXiv 2111.02863](https://arxiv.org/abs/2111.02863)
- **AUC sampling noise.** Hanley & McNeil (1982, Radiology 143:29–36): SE(A) = √{[A(1−A) + (n₊−1)(Q₁−A²) + (n₋−1)(Q₂−A²)]/(n₊n₋)}, with Q₁ = A/(2−A) and Q₂ = 2A²/(1+A). Software differs between Hanley, Bamber and DeLong variances — [Hanley & McNeil PDF](https://math.ucdavis.edu/~saito/data/roc/hanley-mcneil-auc.pdf); [GraphPad notes](https://www.graphpad.com/guides/prism/latest/statistics/stat_calcualtion_details_for_roc_cu.htm)
- **Sources of variance in ML benchmarks** (Bouthillier et al. 2021, MLSys). "Bootstrapping data stands out as the most important source of variance", while "model initialization generally is less than 50% of the variance of bootstrap". Randomising more sources of variation per trial gives better estimators (51× compute reduction vs the ideal) — [arXiv 2103.03098](https://arxiv.org/abs/2103.03098)

### Inferences
- **Size of the y-noise.** From the Hanley–McNeil formula, the marginal (test-set) SE of one gold label's AUC on 58 studies:
  - MCL (9 positives): ≈0.093 at AUC 0.80, ≈0.071 at 0.90.
  - Lateral OA (11 positives): ≈0.085 / 0.064.
  - Medial OA (15 positives): ≈0.074 / 0.056.
  - ACL (24 positives): ≈0.062 / 0.045.
  - For comparison, report AUC on 4,349 studies at 20% prevalence has SE ≈0.0085.

  Most of the gold test-set error is **shared** across configs, since all are scored on the same 58 studies. That shared part moves the intercept, not the slope. The noise that matters for slopes is the config-specific (config × study interaction) part, which a paired bootstrap or DeLong covariance measures. The dashboard already uses a paired bootstrap for macro differences (line 54). Even so, per-label gold y-noise across configs is plausibly several times larger than the true config spread, so per-label transfer betas will have very wide CIs. That is variance, not bias.
- **Size of the x-noise.** Report AUC has negligible test-set noise, so its x-noise is mostly seed noise: SD_seed²/k for a config with k seeds. Illustrative reliabilities, λ = SD_true²/(SD_true² + SD_noise²), with per-config noise SD 0.003: true config SD 0.010 → λ 0.92; 0.005 → 0.74; 0.003 → 0.50; 0.002 → 0.31. A config set of small tweaks around the best config is exactly the low-λ case. Per-label report AUCs have more seed noise than the macro, so per-label transfer betas and model betas are attenuated more.
- **Estimating the noise variance.** Each config has 2–30 seed replicates, so:
  - x-noise for config i is σ̂²_within,x / k_i, using the pooled within-config seed variance of report AUC.
  - y-noise is σ̂²_within,y / k_i plus the config × study interaction from a paired bootstrap of gold studies.
  - λ̂ = 1 − mean(σ²_within,x/k_i) / var(observed config means). The corrected slope is β̂_OLS/λ̂ (Frost & Thompson dilution ratio). Equivalently, run Deming or York with per-point variances.
- **Correlated errors (beyond Hutcheon).** A lucky seed is better on both report and gold, so the seed-level x/y errors covary (σ_uv > 0). The large-sample OLS slope is then (β·σ²_ξ + σ_uv)/(σ²_ξ + σ²_u), standard EIV algebra (cf. Fuller 1987; York regression). The across-config beta is pulled toward the **within-config (seed-to-seed) slope**, not toward 0. A between/within decomposition separates the two:
  - Fit a bivariate random-effects model with seeds nested in configs.
  - Report the between-config covariance ratio Σ_B[gold, report]/Σ_B[report, report] as the "true" transfer beta.
  - Report the within-config ratio separately.

  This is the method-of-moments version of regression calibration with replicates.
- **Unequal seed counts.** Seed counts range from 2 to 30, so the noise variance differs ~15× across points. Unweighted OLS (current `beta()`) is inefficient and lets 2-seed configs carry as much weight as 30-seed ones. Weighted Deming or York regression, or the mixed model above, handles this.
- **When correction is not needed.** For pure prediction ("given this config's report AUC, what gold AUC do we expect?") the attenuated OLS slope is the right predictor, as noted on the regression dilution page. For the scientific question ("does a real report gain carry to gold one for one?") the corrected slope is needed. The dashboard's "transfer beta" is the second kind.

### Gaps
- Carroll, Ruppert, Stefanski & Crainiceanu (2006) and Fuller (1987) were not accessed directly. The correlated-error bias formula above is standard EIV algebra, not quoted from them.
- Not found: published guidance on how bad λ gets for ML config sweeps specifically.
- Not retrieved: DeLong et al. 1988 (correlated AUCs on one test set).
- Hutcheon full text was blocked (403), so its worked example was not extracted.

## Q4. Statistics with ~10–25 non-independent points: OLS, bootstrap, robust and hierarchical alternatives

### Takeaway
With 8–25 hand-chosen, nested configs, the percentile bootstrap over configs has an ill-defined target, because configs are not an iid sample from a population. It also likely undercovers, and it ignores seed-level and shared-test-set structure. Better options:
- **Rank statistics** (Spearman or Kendall between report and gold across configs), which answer the selection question directly.
- **Theil–Sen** for outlier robustness. It does not fix attenuation.
- A **hierarchical bivariate model with seeds nested in configs**, which gives honest uncertainty and an EIV-corrected slope.
- A **study-level paired bootstrap on gold** for the y-uncertainty.

### Cited Findings
- **Theil–Sen** is the median of pairwise slopes, with breakdown point ≈29.3% (1 − 1/√2); the repeated-median variant reaches 50%. It "competes well against least squares even for normally distributed data" and is more accurate for skewed or heteroscedastic data. A CI can be taken from the middle 95% of pairwise slopes. Primary sources: Theil 1950, Sen 1968. The page does not cover errors in x — [Wikipedia: Theil–Sen](https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator)
- Accuracy-on-the-line papers use bootstrap-over-models CIs, but with hundreds of models: Recht with 100,000 resamples; Taori with 1,000 resamples over 204 models — [arXiv 1902.10811](https://arxiv.org/abs/1902.10811); [arXiv 2007.00644](https://arxiv.org/abs/2007.00644)
- Teney et al.: which models are included determines whether the trend looks positive or inverse ("biased selection of models"). A config set built by greedy improvement around one base config is such a selected sample — [arXiv 2209.00613](https://arxiv.org/abs/2209.00613)
- Miller et al.: ImageNet-pretrained and from-scratch iWildCam models lie on **different lines**, R² 0.881 vs 0.536. Pooling heterogeneous model families into one fit yields a slope for neither — [arXiv 2107.04649](https://arxiv.org/abs/2107.04649)
- Ruan et al. 2024 (NeurIPS) report useful scaling predictions "by evaluating only 10–20 models", but over 77 models from 21 families with wide capability ranges. This is the closest published evidence that about 10–20 points can suffice when the x-range is wide — [arXiv 2405.10938](https://arxiv.org/abs/2405.10938)

### Inferences
- **Slope precision.** Var(β̂) ≈ σ²_resid / [(n−1)·var(x)]. With n ≈ 10–25 and a small var(x), the slope SE will often exceed the slope itself. A 90% percentile bootstrap with n = 8–12 typically undercovers; the ratio cov/var has heavy tails when a resample happens to have a small x-spread. Prefer BCa or a t-based interval with n−2 df, and report n next to every beta. (General bootstrap knowledge; no source fetched.)
- **What the bootstrap treats as independent.** Configs share the 58 gold studies, folds, data and often seeds, and are nested variants of the base config. Resampling configs treats them as exchangeable draws, which they are not. The resulting interval is "how stable is the fit to dropping or duplicating configs", a jackknife-style sensitivity, not a sampling CI. Alternatives:
  - **Two-level bootstrap**: resample gold studies (shared across configs, preserving pairing) × resample seeds within config. This gives the uncertainty from the actual noise sources.
  - **Bivariate mixed model**: (report_ij, gold_ij) for seed j of config i, with config random effects (2×2 Σ_B) and seed residuals (2×2 Σ_W). β_between = Σ_B,12/Σ_B,11, with a profile-likelihood or Bayesian CI. A Bayesian EIV fit (e.g. Stan/brms with known measurement SDs per config) is the same idea with priors. These are reasonable with about 15 configs × several seeds.
- **Rank agreement is the decision-relevant statistic.** The question the betas serve is "does choosing on report AUC pick configs that are better on gold?" That is answered by Kendall's τ or Spearman's ρ between config means. Under H0, SD(τ) ≈ √[2(2n+5)/(9n(n−1))] (standard null variance, not fetched), about 0.19 at n = 15. With the gold noise above, expect τ estimates to be imprecise as well, but they are not distorted by scale or ceiling.
- **Theil–Sen** guards against one or two outlying configs, such as a broken run, but is as attenuated as OLS by x-noise. Use it as a robustness check next to an EIV-corrected estimate, not instead of one.
- **Strata.** Do not pool configs from different regimes (frozen-encoder heads vs fine-tuned, different encoders or pretraining, different label sets) into one fit. Miller's iWildCam result suggests they lie on different lines. The dashboard already restricts to configs on the current best config's labels (`same_labels`), which is the right instinct, and could also stratify by encoder.

### Gaps
- No source fetched on bootstrap coverage of regression slopes at n ≈ 10–25. Standard references (Efron & Tibshirani 1993; Davison & Hinkley 1997) were not accessed.
- Not found: a published ML study fitting hierarchical bivariate models of seed × config × test-set noise for cross-model metric regressions.

## Q5. Is the CAPM / market-beta analogy used in ML evaluation? Standard terminology

### Takeaway
I found no ML evaluation paper that uses CAPM, "beta" or "characteristic line" for cross-model metric regressions. Standard terms are "linear trend / accuracy-on-the-line", "effective robustness" (the residual, i.e. Jensen's alpha), "relative robustness", "transfer" (Kornblith), "observational scaling laws / principal capability measures" (a factor model) and, in psychometric framing, IRT "ability" and "discrimination". The analogy is mathematically apt (single-index factor model, OLS slope), but the finance meaning of beta as risk does not carry over.

### Cited Findings
- A web search for the CAPM-beta analogy in ML model evaluation returned only finance papers comparing CAPM with ML return predictors, plus generic beta/alpha explanations ("beta and alpha are estimated by regressing the asset's excess returns on the market's excess returns using OLS"). No ML-evaluation use was found — [metricgate beta/alpha doc](https://metricgate.com/docs/de/beta-and-alpha/); [arXiv 1511.07101](https://arxiv.org/abs/1511.07101)
- Effective robustness ρ(f) = acc₂(f) − β(acc₁(f)), the deviation from a baseline line fitted on standard models (Taori et al. 2020) — [arXiv 2007.00644](https://arxiv.org/abs/2007.00644)
- **Observational scaling laws** (Ruan, Maddison & Hashimoto 2024, NeurIPS). Over 77 LMs in 21 families, the benchmark-by-model matrix is low-rank: "the top 3 PCs explaining ∼97% of the variance". Each benchmark score is modelled through a few "principal capability measures" with sigmoidal links. This is a multi-factor model across models, the ML counterpart of a market factor model — [arXiv 2405.10938](https://arxiv.org/abs/2405.10938)
- **IRT in ML** (Martínez-Plumed et al. 2019, *Artificial Intelligence*). Instances are items and classifiers are respondents. Each item has difficulty, **discrimination** (the slope of the item characteristic curve vs ability) and guessing parameters, and classifier ability gives "classifier characteristic curves" — [RiuNet record](https://riunet.upv.es/handle/10251/139921). β³-IRT (Chen et al., AISTATS 2019) extends IRT to continuous responses such as predicted probabilities — [PMLR v89](https://proceedings.mlr.press/v89/chen19b.html)

### Inferences
- Mapping, for the write-up:
  - "Characteristic line" → accuracy-on-the-line / linear trend.
  - "Alpha of a config" (residual) → effective robustness.
  - "Transfer beta" → slope of the linear trend; on a logit scale, Kornblith-style transfer.
  - "Model beta" → loading on the first capability factor (Ruan), or IRT discrimination per label.
  - "Plane" → a two-factor model, though one factor (minutes) is a design variable, not a market return.
- The CAPM framing invites over-reading. In finance, beta is estimated from long, roughly iid return series and the market factor is observed without error. Here, x is noisy (Q3), points are few and selected (Q4), and the "market" (mean of the other labels) shares noise with the asset (Q6).

### Gaps
- The search was a single general query; a targeted Semantic Scholar search might find isolated uses of "beta" for metric sensitivities. I found no evidence either way beyond that.

## Q6. Per-label co-movement across models ("model beta"): which labels benefit more from better models?

### Takeaway
The closest literature is **latent-factor and IRT analyses of model-by-task matrices**: Ruan's observational scaling laws, Kornblith's logit-additive model of model and dataset effects, and IRT discrimination. All of them work on **logit or probit scales**, and all find a dominant general factor with item- or task-specific loadings. For this project, "model beta" on raw report AUC conflates three things: a label's true responsiveness to model quality, its report-label noise (Menon's 1−α−β factor shrinks report-AUC movements), and its ceiling.

### Cited Findings
- Kornblith et al. treat model and dataset effects as **additive on log-odds accuracy**. The transformation is needed because "the meaning of a 1% additive increase in accuracy is different if it is relative to a base accuracy of 50% vs. 99%"; an additive logit change Δ is a multiplicative change e^Δ in the odds — [arXiv 1805.08974](https://arxiv.org/abs/1805.08974)
- Ruan et al.: a few capability dimensions explain most variability across benchmarks (top 3 PCs ≈97%). Per-benchmark loadings differ, and the PCs are interpretable, e.g. general, reasoning and programming emphasis — [arXiv 2405.10938](https://arxiv.org/abs/2405.10938)
- IRT discrimination is the slope of an item's response curve against latent ability. Martínez-Plumed et al. relate discrimination and difficulty to instance hardness — [RiuNet record](https://riunet.upv.es/handle/10251/139921); [arXiv 1903.04016](https://arxiv.org/abs/1903.04016)
- Under class-conditional noise, report AUC − ½ = (1−α−β)(true AUC − ½) (Menon et al. 2015 Cor. 3). A label with noisier report labels therefore moves less on report AUC for the same true improvement — [PMLR v37](https://proceedings.mlr.press/v37/menon15.pdf)
- Liang et al.: improvements are not shared evenly across subpopulations; aggregate gains can hide nonlinear subgroup gains — [arXiv 2305.02995](https://arxiv.org/abs/2305.02995)

### Inferences
- **Confounds in the current model beta** (label j's report AUC on the mean of the other 11):
  1. **Label noise.** Labels with worse report extraction get lower betas mechanically, by a factor (1−α_j−β_j)/(mean factor of the others). To compare responsiveness, divide each label's (report AUC − ½) by its estimated (PPV+NPV−1) from the 58 gold studies, or compute model betas on gold, which is noisier.
  2. **Ceiling.** Labels near AUC 0.95 (e.g. effusion) have compressed raw-scale movement. Compute on probit or logit AUC.
  3. **Shared seed noise.** A good seed lifts all labels, so x and y errors are positively correlated, inflating betas toward the within-config co-movement (Q3). The mean of 11 labels has less noise than one label, so x-attenuation is milder here than in the transfer beta, but the correlated-error term remains.
- **Better formulation.** A one-factor model (PCA or factor analysis on the configs × 12 matrix of probit report AUCs), or a 2PL-IRT-style loading per label. With 10–25 configs and 12 labels, a single factor is the most the data can support. Excluding label j from its own regressor, as the code already does, avoids the part-whole correlation.
- **Use for strategy.** Labels with high loading benefit from general model improvements (S3/S4 encoder or head changes). Labels with low loading and decent label quality need label-specific work (S1 rules, S2 auxiliary targets, series-slot choices). Labels with low loading and poor PPV+NPV are label-limited, and S1 is the lever.

### Gaps
- Not found: literature specific to multi-label medical imaging on which findings scale with model quality. The CheXpert/MIMIC per-label literature was not searched within this budget.
- Not found: a per-class "accuracy-on-the-line" analysis.

## Q7. How should the dashboard's slopes be interpreted and computed? (synthesis for this project)

### Takeaway
The betas are a reasonable *descriptive* tool with good literature analogues, but as implemented they combine five problems:
- raw-scale fits;
- unweighted OLS attenuated by seed noise in x;
- correlated x/y seed noise;
- a percentile bootstrap over a few selected configs;
- for the plane, a threshold-dependent outcome regressed on a design variable.

Read them as "direction and rough magnitude", only when the CI excludes the contrast of interest. Gold still decides, and gold differences < 0.02 remain noise.

### Cited Findings
- The literature's practice: probit or logit axes; baseline line fitted on standard models only; residual (effective robustness) as the per-model diagnostic; bootstrap over models with hundreds of points — [Miller 2021](https://arxiv.org/abs/2107.04649); [Taori 2020](https://arxiv.org/abs/2007.00644); [Recht 2019](https://arxiv.org/abs/1902.10811)
- The x-noise correction needs an error-variance estimate, usually from replicates. Deming needs the variance ratio; SIMEX needs the error variance; York handles correlated x/y errors — [Regression dilution](https://en.wikipedia.org/wiki/Regression_dilution); [Deming regression](https://en.wikipedia.org/wiki/Deming_regression); [EPA SIMEX summary](https://archive.epa.gov/nheerl/arm/web/pdf/m11.pdf)
- Ideal-noise theory predicts transfer slope ≥ 1 on the raw AUC scale. A slope < 1 indicates noise correlated with model errors, attenuation, or shift — [Menon 2015](https://proceedings.mlr.press/v37/menon15.pdf); [Valenstein via arXiv 2204.00180](https://arxiv.org/abs/2204.00180)

### Inferences
Concrete changes, in priority order. Most serve S1 and S3: knowing when report-AUC gains are real is what makes the fast feature-cache loop trustworthy.
1. **Add rank agreement across configs**: Kendall τ between report macro and gold macro, with a gold-study bootstrap. It answers "can I select on report AUC?" directly.
2. **Estimate reliability from seeds.** Compute λ̂ = 1 − mean(s²_within/k)/var(config means) for report macro and per label, and show it beside each beta. If λ̂ < ~0.7, label the beta "attenuated" and also show β/λ̂ or a Deming/York fit with per-config variances.
3. **Fit on probit(AUC)** for all betas, following Miller and Recht. Report alongside the raw fit.
4. **Plot residuals.** Show each config's distance from the line (effective robustness), not just the line. Configs well above it are the ones whose report gains carried to gold beyond the trend.
5. **Theory check per label.** Predicted transfer beta = 1/(PPV+NPV−1) of report labels vs gold on the 58 studies (aggregates only). Observed ≪ predicted points to report-labeller bias the models are learning, which is S1 work.
6. **Decompose the two shifts.** Compute report-label AUC on the 58 gold studies vs gold AUC on the same 58 (label noise), and report AUC on the 58 vs on the 4,349 (subpopulation).
7. **Replace or augment the bootstrap.** Use a two-level bootstrap (gold studies × seeds), or the bivariate mixed model for a between-config slope with honest uncertainty. Keep the config bootstrap only as a sensitivity check, and show n.
8. **Plane caveats.**
   - Accuracy at a 0.5 threshold depends on calibration and prevalence, which the project deliberately does not tune (AUC is rank-based), so beta_auc mixes ranking and calibration.
   - Minutes is a chosen design variable that co-varies with what was changed (epochs, encoder), so beta_time is confounded and not a causal "cost of time".
   - With 3 parameters on 8–25 points and likely AUC–time collinearity, both betas will be unstable. Report the variance inflation factor (VIF), or prefer two separate plots: gold vs report, and gold or report vs minutes (a Pareto front for S4).
9. **Stratify** by encoder or pretraining regime before fitting (Miller's iWildCam finding), and note any config added mid-sweep with a different regime.

### Gaps
- The actual spread of config means, seed SDs and λ for this project's runs were not computed here. That needs the run logs, and is a quick aggregate-only computation from `runs`.
- It is not verified whether the gold studies' report labels are already used in training folds in a way that would correlate report and gold errors further.
