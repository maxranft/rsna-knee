# Macro beta holds up; flat epochs mislead

Of the dashboard's diagnostics, only one is both statistically sound and useful for decisions: the **macro transfer beta of about 1.1**. It is the "accuracy-on-the-line" trend from the distribution-shift literature, and every estimator tried gives the same answer, because report AUC is 99% signal across configs and attenuation is negligible. Delta is a 3-epoch secant of the validation curve, and "flat" is a plateau detector, like Keras's `min_delta` with no patience. Under the project's OneCycle schedule both mostly track the learning-rate decay. **Flat fires at epoch 13–16 whatever the budget**, it counts falling curves as flat, and in 28-epoch distilled runs it fires while 0.002–0.005 AUC is still to come. Gamma has no precedent in the literature and is chance-level noise after epoch 12. The accuracy–AUC–minutes plane carries no information. The real statistical weaknesses are elsewhere:

- The transfer intervals hold the 58 gold studies fixed. Resampling them widens the macro interval from [0.92, 1.34] to [0.57, 1.64].
- Configs with only 1–2 seeds swing the per-label betas.
- The 3-seed auto-adopt rule produced a textbook winner's curse. Over 25 seeds, **the d384 "current best" ties d256 on report and loses 0.0054 on gold (interval excludes zero), while taking about twice as long to train**.

For the six days left before 2026-10-22:

- Keep 28 epochs with distillation, and move to d256.
- Choose on replicated report macro AUC, and let gold veto only gaps of 0.02 or more.
- Point any remaining label work at ACL.
- On the dashboard, replace flat, gamma and the plane with peak epoch, end-minus-peak, and an interval that also resamples the gold studies.

## Each diagnostic has a textbook twin, and none of them is a Greek

The finance vocabulary hides fairly standard quantities. In option pricing, delta and gamma are the first and second derivatives with respect to the underlying price. The derivative with respect to time is theta, and the time derivative of delta is charm ([Brilliant](https://brilliant.org/wiki/option-greeks-gamma/); [Macroption](https://www.macroption.com/option-charm)). An AUC gain per epoch is therefore a theta, and "gamma" as defined here has no Greek at all. A search for CAPM-style betas or "characteristic lines" in ML evaluation turned up only finance papers ([cross-model notes](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/cross_model_fits_literature.md)). ML also already uses "delta" for Keras's `min_delta`, "the minimum change in the monitored quantity to qualify as an improvement", which is exactly what the flat threshold is ([Keras](https://keras.io/api/callbacks/early_stopping/)).

| Dashboard name | Established equivalent | Key references | What it measures in this project |
|---|---|---|---|
| **delta** (3-epoch mean of the fold-averaged ΔAUC) | Learning-curve slope; algebraically a 3-epoch secant, (m[t] − m[t−3])/3 | Prechelt's "training progress" ([Prechelt 1998](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf)) | How fast the schedule is still moving the weights, about 1.5 epochs late |
| **flat** (delta < 0.001) | Plateau detector: `min_delta` with no patience and no best-so-far comparison. Not an overfitting rule like Prechelt's GL or UP | [Prechelt](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf); [Keras](https://keras.io/api/callbacks/early_stopping/) | Mostly the point where OneCycle's learning rate has decayed |
| **gamma** (change in delta) | Second difference of the validation curve. No published use as a diagnostic was found; the nearest Greek is charm | [learning-curve notes](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/learning_curve_literature.md) | Noise after about epoch 10 |
| **transfer beta** and the characteristic line | Linear trend ("accuracy on the line"). A config's residual from the line is its "effective robustness", the analogue of Jensen's alpha | [Recht 2019](https://arxiv.org/abs/1902.10811); [Taori 2020](https://arxiv.org/abs/2007.00644); [Miller 2021](https://arxiv.org/abs/2107.04649) | Whether report-AUC gains reach gold |
| **model beta** | Loading on a general capability factor; IRT discrimination | [Ruan 2024](https://arxiv.org/abs/2405.10938); [Martínez-Plumed 2019](https://riunet.upv.es/handle/10251/139921) | Which labels move with general model improvements |
| **plane** (accuracy ~ AUC + minutes) | No standard analogue: a threshold metric regressed on a ranking metric and a design variable | — | Nothing usable (see below) |

The naming matters because each literature comes with known failure modes.

**Learning-curve extrapolation.** These methods fit parametric curves: Domhan et al. ensemble 11 curve families with MCMC, and LC-PFN uses a transformer trained on synthetic curves. They use the fitted curves to *stop clearly bad runs* during hyperparameter search, for speed-ups of about 2× and 2–6× respectively. They are not used to decide when a good run is finished ([Domhan et al. 2015](https://www.ijcai.org/Proceedings/15/Papers/487.pdf); [Adriaensen et al. 2023](https://arxiv.org/abs/2310.20447)).

**Accuracy on the line.** These papers fit their lines to **39–1,060 models spanning tens of accuracy points, with R² of 0.94–0.997**, on probit scales. They also document where the line breaks: subpopulation shift, small correlated test sets such as Camelyon17's 50 slides, and models from different pretraining regimes ([Miller et al. 2021](https://arxiv.org/abs/2107.04649)).

The project's line rests on 35 configs, spread over roughly 0.76–0.80 report AUC and scored against 58 enriched gold studies ([project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md)). That puts it close to the cases where the line breaks. The next two sections test each diagnostic against the 155 finished five-fold runs.

## The flat flag reads the OneCycle schedule, not convergence

**The arithmetic is correct.** Over all 155 runs, averaging the within-fold differences gives the same result as differencing the fold-mean curve (largest discrepancy 2.7e-16). The 3-epoch mean collapses to (m[t] − m[t−3])/3, a secant that lags about 1.5 epochs. For runs still in progress, averaging within-fold differences is the right choice: the obvious alternative would add spurious steps of up to +0.0142 when a partial fold drops out. There is one latent bug. `greeks` drops `None` values before indexing by position, so a `None` in the middle of a fold would shift every later delta of that fold by one epoch. It has never fired ([project learning-curve analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_delta_gamma.md); [dashboard.py](../src/knee/dashboard.py)).

**The problem is the meaning, and the cause is the schedule.** `train_head.py` uses `OneCycleLR` with a 10% warm-up and a cosine anneal down to max_lr/2.5×10⁵, stepped every batch ([train_head.py](../scripts/train_head.py); [PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.optim.lr_scheduler.OneCycleLR.html)). With a schedule tied to the run length, "the training duration must be known in advance and match the cosine length", and intermediate points "underestimate" what a run of that length would reach ([Hägele et al. 2024](https://arxiv.org/html/2405.18392v3)). The per-epoch gain is therefore forced towards zero at the end of every run. The project's runs show exactly this. The flat epoch is about 10 at a 12-epoch budget, 13.0 at 16, 13.7 at 20 and 15.8 at 28, so flat/E falls from 0.83 to 0.56 ([project analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_delta_gamma.md)).

- **16-epoch runs:** flat arrives when the learning rate is at about 10% of its maximum, so it marks the anneal and nothing more.
- **28-epoch distilled runs:** flat fires while the learning rate is still at **40–60% of maximum**. Every d256 seed then gains another **+0.0022 to +0.0053** (mean +0.0036), which is two to four times the report seed SD.

**A flat epoch does not mean the budget is big enough.**

**The noise is real, and the threshold is not tied to it.** Per-fold epoch-to-epoch jitter is 0.004–0.006 early in a run, 0.0013–0.0023 in the middle and 0.0003–0.0008 at the end. Each fold's validation set is fixed, so this is optimisation noise, and it shrinks as the learning rate falls. That is a second reason every annealed run eventually "goes flat". After averaging over folds, the seed SD of delta at the flat epoch is 0.0003–0.0004, so the 0.001 threshold is detectable. The *epoch* at which it is crossed is still uncertain by 0.7–2.1 epochs. The 25 d384 seeds cross the threshold 3.2 times per run on average, and their flat epoch ranges from 12 to 21 ([project analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_delta_gamma.md)).

The established criteria handle this in two ways: they compare against the *best-so-far* score, and they use patience. Prechelt's GL measures the percentage rise of validation error over its minimum, and UP stops after validation error rises over several successive strips of epochs. Both return the best weights. Flat is closest to Prechelt's training-progress measure, which he used only as a termination backstop and only on *training* error. Even good criteria found the best solution of a run only 60–80% of the time, and slower criteria bought about 4% lower error for about 4× the training time ([Prechelt 1998](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf)). A fixed per-epoch threshold also ignores how much time is left: 0.0009 per epoch over 12 remaining epochs adds up to about 0.01.

**Flat calls declines flat, and one dashboard claim is false.** The test is `v >= FLAT`, so negative deltas pass as flat.

- The three 24-epoch runs print "flat from epoch 12/11/12 of 24". Their fold-mean AUC actually peaked at epochs 11/9/11 and then *fell* by 0.0091–0.0118.
- The template's note says "every run ends flat". That is false for the three EMA-0.999 runs, whose final delta is +0.0013 to +0.0015 because the EMA weights lag, and for the 8-epoch run ([template.html](../scripts/dashboard_ui/template.html); [project analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_delta_gamma.md)).

**Gamma is noise.** At steady state, gamma is (a_t − a_{t−1} − a_{t−3} + a_{t−4})/3, a second difference spanning four epochs. Differencing amplifies white noise: the variance is 2σ² for first differences and 6σ² for second differences ([learning-curve notes](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/learning_curve_literature.md)). The terminal display shows gamma per fold, and its SD there is about 0.0010, the same size as the flat threshold itself. From epoch 12 onwards, **only 52–68% of d384 seeds agree on the sign of gamma**, which is chance level. Gamma is reliably negative only in epochs 4–8, where any saturating curve slows down ([project analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_delta_gamma.md)). No ML paper was found that uses the curvature of a validation curve as a stopping signal. Where curvature is wanted, it is read off a fitted smooth curve ([Domhan et al. 2015](https://www.ijcai.org/Proceedings/15/Papers/487.pdf)).

**Simpler summaries do carry signal.** End − peak is how far a curve has fallen from its own peak by the last epoch. Across 24 configs it correlates with both report AUC (ρ = +0.63) and gold AUC (+0.59). Flat/E correlates at −0.02 and −0.05. Within a config, across seeds, the flat epoch predicts neither metric (all p ≥ 0.09).

In the adopted configs, the curve peaks at 87–92% of the budget and ends within 0.0001–0.0002 of its peak. **Saving the last epoch is therefore correct.** Restoring each fold's best epoch would gain nothing and would bias the OOF score upwards, the selection effect that Cawley and Talbot show can be as large as real differences between methods ([JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)).

Validation loss is not a usable stopping signal either:

- It bottoms out at 0.36–0.66 of the budget, and AUC peaks later in 94% of runs.
- Stopping at the loss minimum would cost 0.003–0.004 AUC in the distilled configs.
- This is the usual pattern of growing overconfidence while ranking keeps improving, and the metric only cares about ranking.

The literature's recommended way to choose an epoch budget is also the one the project already follows: run the full schedule at several budgets and compare the end-of-schedule scores ([Hägele et al. 2024](https://arxiv.org/html/2405.18392v3)).

## Macro transfer of 1.1 survives every estimator; most per-label betas do not

The dashboard fits one point per config: the unweighted mean of its seeds. Its intervals come from a percentile bootstrap that resamples configs 2,000 times. `MIN_CONFIGS = 8` applies, but **no minimum seed count does**. Today's fit is gold = −0.136 + 1.105 × report, with 90% interval [0.923, 1.335] and r = 0.87 over 35 configs ([project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md); [dashboard.py](../src/knee/dashboard.py)).

**The two pitfalls the literature warns about most do not apply here.**

*Attenuation.* Noise in x biases an OLS slope towards zero, and adding more points does not fix it ([Regression dilution](https://en.wikipedia.org/wiki/Regression_dilution); [Hutcheon et al., BMJ 2010](https://www.bmj.com/content/340/bmj.c2289)). Here the between-config SD of report macro AUC is 0.0091, six times the pooled seed SD of 0.0015. That gives a **reliability ratio λ_x of 0.99** for macro and 0.87–0.97 per label. Every estimator lands in the same place:

| Estimator | Macro transfer beta |
|---|---|
| OLS (dashboard) | 1.10 |
| Moment-corrected | 1.11 |
| Deming | 1.12 |
| Split-half IV | 1.10–1.26 |
| Theil–Sen | 1.13–1.20 |
| Weighted least squares | 1.16 [1.04, 1.32] |

Deming collapses onto OLS because gold noise is about 32 times report noise ([Deming regression](https://en.wikipedia.org/wiki/Deming_regression)).

*Correlated seed errors.* Shared seeds could correlate the report and gold errors. In practice, the within-config correlation between a seed's report and gold deviations is +0.09. The seeds 42/43/44, shared by 24 configs, induce no detectable common noise: the mean pairwise correlation of seed deviations is −0.03 (p = 0.73 for report, 0.81 for gold) ([project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md)).

**The macro point estimate can be trusted.**

**The intervals answer the wrong question.** Resampling configs over a fixed set of 58 studies asks: "would another set of configs give the same slope on these studies?" The decision needs "will report gains show up on unseen cases?" A two-level bootstrap resamples gold studies as well as configs. It widens the macro interval to **[0.57, 1.64]**, and per-label intervals by 1.5–2.5×.

The configs are also a project trajectory, not a sample. Early configs differ from the base in four to six settings at once, and all were selected on report AUC. Teney et al. show that this kind of selection of models can hide or even reverse ID/OOD trends ([Teney et al. 2023](https://arxiv.org/abs/2209.00613)). Miller et al. show that different regimes lie on different lines ([Miller et al. 2021](https://arxiv.org/abs/2107.04649)). The slope therefore summarises "the changes kept so far moved report and gold together". It says nothing about a new kind of change, such as a new labeller.

**Per-label betas are fragile, and the template reads them backwards.** Per label, the gold seed SD is 0.018–0.039 and the report seed SD only 0.002–0.006. Most of the uncertainty is in gold.

- **Which labels clear zero:** under the two-level bootstrap only **Effusion [1.09, 4.63] and Fracture [0.70, 2.99]** clear zero across all configs. Medial OA clears it only within the configs with at least 3 seeds, and MCL only among the top configs.
- **What the 1–2-seed configs do:** eleven such configs date from 2026-09-29/30 and sit at the low end of report AUC, where they have the most leverage. Dropping them moves the per-label betas a lot but barely moves macro (1.10 → 1.09):

| Label | Beta, all configs | Beta, configs with ≥3 seeds |
|---|---|---|
| Lateral Meniscus | 1.49 | 0.75 |
| Medial Meniscus | −0.75 | +0.49 |
| Synovitis | −1.05 | −0.40 |
| PF OA | 1.39 | 0.65 |

For Lateral OA and Synovitis, gold's reliability across configs is about zero: no change tried so far has moved gold detectably for either label.

The template says that β near 1 means "real" transfer and that an interval reaching 0 means "the rule needs work". Theory says otherwise. Menon et al. assume class-conditional label noise: the chance that a report label is wrong depends only on the true label, not on the image. Under that assumption, AUC on corrupted labels equals (1−α−β)·clean AUC + (α+β)/2, where α and β are the two noise rates ([Menon et al. 2015](https://proceedings.mlr.press/v37/menon15.pdf)). Optimising report AUC therefore optimises clean AUC, and Δgold = Δreport/(1−α−β), which is **at least Δreport**. A noisy validation set is likewise reliable for model selection under this kind of noise ([Chen et al. 2021](https://arxiv.org/abs/2012.04193)). Effusion's β of 2.75, against about 2.3 predicted from its labeller errors, is what full transfer looks like. An interval through zero mostly means there are too few gold positives to tell.

The real failure mode is label noise that is correlated with model errors. When the reference standard's errors are correlated with the model's, apparent accuracy is inflated ([Valenstein, via arXiv 2204.00180](https://arxiv.org/abs/2204.00180)). After correcting test-set label errors, smaller models can overtake bigger ones ([Northcutt et al. 2021](https://arxiv.org/abs/2103.14749)). The project shows this signature: **seven labels have gold AUC below report AUC**, which pure label noise cannot produce. The gold set is also enriched (1–9 positives per study, none with zero), so gold differs from the report set in two ways at once: in its labels and in its population ([project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md)).

**Per label: transfer evidence and headroom.** Gold AUC is the mean of the 25 d384 base seeds, with a 90% gold-study bootstrap interval, before the Synovitis mix. Sign agreement covers config pairs whose report AUC differs by more than 0.01; it is given as observed against the rate expected if β = 1 with gold seed noise only.

| Label | Gold positives | Base gold AUC [90%] | Labeller–gold AUC | Transfer (two-level) | Sign agreement, observed vs expected | Model beta |
|---|---|---|---|---|---|---|
| Medial Meniscus | 26 | 0.592 [0.474, 0.701] | 0.900 | ≈ 0 | 0.70 vs 0.76 | 0.89–1.06 |
| MCL | 9 | 0.654 [0.483, 0.803] | 0.927 | > 0 only in top configs | 0.72 vs 0.73 | 1.00–1.05 |
| Synovitis | 27 | 0.521 [0.408, 0.639] | 0.714 | ≤ 0 | **0.39** vs 0.76 | 0.89–0.98 |
| Lateral Meniscus | 23 | 0.692 [0.587, 0.793] | 0.865 | ≈ 0 | 0.62 vs 0.73 | 0.90–0.96 |
| ACL | 24 | 0.718 [0.610, 0.822] | 0.862 | ≈ 0 | 0.62 vs 0.85 | **1.16–1.30** |
| PF OA | 21 | 0.738 [0.629, 0.840] | 0.870 | ≈ 0 | 0.63 vs 0.87 | 0.98–1.05 |
| Fracture | 18 | 0.778 [0.658, 0.884] | 0.906 | **> 0** | 0.83 vs 0.83 | 1.20–1.34 |
| Lateral OA | 11 | 0.700 [0.553, 0.839] | 0.795 | ≈ 0 | 0.50 vs 0.77 | 0.85–0.92 |
| Baker's | 12 | 0.860 [0.767, 0.937] | 0.889 | ≈ 0 | 0.69 vs 0.86 | 0.74–0.83 |
| Medial OA | 15 | 0.943 [0.897, 0.980] | 0.949 | **> 0** (≥3-seed configs) | 0.91 vs 0.88 | 0.73–0.77 |
| Contusion | 19 | 0.856 [0.760, 0.935] | 0.835 | ≈ 0 | 0.60 vs 0.86 | 1.16–1.34 |
| Effusion | 35 | 0.904 [0.844, 0.954] | 0.820 | **> 0** | 0.94 vs 0.81 | 0.65–0.74 |

Source: [project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md).

**Model beta is the stable diagnostic.** It is measured entirely on the 4,349 report studies, so its 90% intervals are only about ±0.1–0.2 wide.

- **ACL, Contusion and Fracture** respond more than average to whatever improved the model (β of 1.2–1.3).
- **Effusion and Medial OA** respond less (0.65–0.77), because they are near their ceiling.

Its literature analogue is a loading on a general capability factor: across 77 language models, three principal components explain about 97% of benchmark variance ([Ruan et al. 2024](https://arxiv.org/abs/2405.10938)). That work, like Kornblith's transfer study, uses logit or probit scales, because a 1-point gain means something different at 50% than at 99% ([Kornblith et al. 2019](https://arxiv.org/abs/1805.08974)). ACL shows how to read the two betas together: a high model beta with zero transfer means the model is getting better at the *labeller's* ACL, not gold's.

**The plane is empty.** The fitted plane is accuracy = 0.541 + 0.360·AUC − 3.9×10⁻⁶·minutes.

- **The accuracy axis is just AUC.** Its R² of 0.917 is the same as with AUC alone. Config accuracy at the 0.5 threshold (0.8167–0.8295) is only slightly above the 0.792 an all-negative classifier scores. For MCL, Lateral Meniscus and Fracture, the model's accuracy is at or below all-negative.
- **The time axis is machine load.** β_time is −0.0002 accuracy per hour (t = −0.15). Training minutes range from 28.1 to 64.9 across the 25 seeds of a *single* config, so they mostly measure what else the Mac was doing.
- **Neither axis is what the competition scores.** Training time is not scored. The efficiency prize scores Kaggle runtime at about **0.044 AUC per hour**, and head inference is about 0.1 minutes of a 17-minute pipeline ([project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md); [competition.md](../docs/competition.md)).

## Gold can veto a 0.02 gap but cannot rank the top cluster

**How noisy each yardstick is.** The pooled seed SD is **0.0015 for report macro and 0.0085 for gold macro**. The project's working figures of "about 0.004 / 0.010" were three-seed ranges: they overstated report noise and understated gold noise. On top of seed noise, gold carries study-sampling noise. The base config's gold macro of 0.746 has a 90% gold-study interval of [0.714, 0.778] ([project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md)). Synthetic simulations of a 58-study, 12-label set give these SDs ([small-sample notes](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/small_sample_auc_statistics.md)):

- **Macro AUC of one model:** 0.022–0.045, depending on how correlated the labels are.
- **Paired difference between two models:** 0.010–0.031, depending mostly on how correlated the two models' scores are.
- **A single label with 8 positives:** 0.058–0.108. Standard 90% intervals at 5–8 positives and AUC 0.8–0.9 actually cover only 73–85% of the time.

The "weakest 3 labels" logged in experiments.md are therefore mostly noise.

**The 0.02 rule is a floor, not a decision threshold.** In the same simulation, the smallest gold difference detectable with 80% power is **0.034–0.047** for closely related models. A true +0.02 gain reaches P(better) > 0.95 only 28–44% of the time, and shows the wrong sign 5–14% of the time. The same gain measured on the report set has a z-score about eight times larger.

The project's own config pairs show the same thing:

| Report macro gap between two configs | Share of pairs where gold moves the same way |
|---|---|
| 0.005–0.01 | 90% |
| More than 0.01 | 99–100% |
| Below 0.005 | 60–62% |

The 60–62% for small gaps is about what perfect transfer combined with gold's own noise would produce, so **a gold "disagreement" on a small report gap is not evidence** ([project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md)). The ground rule already contains the right floor ("gold differences below about 0.02 are noise"). Applied literally, a gold difference under 0.02 is not a disagreement, so report decides.

Report AUC needs two guards of its own:

- **Labeller blind spots.** The models are trained on the same labeller they are scored against, so they can learn its blind spots. That is a bias no bootstrap can show. The cheap check is to see which labels a report gain comes from. A gain concentrated in Synovitis, ACL, Contusion or Lateral OA is less likely to reach gold than one spread across labels.
- **Changing labels changes the ruler.** A change to the labels themselves moves the report yardstick. Such changes have to be judged on gold, with a rule fixed before scoring.

**Three seeds cannot carry a decision.**

- **Welch's test:** with 3 seeds against 3, the median 90% interval is ±1.76 seed SDs, and the power to detect a 1-SD difference is 22%. About 17 seeds per arm are needed for 80% power at 1 SD.
- **Cliff's delta:** on 3×3 seeds it can take only 10 values, and the extreme value ±1 has a two-sided p of 0.10.
- **The bootstrap's P(better):** when the true difference is zero, it still exceeds 0.75 in 20–30% of datasets ([small-sample notes](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/small_sample_auc_statistics.md)).

These match the multi-seed literature: Bouthillier et al. find data sampling to be the largest source of variance ([2021](https://arxiv.org/abs/2103.03098)), and Agarwal et al. find that bootstrap intervals over 3 runs under-cover ([2021](https://arxiv.org/abs/2108.13264)). With a report seed SD of 0.0015, three seeds a side resolve differences of about 0.003.

**The winner's curse has already struck twice.** In simulation, the best of 10 truly equal configs beats the reference on gold by +0.014 on average, and by more than 0.02 in 27% of cases. When real effects are small, the winner's observed gain is about four times its true gain ([small-sample notes](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/small_sample_auc_statistics.md); [Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html)). The project has two cases:

| Change | Auto-rule verdict (3 seeds) | Later evidence |
|---|---|---|
| d384 (vs d256) | Adopted: report +0.0020 (CI +0.0008..+0.0034); gold −0.0164 (P 0.03), let through by "gold no worse than −0.02" | 25 vs 11 seeds: report −0.0006; **gold −0.0054 (CI −0.0098..−0.0009)** |
| 20 epochs without distillation (vs 16) | Adopted: report +0.0019, CI lower bound exactly +0.0000 | Single models tie over 9 vs 23 seeds (report −0.0002) |

Sources: [experiments.md](../experiments.md); [project analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_delta_gamma.md). The auto rule's report clause is too sensitive to three-seed luck, and its gold clause is too lax to veto anything.

**The public leaderboard is no sharper.** The simulation assumes a public share of about one third; the real split is unknown. At about 1,300 test studies:

- A single submission's macro AUC has an SD of about 0.006, and the public part alone about 0.010.
- One model's public and private scores can differ by ±0.020 (90%) from sampling alone.
- Two of your own submissions that truly differ by 0.005 swap order on the public board 16–23% of the time.

Across more than 100 Kaggle competitions, sampling noise rather than adaptive overfitting dominated the gap between public and private scores ([Roelofs et al. 2019](https://papers.nips.cc/paper/9117-a-meta-analysis-of-overfitting-in-machine-learning)). Differences on the public board of under about 0.01 between your own submissions cannot settle late tweaks.

## Six days left: d256 at 28 epochs, ACL labels, a quieter dashboard

The current picks are:

- **Accurate:** `mix-20261006-114310`, with 160 heads, TTA and the Synovitis mix (report 0.8031, gold 0.771).
- **Efficiency:** `mix-20261006-114316`, with nine distilled 20-epoch d256 seeds, i.e. 45 heads and no TTA (report 0.8034, gold 0.7652).

The ensemble is saturated: using 73 runs instead of 23 added only +0.0006 report for three times the heads ([experiments.md](../experiments.md)). Only one training run fits on the Mac at a time. A d256 28-epoch run takes about 22 minutes, and a d384 run 28–65 minutes. Evidence for the table: [project analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_delta_gamma.md) and [project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md).

| Decision | Recommendation | Evidence | Bet |
|---|---|---|---|
| Epoch budget | **Keep 28 epochs with distillation.** Do not shorten on the strength of "flat", and do not stop at epoch 21 (that loses about 0.0006 and leaves the weights unannealed) | 20 → 28 epochs at d256: report **+0.0027 [+0.0019, +0.0035]**, gold **+0.0055 [+0.0016, +0.0095]**, for +38% training time | S3 |
| Beyond 28 epochs | Skip in the final week | Untested, with diminishing returns. A 3-seed check against 11 seeds has a 90% interval of roughly ±0.002 (derived from the 0.0015 seed SD), wider than any plausible gain | S3 |
| Config | **Use d256, not d384.** If auto-improve keeps running, re-base it on d256 at 28 epochs | d384 − d256: report −0.0006 [−0.0013, +0.0001], gold **−0.0054 [−0.0098, −0.0009]**, about 2× training time | S3, S4 |
| Efficiency submission | Build an OOF ensemble from the 11 existing d256 28-epoch seeds and compare it with the current 9 d256 20-epoch seeds. Swap if paired report AUC improves and gold is not worse by 0.02 or more. No training is needed, and inference cost is identical (same head) | 3-run ensembles: report +0.0016 [+0.0002, +0.0029]; heads take about 0.1 of the 17 pipeline minutes | S4 |
| Accurate submission | Run the same OOF check, adding the d256 28-epoch seeds or swapping out weaker members. Expect at most about +0.001 | Saturation: +0.0003 to +0.0006 for many more heads | S3 |
| Selection rule | Rank by report-macro seed means (at least 3 seeds). Treat gaps under 0.003 as ties. Break ties on gold only at a gap of 0.02 or more, otherwise on cost. Check which labels a gain comes from | Pair agreement; macro transfer ≈ 1.1 | S3 |
| Auto-improve | Stop auto-adopting on 3 seeds. Replicate any winner with fresh seeds before it enters a submission, and log how many candidates were tried per adoption | The d384 and 20-epoch cases | S3 |
| Final picks | Do not choose between your own submissions on public-board differences under about 0.01 | Leaderboard noise simulation | S3, S4 |

**Which labels to invest in.** The transfer table splits the labels into three groups.

*Label-limited (S1).* For these labels, model work is wasted until the labels match gold's definitions.

- **Synovitis** anti-transfers (sign agreement 0.39) and has the weakest labeller (0.714). Its main lever has already been pulled: the pre-registered Synovitis := rank-mean(Synovitis, Effusion) mix lifted the 23-seed ensemble's gold Synovitis from 0.529 to 0.664 ([experiments.md](../experiments.md)). That leaves only about 0.05 to the labeller's level, worth roughly 0.004 macro. The post-hoc variants that scored higher on gold were deliberately not adopted, and the public board cannot resolve differences that small.
- **ACL** is now the clearest label-side problem. It has the highest model beta, zero measurable transfer, and a gap of 0.14 between gold AUC (0.718) and labeller AUC (0.862). The ACL@abnormal labeller variant raised labeller–gold AUC by +0.031 and was not adopted ([project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md)). In six days it is the only realistic S1 move, and only if its scores already exist for all report studies. Test it with one 3-seed d256 28-epoch run (about 66 minutes), judged on gold with a rule fixed in advance, as was done for Synovitis. Be clear-eyed about the odds: ACL's gold AUC has an SD of about 0.06, so only a large change will be visible. Treat it as a bet, not a verified gain.
- **Lateral OA** belongs here too, but the labeller log already records that rewording against the 58 studies has run its course for it, along with Synovitis and Contusion.

*Image-limited (S3, after the deadline).* **Medial Meniscus** has a high labeller ceiling (0.90), a gold AUC of 0.59 and no transfer. That points to missing image signal, not label noise, and that is not final-week work. **MCL** has the second-largest gap but only 9 gold positives.

*Leave alone.* **Medial OA, Effusion, Contusion and Baker's** are at or above their labeller ceilings. No S2 auxiliary-target work is supported by this evidence in the time left.

**Dashboard changes (S3, with one for S4), in priority order.** The first five are small code edits.

1. Replace "flat from epoch X" with **peak epoch (and peak/E), end − peak, and the gain still to come compared with the seed SD**. If a plateau flag stays, rename it "plateau", flag declines separately, and require two to three epochs below the threshold before it fires. Delete the false "every run ends flat" sentence.
2. Remove gamma, or show it only with a seed band.
3. Apply `MIN_SEEDS = 3` to the betas and the plane, or weight configs by their noise.
4. Reword the transfer note: real transfer through noisy labels shows up as β ≳ 1, and an interval through zero means "too few gold positives to tell".
5. Re-point "Current reference" from `20260929-140030` (report 0.7797, gold 0.7242) to the current best config.
6. **(S4)** Replace the plane with AUC against *measured* Kaggle inference runtime. The dashboard flags its efficiency numbers as unmeasured.

After the deadline:

- Add the two-level bootstrap. Cached, it ran well within a 10-minute CPU call ([project beta analysis](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/project_betas.md)).
- Add Kendall's τ between report and gold rankings, and a residual ("effective robustness") plot ([Taori et al. 2020](https://arxiv.org/abs/2007.00644)).
- Fit the lines on a probit scale.
- Plot learning curves against the fraction of the schedule completed.
- Fix the `None`-indexing bug.
- Treat the individuals control chart as a broken-run monitor only. With 5–20 seeds, its false-alarm rate is 1.1–5.7% per point instead of 0.27% ([small-sample notes](../research_notes/Delta%20gamma%20and%20beta%20diagnostics/small_sample_auc_statistics.md)).
- If readable learning curves matter again, try a warm-up / constant / short-cooldown schedule or Schedule-Free AdamW. Either keeps the slope meaningful during training ([Hägele et al. 2024](https://arxiv.org/html/2405.18392v3); [Defazio et al. 2024](https://arxiv.org/abs/2405.15682)).

## Conclusion

The most elaborate diagnostics carry the least information. Gamma, flat and the plane mostly restate the learning-rate schedule, seed noise or machine load. The plainest number, macro AUC on 4,349 report-labelled studies replicated over three or more seeds, resolves differences about eight times finer than the 58 gold studies, and its gains reach gold at a measured slope of about 1.1. The project's binding constraint is resolution, not compute. Under-training did not cost anything this month; selection noise did. The auto-improve loop promoted a slower config that is worse on gold, then spent most of the following week of Mac time exploring around it. In the final week the advantage lies in discipline: freeze the selection rule, replicate before adopting, and prefer the cheaper of two tied configs.

Two further implications are less obvious. First, gold should act as a veto for large gaps, not as a ranking tool, *except* for label changes, where gold is the only fair yardstick. Late S1 work is therefore the hardest kind to verify, which argues for testing only pre-registered, already-built variants such as ACL@abnormal. Second, the efficiency track is decided by preprocessing and encoder runtime, not by heads or epochs. Epoch budget and head width are purely training-time (S3) choices, so the S4 submission should simply take the most accurate heads that run in the same pipeline. The swap to d256 at 28 epochs is the one change that is both measurable and free at inference. Several things remain unmeasured: per-epoch gold curves, how well transfer holds for rank ensembles, the test set's label mix, and the public/private split. Any of these could shift the margins, but none argues for a different direction.

*This report uses only aggregate metrics from the project's run logs and published literature. It contains no report text and no images.*
