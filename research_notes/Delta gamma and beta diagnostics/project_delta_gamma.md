# How the project's own learning curves behave on delta, gamma and "flat", and whether the diagnostics are sound

Data: the 155 finished 5-fold single-model runs in `runs/` that have a `history.jsonl` (2026-09-29 to 2026-10-15), plus 2 aborted runs that were excluded. Inputs read were metrics only: `args.json`, `metrics.json`, `history.jsonl`, `experiments.md` and source code. No report text, images or `external scan/` were opened. Unless a row says otherwise, "AUC" in the learning curves means the per-epoch validation macro AUC that `history.jsonl` records. That is, for each fold, the mean over the 12 labels of the AUC on that fold's report-only validation studies (about 870) against report labels thresholded at 0.5, then averaged over the 5 folds. **It is never gold.** "report" and "gold" in the outcome columns are the pooled OOF macro AUCs from `metrics.json`.

## Q1. How delta, gamma and flat are computed, and whether there are bugs or statistical problems

### Takeaway
The arithmetic is correct. For finished runs, averaging the within-fold differences is identical to differencing the fold-mean curve (largest discrepancy 2.7e-16 over 155 runs). The "3-epoch mean of first differences" telescopes to a 3-epoch secant, (m[t] − m[t−3])/3. For runs still in progress, averaging within-fold differences is the right choice. The problems are in what the numbers mean and how noisy they are:
- "flat" counts a falling curve as flat.
- It has no hysteresis.
- Its 0.001 threshold is not tied to any noise level that matters for a decision.
- gamma is a 4-epoch second difference and is mostly noise after epoch 10.
- A latent None-handling bug exists but has never fired.
- One claim on the dashboard ("every run ends flat") is false for 4 runs.

### Method (code run)
Scripts are in the scratchpad (`lc_load.py`, `lc_analyse.py`, `lc_part2.py`, `lc_part3.py`, `lc_part4.py`) and were run with `.venv/bin/python`. `greeks`, `compact_history` and `train_minutes` were **copied verbatim** from `src/knee/dashboard.py`. In outline:
```python
# per run: M = folds x epochs AUC matrix (history.jsonl), L = same for val_loss; m = M.mean(0)
peak = argmax(m)+1;  end_minus_peak = m[-1]-m.max();  gain_lastQ = m[-1]-m[E-round(E/4)-1]
g = greeks(compact_history(hist))            # verbatim copy; flat, delta[-1], gamma[-1]
sigma_fold = std(diff(M[f, phase], 2), ddof=1)/sqrt(6)   # white-noise sigma from 2nd differences
# identities checked:
assert max|diff(M,axis=1).mean(0) - diff(m)| < 1e-12
assert delta[i] == (m[i+1]-m[max(0,i-2)])/(min(i,2)+1)   # tol 2e-5 (5-dp rounding)
# config = args minus (seed, log, no_open, workers), defaults filled as dashboard.OFF
# budget comparisons: Welch 90% interval and Cliff's delta on seed-level metrics.json values
```

### Cited Findings
- `FLAT = 0.001` is defined as the "validation macro AUC per epoch: a learning curve gaining less than this counts as flat". — [dashboard.py L33](file:///Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py)
- `greeks(hist, k=3)` works in five steps. — [dashboard.py L111-127](file:///Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py)
  1. It drops `None` values from each fold's list.
  2. It computes `d[t] = mean over folds with len(f) > t of f[t] − f[t−1]`.
  3. It sets `delta[i] = mean(d[max(0,i−2) : i+1])`, rounded to 5 dp, and `gamma` to the successive differences of `delta`.
  4. It sets `over = {i : delta[i] ≥ FLAT}`. `flat` is None if the last delta is ≥ FLAT, otherwise `over[-1] + 3`, or 2 if `over` is empty.
  5. `now` is the same calculation on the newest fold alone.
- `delta[i]` is the gain going into epoch i+2, with epochs numbered from 1. `path3d` maps epoch e to `delta[e−2]` and `gamma[e−3]`, which is consistent. — [dashboard.py L124, L130-145](file:///Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py)
- The terminal delta and gamma come from `ProgressTracker._greeks`. They are **per fold**, not averaged over folds, with the same trailing 3-window: `gamma = mean(d[-3:]) − mean(d[-4:-1])`. For a single fold this matches the dashboard. — [progress.py L80-86](file:///Users/maxranft/Projects/rsna-knee/src/knee/progress.py)
- Validation AUC per epoch uses report-only studies against thresholded report labels. Validation loss is BCE against the *training targets* on **all** validation studies, gold included. The checkpoint saved, and so the OOF, is the **last epoch**: there is no early stopping. — [train_head.py L57-69, L193-199](file:///Users/maxranft/Projects/rsna-knee/scripts/train_head.py)
- The schedule is `OneCycleLR(opt, lr, total_steps=epochs*len(dl_tr), pct_start=0.1)`, stepped every batch. — [train_head.py L175-188](file:///Users/maxranft/Projects/rsna-knee/scripts/train_head.py)
- The installed torch 2.14 defaults for that call are anneal_strategy "cos", div_factor 25, final_div_factor 1e4 (final LR = max_lr/2.5e5) and cycle_momentum True (beta1 cycles 0.95→0.85 against the LR). — [PyTorch OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html); confirmed in `.venv` torch source
- The fraction of max LR at the end of each epoch (simulated from those defaults) is:
  - E=16: epoch 13 0.10, epoch 14 0.05, epoch 16 0.00.
  - E=20: epoch 13 0.33, epoch 14 0.25.
  - E=28: epoch 14 0.59, epoch 15 0.52, epoch 16 0.46, epoch 21 0.18, epoch 24 0.06.

  — [lc_part4.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part4.py)
- **Identities.** Over all 155 finished runs, the mean of the fold differences equals the difference of the fold mean (largest |difference| 2.7e-16). `delta[i] = (m[i+1] − m[i−2])/3` holds for every run within the 2e-5 rounding tolerance, so delta is a 3-epoch secant of the fold-mean curve. Its trailing window is centred about 1.5 epochs behind the epoch it is labelled with. — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)
- **Folds of unequal length (a run in progress).** In a 28-epoch run, cutting fold 4 off at epoch 3, 6, 10, 15, 20 or 27 changed delta by at most 0.0008 and never changed `flat` (15 → 15). Differencing a fold-mean curve over whichever folds are present would instead add spurious steps of up to +0.0142, at the epoch where the partial fold drops out. — [lc_part3.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part3.py)
- **Start-up windows.** `delta[0]` uses 1 difference, `delta[1]` uses 2, and only then are there 3. So `gamma[0] = (d1 − d0)/2` and `gamma[1] = (2·d2 − d0 − d1)/6`, which are not on the same scale as later gamma values (derived from the code above). Gamma at steady state is `(d_t − d_{t−3})/3 = (a_t − a_{t−1} − a_{t−3} + a_{t−4})/3`, a second difference spanning 4 epochs. If the fold-mean jitter σ_m is white noise, SD(delta) = √2·σ_m/3 ≈ 0.47σ_m and SD(gamma) = 2σ_m/3 ≈ 0.67σ_m. Per fold (the terminal values), both are √5 larger. — [lc_part3.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part3.py)
- **Gamma in practice.** From epoch 8 on, the median |seed-mean gamma| / seed SD is 0.29 for the 28-epoch d384 config (25 seeds), 0.61 for the 28-epoch d256 + distillation config (11), 1.14 for 20 epochs + distillation, 0.95 for 20 epochs without distillation and 1.46 for 16 epochs. In 28-epoch runs gamma changes sign at 9.4 to 10.4 of 20 epoch steps within a run, and from epoch 12 on only 52-68% of the d384 seeds agree on its sign, which is chance level. Gamma is consistently negative (agreement 1.00) only in epochs 4-8, where any saturating curve slows down. — [lc_part4.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part4.py)
- **"flat" counts declines as flat.** The test is `v >= FLAT`, so negative deltas pass as flat. The three 24-epoch runs print "flat from epoch 12/11/12 of 24", yet their fold-mean AUC peaked at epochs 11/9/11 and then *fell* by 0.0102/0.0118/0.0091, with delta < 0 in 12-13 of 23 epochs and a minimum delta of −0.0012 to −0.0020. The 12-epoch learning-rate-6e-4 runs print "flat from epoch 8/9 of 12" after declines of 0.0085 and 0.0042. — [lc_part3.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part3.py)
- **No hysteresis.** The smoothed delta crosses FLAT 3.2 times per run on average (at most 7) in the 25 d384 seeds, against 1.0-1.5 times in the other main configs. The flat epoch of the d384 seeds has an SD of 2.1 epochs and ranges from 12 to 21. — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)
- **Latent None bug.** `compact_history` keeps `None` values, but `greeks` removes them *before* indexing by position. A `None` in the middle of a fold would therefore shift every later delta of that fold by one epoch, and `path3d` would show it against the wrong epoch. No `history.jsonl` record in the 155 runs has a `None` AUC, and none has a duplicate (fold, epoch). — [dashboard.py L116, L203-207](file:///Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py); [lc_analyse.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_analyse.py)
- **A false dashboard claim.** The note says "The learning rate falls to zero by the last epoch, so every run ends flat". This is false for 4 runs. All three EMA-0.999 runs end "not flat" (final delta +0.0013 to +0.0015) because they are evaluated on EMA weights, which lag. The 8-epoch run also ends not flat (+0.0015). — [template.html L788](file:///Users/maxranft/Projects/rsna-knee/scripts/dashboard_ui/template.html); [lc_analyse.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_analyse.py)

### Inferences
- Nothing needs to change in the fold-averaging or the indexing. The 3-epoch mean could be written directly as `(m[t] − m[t−3])/3`; that makes it obvious it is a secant with about 1.5 epochs of lag.
- If "flat" is kept, it needs two changes: (a) flag declines separately, for example "peaked at epoch X, then −0.010", instead of calling them flat; and (b) require delta to stay below FLAT for at least 2-3 epochs, or use the seed-band, so that a single crossing does not move the flat epoch.
- Gamma has no decision value. Early in a run it is negative by construction, and late in a run it is noise. The terminal's per-fold gamma is about √5 noisier than the dashboard's. Showing gamma only with seed bands, or not at all, would lose nothing.

### Gaps
- The validation-sampling standard error of a per-epoch change of 0.001 (about 870 studies per fold, paired across epochs) cannot be computed. `history.jsonl` stores AUC values, not per-epoch predictions.

## Q2. Where the curve peaks, how much the final 25% adds, and whether the peak is at the last epoch

### Takeaway
In the adopted configs (slot-drop 0.3, with or without distillation), the fold-mean AUC peaks at 87-92% of the epoch budget and ends within 0.0001-0.0002 of the peak. The final 25% of epochs adds only +0.0005 to +0.0009, which is 0.3-0.6% of the total gain from epoch 1. Real overfitting, where the peak comes early and the curve then declines, occurs only in older or weaker configs: slot-drop 0.1, the 24-epoch runs, 20 epochs without distillation, learning rate 6e-4 and 2 layers. In those, the peak sits at a roughly fixed absolute epoch (about 10 for slot-drop 0.1, 13-14 for the 20-epoch config without distillation) whatever the budget.

### Cited Findings
- By epoch budget, over all finished runs (the fold-mean curve of each run). — [lc_part2.py / summary by E](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)

| E | runs | configs | peak epoch | peak/E | share of runs peaking at the last epoch | share ending ≤0.0005 below peak | mean end−peak | worst end−peak | gain in last 25% | flat epoch | flat/E | not flat | val-loss min/E |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1 | 1 | 8.0 | 1.00 | 1.00 | 1.00 | 0 | 0 | +0.0018 | – | – | 1 | 0.88 |
| 12 | 21 | 10 | 8.9 | 0.74 | 0.10 | 0.43 | −0.0021 | −0.0085 | −0.0009 | 10.0 | 0.83 | 0 | 0.60 |
| 16 | 55 | 11 | 14.3 | 0.89 | 0.40 | 0.82 | −0.0004 | −0.0036 | +0.0009 | 13.0 | 0.81 | 3 (EMA) | 0.66 |
| 20 | 39 | 9 | 15.5 | 0.78 | 0.08 | 0.51 | −0.0008 | −0.0033 | +0.0001 | 13.7 | 0.69 | 0 | 0.56 |
| 24 | 3 | 1 | 10.3 | 0.43 | 0.00 | 0.00 | −0.0103 | −0.0118 | −0.0024 | 11.7 | 0.49 | 0 | 0.36 |
| 28 | 36 | 2 | 24.3 | 0.87 | 0.06 | 1.00 | −0.0002 | −0.0005 | +0.0005 | 15.8 | 0.56 | 0 | 0.49 |

- The main lineages, as seed means. The last column but one is per fold over seeds × folds: the share of folds whose own peak is the last epoch / the share of folds that end within 0.001 of their own peak. — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)

| Config (d, dropout, slot-drop, distillation) | n seeds | report (SD) | gold (SD) | fold-mean peak ep (frac) | end−peak | last-25% gain | per fold: peak at last ep / within 0.001 at end | train min (median) |
|---|---|---|---|---|---|---|---|---|
| 12 ep, dropout 0.1, slot-drop 0.1 (original ref) | 3 | 0.7780 (0.0019) | 0.7186 (0.0052) | 7.7 (0.64) | −0.0030 | −0.0019 | 0.00 / 0.07 | 13.2 |
| 12 ep, dropout 0.3, slot-drop 0.1 | 5 | 0.7825 (0.0014) | 0.7231 (0.0027) | 9.8 (0.82) | −0.0005 | −0.0003 | 0.12 / 0.56 | 14.4 |
| 16 ep, dropout 0.3, slot-drop 0.1 | 5 | 0.7827 (0.0033) | 0.7254 (0.0092) | 10.0 (0.63) | −0.0022 | −0.0011 | 0.00 / 0.12 | 18.3 |
| 24 ep, dropout 0.3, slot-drop 0.1 | 3 | 0.7750 (0.0008) | 0.7238 (0.0122) | 10.3 (0.43) | −0.0103 | −0.0024 | 0.00 / 0.00 | 30.3 |
| 16 ep, dropout 0.3, slot-drop 0.3 (best without distillation) | 23 | 0.7875 (0.0014) | 0.7366 (0.0076) | 14.7 (0.92) | −0.0001 | +0.0006 | 0.08 / 0.83 | 17.3 |
| 20 ep, same, no distillation | 9 | 0.7873 (0.0009) | 0.7396 (0.0097) | 14.0 (0.70) | −0.0011 | −0.0004 | 0.00 / 0.27 | 20.2 |
| 20 ep + distillation 0.5, d256 | 9 | 0.7972 (0.0010) | 0.7460 (0.0032) | 18.1 (0.91) | −0.0001 | +0.0009 | 0.07 / 0.91 | 15.7 |
| 28 ep + distillation 0.5, d256 | 11 | 0.7999 (0.0010) | 0.7515 (0.0067) | 24.4 (0.87) | −0.0002 | +0.0006 | 0.00 / 0.91 | 21.7 |
| 28 ep + distillation 0.5, d384 ("current best") | 25 | 0.7993 (0.0013) | 0.7461 (0.0083) | 24.3 (0.87) | −0.0002 | +0.0005 | 0.04 / 0.69 | 43.8 |

- The seed-mean fold-mean curve at 25/50/75/90/100% of the budget. — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)
  - 16 ep, slot-drop 0.3: 0.7470 / 0.7813 / 0.7882 / 0.7887 / 0.7888 (peak at epoch 16).
  - 20 ep, no distillation: 0.7584 / 0.7859 / 0.7888 / 0.7885 / 0.7884 (peak at 13).
  - 20 ep + distillation: 0.7748 / 0.7933 / 0.7974 / 0.7982 / 0.7982 (peak at 20).
  - 28 ep + distillation, d256: 0.7854 / 0.7974 / 0.8003 / 0.8010 / 0.8009 (peak at 26).
  - 28 ep + distillation, d384: 0.7854 / 0.7966 / 0.7998 / 0.8003 / 0.8003 (peak at 28).
  - 24 ep, slot-drop 0.1: 0.7765 / 0.7847 / 0.7786 / 0.7763 / 0.7762 (peak at 11).
  - Over the last 10% of epochs the change is between −0.0001 and +0.0000 in every one of these lineages.
- `experiments.md` agrees on the overfitting cases:
  - The original 12-epoch reference "peaks at epoch 7 in 4/5 folds (train 0.31 vs val 0.42 BCE: overfits)".
  - The 24-epoch runs: "val AUC peaks near epoch 9 then falls, train 0.25 vs val 0.49 BCE".
  - An 8-epoch run was rejected (report −0.0049) because "the shorter OneCycle schedule also shortens annealing, so it does not reproduce the 12-epoch run's epoch-7 peak".

  — [experiments.md rows 20260929-140030, 20260930-042243, 20260929-160636](file:///Users/maxranft/Projects/rsna-knee/experiments.md)

### Inferences
- In the adopted configs, the peak is effectively at the end of the run. Saving the last epoch costs at most 0.0001-0.0002 against an oracle epoch choice on the fold-mean curve, and 0.0004-0.0008 per fold. Early stopping would gain nothing and would make the OOF optimistic.
- Distillation is what turned the peak from an absolute epoch into a fraction of the budget. Without it, the 20-epoch config peaks at epoch 13-14, just as the 16-epoch config does at 13-15. With it, the peak moves to 18/20 and 24/28. That is the precondition for longer budgets helping (Q5).
- The annealing tail (last 25%) is not where the gains of a longer budget come from: it adds only +0.0005. The gain comes from a longer high-learning-rate phase, so the anneal cannot be cut on its own. Shortening OneCycle compresses the whole schedule, as the 8-epoch run showed.

### Gaps
- There is no per-epoch gold curve; only the report-label AUC is logged. Gold could peak earlier or later than report, and nothing here can show it.
- No distilled config was run at 16 or at 36+ epochs, so budget and config are confounded across lineages. Only the 20 → 28 pair with distillation at d256 is a clean comparison.

## Q3. Does "flat from epoch X" carry information beyond the learning-rate schedule?

### Takeaway
Mostly it reflects the schedule. The absolute flat epoch sits at about 13-16 whatever the budget, so flat/E falls from 0.82 at 16 epochs to 0.52-0.59 at 28. At a fixed budget, configs do differ: at 16 epochs flat ranges from 11.0 to 14.3 (Kruskal-Wallis p < 0.001), and the config mean correlates with report AUC (ρ ≈ +0.67-0.69) but not significantly with gold. Across seeds of the same config, flat does not predict the outcome. A simpler measure, the decline from the peak (end − peak), predicts both report and gold better. In 28-epoch distilled runs, "flat" fires while the learning rate is still at 40-60% of its maximum, and 0.0024-0.0036 of AUC is still to come after it.

### Cited Findings
- Flat epoch by budget: 10.0 at 12 epochs (0.83 of E), 13.0 at 16 (0.81), 13.7 at 20 (0.69) and 15.8 at 28 (0.56). — [summary by E](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)
- At 16 epochs, flat differs across the 10 configs with at least 3 seeds (Kruskal-Wallis H = 35.4, p < 0.001). Means range from 11.0 (bagging) and 11.2 (slot-drop 0.1) to 14.0 (batch size 32, slice-drop) and 14.3 (slot-drop 0.5). — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)
- At 20 epochs there is no significant difference across 9 configs (H = 10.7, p = 0.22; means 12.7-15.0). At 28 epochs, d384 flattens later than d256 (16.4 ± 2.1 vs 14.5 ± 1.0, p = 0.004), yet d256 has the better report and gold (Q5). — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)
- Correlations of the config-mean flat epoch with outcome (Spearman): — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)
  - At 16 epochs (10 configs): ρ = +0.67 with report (p = 0.03), +0.11 with gold (p = 0.76).
  - At 20 epochs (9 configs): +0.69 with report (p = 0.04), +0.44 with gold (p = 0.24).
  - Across all 24 configs with at least 3 seeds, using flat/E: −0.02 with report and −0.05 with gold, because the budget dominates.
- For comparison, end − peak (how far the curve has fallen from its peak by the end) correlates more strongly. — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)
  - Across 24 configs: ρ = +0.63 with report (p = 0.001) and +0.59 with gold (p = 0.003).
  - At 20 epochs: +0.98 with report and +0.82 with gold.
  - At 16 epochs: +0.60 with report (p = 0.07) and +0.26 with gold.
- Within a config, across seeds, flat does not predict the outcome. — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)
  - 16 epochs (23 seeds, flat 12-15): ρ = +0.23 with report, −0.36 with gold (p = 0.09).
  - 28-epoch d384 (25 seeds, flat 12-21): +0.19 and +0.22.
  - 28-epoch d256 (11 seeds): +0.38 and −0.29.
  - 20-epoch configs: |ρ| ≤ 0.25.
  - All p ≥ 0.09.
- Fold-mean AUC gained after the flat epoch (m[E] − m[flat]): — [lc_part4.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part4.py)

| Config | Mean gain after flat | Range |
|---|---|---|
| 16 ep | +0.0003 | −0.0006 to +0.0010 |
| 20 ep, no distillation | −0.0005 | |
| 20 ep + distillation | +0.0019 | +0.0004 to +0.0032 |
| 28 ep + distillation, d256 | +0.0036 | +0.0022 to +0.0053 |
| 28 ep + distillation, d384 | +0.0024 | +0.0001 to +0.0069 |
| 24 ep, slot-drop 0.1 | −0.0087 | |

### Inferences
- In 16-epoch runs, flat arrives when the learning rate is down to about 10% of its maximum, so "flat at epoch 13" is the anneal and nothing more. In 28-epoch distilled runs, flat (epoch 14.5-16.4) comes while the learning rate is at 40-60% of its maximum. After it, the curve keeps climbing slowly but consistently: the gain after flat is +0.0022 to +0.0053 for every d256 seed, 2-4 times the report seed SD of about 0.001. **A flat epoch must therefore not be read as "budget large enough".**
- The between-config signal that flat does carry at a fixed budget is mostly whether the config overfits early (flat early, peak early, then decline). End − peak, or peak/E, measures that directly, with less noise and a clear sign. The dashboard would be better served by showing peak epoch and end − peak per run than flat.

### Gaps
- The correlations rest on 9-10 configs per budget, whose seeds overlap in time and machine state. They show association only; the configs differ in several ways at once (for example the bagging and EMA runs).

## Q4. How large the epoch-to-epoch noise is compared with FLAT = 0.001

### Takeaway
Per-fold epoch-to-epoch jitter is about 0.004-0.006 early in a run, 0.0013-0.0023 in the middle and 0.0003-0.0008 at the end. FLAT = 0.001 therefore sits below the mid-run per-fold noise and above the end-of-run noise. After averaging over folds and taking the 3-epoch secant, the seed SD of delta at the flat epoch is 0.0003-0.0004, so the threshold is detectable at about 2.5-3 SD. But the epoch at which it is crossed is uncertain by 0.7-2.1 epochs. More importantly, 0.001 per epoch is not tied to anything that matters for a decision: accumulated sub-threshold gains after "flat" (0.002-0.004) exceed the report seed SD (about 0.001). The jitter is optimisation noise, because each fold's validation set is fixed, so it shrinks as the learning rate falls; that is why every annealed run "goes flat".

### Cited Findings
- Per-fold σ (SD of second differences / √6) and the median per-fold |epoch-to-epoch change|, by phase of training, with the seed spread of delta. Early = epochs 2 to 0.4E, mid = 0.4E to 0.75E, late = the last 25%. — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)

| Config | σ early / mid / late | median abs. change, mid / late | seed mean ± SD of delta at mean flat epoch | share of seeds below FLAT there | delta at last epoch (mean / SD) |
|---|---|---|---|---|---|
| 16 ep, slot-drop 0.3 (23 seeds) | 0.0050 / 0.0020 / 0.0006 | 0.0031 / 0.0004 | +0.0014 ± 0.0004 (epoch 13) | 0.13 | +0.00008 / 0.00010 |
| 20 ep, no distillation (9) | 0.0064 / 0.0018 / 0.0006 | 0.0021 / 0.0004 | +0.0010 ± 0.0004 (epoch 14) | 0.44 | −0.00004 / 0.00013 |
| 20 ep + distillation (9) | 0.0045 / 0.0015 / 0.0004 | 0.0018 / 0.0004 | +0.0012 ± 0.0003 (epoch 13) | 0.22 | +0.00003 / 0.00008 |
| 28 ep + distillation, d256 (11) | 0.0038 / 0.0013 / 0.0003 | 0.0012 / 0.0003 | +0.0011 ± 0.0003 (epoch 14) | 0.36 | −0.00001 / 0.00003 |
| 28 ep + distillation, d384 (25) | 0.0042 / 0.0015 / 0.0004 | 0.0014 / 0.0003 | +0.0009 ± 0.0004 (epoch 16) | 0.68 | +0.00001 / 0.00006 |

- The configs not in the table above (12 epochs and slot-drop 0.1 lineages) show σ late of 0.0004-0.0008 and median |Δ| mid of 0.0018-0.0056. — [lc_part2.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part2.py)
- Theory checks out against these numbers. With a per-fold σ of 0.0015 in mid-training, the fold-mean σ is 0.00067, giving SD(delta) = 0.00032 (the observed seed SD is 0.0003-0.0004) and SD(gamma) = 0.00045. The per-fold terminal values from progress.py have SD(delta) = 0.0007 and SD(gamma) = 0.0010, about the size of FLAT itself. — [lc_part3.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part3.py)
- The seed SD of the final fold-mean validation AUC is 0.0009-0.0014, the same as the seed SD of pooled report AUC (0.0009-0.0014). Gold seed SD is 0.0032-0.0097. A single fold's final AUC varies by 0.0014-0.0041 across seeds. — [lc_part2.py, lc_part4.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part4.py)
- The project's own noise rules are "report macro varies by ~0.004, gold by ~0.010" across seeds, and "Gold differences below about 0.02 are noise". — [experiments.md header](file:///Users/maxranft/Projects/rsna-knee/experiments.md)

### Inferences
- The question FLAT should answer is "would more epochs at this budget change the final AUC by more than seed noise?". A per-epoch threshold cannot answer it, because a rate of 0.0009 per epoch over 12 remaining epochs adds up to about 0.01. A more meaningful rule is the cumulative remaining gain, m[E] − m[e], against the report seed SD (about 0.001). On that rule, 16-epoch runs are effectively done by epoch 12-13; 28-epoch distilled runs are not done until about epoch 21-24.
- Because the jitter is proportional to the learning rate (σ late ≈ σ mid / 3-4), "flat" partly measures the learning-rate decay itself. That confirms Q3: the label is mostly the schedule.

### Gaps
- The validation-set sampling error, the uncertainty that matters for whether +0.001 carries over to the test set, cannot be estimated without per-epoch OOF predictions.

## Q5. Did longer training (16 → 20 → 28 epochs) improve OOF report and gold AUC, and at what time cost? What follows for the final week?

### Takeaway
It depends on regularisation:
- **Without distillation**, longer budgets did not help single models: 12 → 16 gave a tie, 16 → 20 gave report −0.0002, and 16 → 24 gave report −0.0077.
- **With distillation (d256)**, 20 → 28 epochs gave the only clear, all-seed gain: report +0.0027 (90% CI +0.0019 to +0.0035, Cliff's δ +0.93) and gold +0.0055 (CI +0.0016 to +0.0095), for +38% training time (15.7 → 21.7 min).
- Separately, the current "best" d384 is *not* better than d256 at 28 epochs on 25 vs 11 seeds: report −0.0006 and gold −0.0054 (CI −0.0098 to −0.0009), at about 1.3-2× the training time.

### Cited Findings
- Seed-level comparisons from `metrics.json` (pooled OOF), with Welch 90% intervals: — [lc_part3.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part3.py); verdicts from [experiments.md](file:///Users/maxranft/Projects/rsna-knee/experiments.md)

| Comparison | seeds | report diff (90% CI), Cliff | gold diff (90% CI), Cliff | median training min (per epoch) | experiments.md verdict (3-run ensembles) |
|---|---|---|---|---|---|
| 12 → 16, dropout 0.3, slot-drop 0.1 | 5 v 5 | +0.0002 (−0.0030..+0.0033), −0.08 | +0.0023 (−0.0064..+0.0110), +0.04 | 14.4 → 18.3 (1.20 → 1.15) | Adopted for ensemble members: ensemble report +0.0047 (CI +0.0026..+0.0066), gold +0.006; "single models tie", seeds more diverse |
| 16 → 24, same (seeds 42-44) | 3 v 3 | −0.0077 (−0.0148..−0.0006), −1.00 | −0.0050 (−0.0250..+0.0150), −0.33 | 16.5 → 30.3 | Rejected: ensemble report −0.0010, gold +0.0015 |
| 16 → 20, best config without distillation | 9 v 23 | −0.0002 (−0.0010..+0.0005), −0.19 | +0.0030 (−0.0035..+0.0094), +0.14 | 17.3 → 20.2 (1.08 → 1.01) | Adopted (auto): ensemble report +0.0019 (CI +0.0000..+0.0036), gold −0.0068 |
| 20 → 28, distillation 0.5, d256 | 11 v 9 | **+0.0027 (+0.0019..+0.0035), +0.93** | **+0.0055 (+0.0016..+0.0095), +0.56** | 15.7 → 21.7 (0.78 → 0.78) | Adopted (auto): ensemble report +0.0016 (CI +0.0002..+0.0029), gold +0.0122 (P 0.95) |
| d384 vs d256, both 28 ep + distillation | 25 v 11 | −0.0006 (−0.0013..+0.0001), −0.26 | **−0.0054 (−0.0098..−0.0009), −0.40** | 43.8 vs 21.7 (1.56 vs 0.78) | d384 Adopted (auto) on 3 seeds: report +0.0020 (CI +0.0008..+0.0034), gold −0.0164 (P 0.03), accepted under "gold no worse than −0.02" |

- On the paired seeds 42-44, 28 minus 20 epochs (distillation, d256) gives report −0.0004 / +0.0017 / +0.0012 and gold +0.0093 / +0.0200 / +0.0077. — [lc_part3.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part3.py)
- On the same seeds, d384 minus d256 at 28 epochs gives report +0.0004 / +0.0019 / +0.0035 but gold −0.0309 / −0.0173 / −0.0044. With 25 vs 11 seeds, the report advantage disappears. — [experiments.md rows 20261006-154048..171638 and 174748..202938](file:///Users/maxranft/Projects/rsna-knee/experiments.md)
- Training minutes are `train_minutes` (median epoch time × epochs), which leaves out pauses and sleep. Machine state still varies: d384 runs at 28 epochs took 28.1-64.9 min, the last three seeds (104-106) 28.1 / 31.3 / 32.0. The d256 runs at 28 epochs took 21.2-21.9 min for 8 of 11 seeds. So d384 costs at least about 1.3× and typically about 2× per run. — [lc_part3.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part3.py); [dashboard.py train_minutes L210-215](file:///Users/maxranft/Projects/rsna-knee/src/knee/dashboard.py)
- The ensemble is saturated: 21 vs 18 members gives report +0.0003, and running all 73 same-architecture runs instead of 23 gives only +0.0006 report "for 3x the heads". The adopted accurate submission is mix-20261006-114310 (gold 0.771, report 0.8031) and the efficiency submission is mix-20261006-114316 (gold 0.7652, report 0.8034, 9 distilled 20-epoch seeds). — [experiments.md](file:///Users/maxranft/Projects/rsna-knee/experiments.md)
- The deadline for the final submission is 2026-10-22, and only one training can run on the Mac at a time. — [CLAUDE.md](file:///Users/maxranft/Projects/rsna-knee/CLAUDE.md)

### Inferences
- **Epoch budget for the final week: keep 28 epochs with distillation.** Do not shorten it on the strength of the flat epoch (Q3: 0.002-0.004 is gained after flat). The last 25% adds only +0.0005 on the fold-mean curve, but it is part of the same OneCycle schedule. Stopping a 28-epoch schedule at epoch 21 would save about 25% of the time and lose about 0.0006 fold-mean AUC, leaving unannealed weights; it is not worth it given that the ensemble is saturated.
- **Going past 28 epochs is untested and only weakly supported.** With distillation, the peak scales with the budget (peak/E 0.87-0.91) and the last quarter still rises a little, so 36 epochs might add something. Returns are diminishing, though: the 20 → 28 gain was +0.0027 report. At d256, a 3-seed check of 36 epochs costs about 3 × 28 min ≈ 1.4 h of Mac time (serves S3). It should be judged on gold as well as report, since gold rose in the 20 → 28 step.
- **Config choice beats budget choice.** On current evidence, the d256 config at 28 epochs is at least as good as d384 on report, better on gold (CI excludes 0), and roughly twice as fast. Any final-week seeds are better spent on d256. The d384 adoption shows the winner's curse in the 3-seed auto rule: a +0.0020 report edge on 3 seeds did not hold up over 25. This concerns both submissions (S3/S4).
- The useful learning-curve summaries for budget decisions are **peak/E**, **end − peak**, and the **cumulative gain over the last N epochs, compared with the seed SD**. Delta, gamma and flat add nothing beyond these.

### Gaps
- No d384 run at 20 epochs, and no distilled run at 16 or 36 epochs, so the effects of budget and config cannot be fully separated.
- Gains at ensemble level for 20 → 28 come only from 3-run ensembles. The marginal value of longer budgets inside the saturated 160-head ensemble is unmeasured.
- Training time is confounded with the Mac's state (lid closed, battery, throttling, concurrent work).

## Q6. Do validation loss and validation AUC peak at different epochs?

### Takeaway
Yes, consistently. Validation BCE is lowest at about 0.36-0.66 of the budget, while AUC peaks later, in 94% of runs and by a median of 4 epochs. The gap grows with the budget and with distillation: 10-12 epochs at 28 epochs with distillation. In the adopted configs, loss rises only 0.3-0.7% after its minimum while AUC keeps improving, so early stopping on loss would cost 0.0015-0.0043 AUC (0.003-0.004 in the distilled configs). The loss minimum's epoch is unrelated to final report AUC (ρ −0.05), so loss is not a usable stopping signal here. Only in heavy overfitting (24 epochs, slot-drop 0.1: loss +15.8%) do the two agree that training has gone too far.

### Cited Findings
- Loss minimum against AUC peak, on the seed-mean curves: — [lc_part3.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part3.py)

| Config | loss min ep | AUC peak ep | mean per-run gap (AUC peak − loss min) | loss rise min → end | AUC at loss-min ep vs end |
|---|---|---|---|---|---|
| 12 ep, dropout 0.3, slot-drop 0.1 | 8/12 | 9/12 | +2.8 | +0.0046 (1.1%) | 0.7837 vs 0.7839 |
| 16 ep, dropout 0.3, slot-drop 0.1 | 8/16 | 10/16 | +2.8 | +0.0186 (4.6%) | 0.7843 vs 0.7839 |
| 24 ep, dropout 0.3, slot-drop 0.1 | 9/24 | 11/24 | +1.7 | +0.0643 (15.8%) | 0.7858 vs 0.7762 |
| 16 ep, slot-drop 0.3 | 11/16 | 16/16 | +4.7 | +0.0028 (0.7%) | 0.7873 vs 0.7888 |
| 20 ep, no distillation | 10/20 | 13/20 | +3.3 | +0.0100 (2.5%) | 0.7859 vs 0.7884 |
| 20 ep + distillation | 11/20 | 20/20 | +6.0 | +0.0013 (0.3%) | 0.7950 vs 0.7982 |
| 28 ep + distillation, d256 | 13/28 | 26/28 | +11.5 | +0.0024 (0.6%) | 0.7966 vs 0.8009 |
| 28 ep + distillation, d384 | 15/28 | 28/28 | +10.2 | +0.0028 (0.7%) | 0.7972 vs 0.8003 |

- Over all 155 runs, AUC peaks after the loss minimum in 94% of runs (median gap +4 epochs), and Spearman ρ between loss-min/E and report AUC is −0.05. — [lc_part3.py](file:///private/tmp/claude-501/-Users-maxranft-Projects-rsna-knee/82de634b-7df4-44e2-934a-83895e096829/scratchpad/lc_part3.py)
- Validation loss is computed against the training targets (soft report scores, plus gold labels for gold studies) on all validation studies. AUC is computed on report-only studies against labels thresholded at 0.5. — [train_head.py L57-69, L193-198](file:///Users/maxranft/Projects/rsna-knee/scripts/train_head.py)
- `experiments.md` records train/validation BCE gaps of 0.31 vs 0.42 for the 12-epoch reference and 0.25 vs 0.49 for the 24-epoch runs, both labelled overfitting. — [experiments.md](file:///Users/maxranft/Projects/rsna-knee/experiments.md)

### Inferences
- A rising validation BCE with a still-rising AUC is the usual pattern of growing overconfidence: calibration gets worse while ranking still improves. The project's metric is rank-based, and the ground rules say "no calibration", so validation loss should not drive budget choices. The 4-15% loss rises seen in the slot-drop 0.1 and 24-epoch runs are the only cases where loss and AUC agree that training went too far.
- Part of the mismatch is built in, because loss and AUC use different targets and subsets (soft targets including gold, against hard report labels on report-only studies). A loss computed on report-only studies against the same hard labels would make the comparison cleaner.

### Gaps
- Per-label curves (in `history.jsonl` under `val_auc`) were not analysed. Weak labels such as Synovitis, Medial Meniscus and MCL might peak at different epochs from the macro average.
