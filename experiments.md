# Experiment log

One row per run. Same folds and seed unless noted. CV = macro AUC on the pooled out-of-fold predictions. **Gold** (58 studies) is the final arbiter;
**Report** (4,349 report-only studies, against labeller v-N soft labels thresholded at 0.5) is the lower-variance signal.
Gold differences below about 0.02 are noise.

| Date | ID | Strategy | Change (one thing) | Gold macro AUC | Report macro AUC | Weakest 3 labels (gold AUC) | Runtime / study | Notes |
|---|---|---|---|---|---|---|---|---|
