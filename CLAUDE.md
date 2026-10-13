# RSNA Knee Abnormality Detection

Max's entry to the Kaggle competition https://www.kaggle.com/competitions/rsna-knee-abnormality-detection:
12 binary findings per knee MRI study, scored by macro ROC AUC. Offline Kaggle notebook, ≤9 h, about
1,300 test studies. **Entry deadline 2026-10-15, final submission 2026-10-22.**

A follow-on to Max's earlier diabetic-retinopathy project (APTOS 2019, EfficientNet-B0, FastAPI
triage app; see the Obsidian vault chat "2026-08-30 - Re-run the training scripts").

## Goals and strategy
Medal on the main leaderboard, **and** a top-3 finish on the efficiency track. No big-GPU budget,
so we compete on **labels, iteration speed and efficiency**, not compute. Full detail and
milestones are in [docs/plan.md](docs/plan.md).
- **S1:** the best report-derived labels (rules tuned against gold, soft labels).
- **S2:** extra auxiliary targets extracted from reports.
- **S3:** frozen-encoder feature cache, so most experiments run in minutes on the Mac or CPU.
- **S4:** one small distilled model for the efficiency track, ≤ 30 min runtime.

When proposing work, say which of S1–S4 it serves. If it serves none, question it.

## Read first
- [docs/competition.md](docs/competition.md) — rules, metric (including the efficiency prize), data layout, deadlines.
- [docs/data.md](docs/data.md) — what the data actually contains: 58 gold studies, series coverage, report languages, DICOM survey.
- [docs/labels.md](docs/labels.md) — the 12 targets: MRI signs, best series, report phrasing, correlations, series-slot plan.
- [docs/plan.md](docs/plan.md) — decisions so far: all-cloud compute, model size, training approach, order of work.
- [reference/STUDY.md](reference/STUDY.md) — what the RSNA 2024 lumbar top-9 did and what ports to the knee.
- [docs/landscape.md](docs/landscape.md) — clinical AI products and open resources.

## Ground rules (settled — don't relitigate)
- **Validate on both, gold decides (2026-09-23).** Every run reports (a) macro AUC on the pooled OOF predictions for the 58 gold studies, the final arbiter, and (b) macro AUC against the report-derived labels on the 4,349 OOF report-only studies, a lower-variance signal for choosing between models. When they disagree, gold wins. Gold differences below about 0.02 are noise.
- **Never flip left/right** unless the knee has been canonicalised *and* Medial↔Lateral labels (Meniscus, OA) are swapped at the same time. Lumbar 7th place lost score to exactly this bug.
- AUC is rank-based: no calibration or threshold post-processing. Rank-average ensembles.
- Keep report text inside Kaggle (open-weights LLM on Kaggle). Don't send it to external APIs unless the rules are confirmed to allow it.
  **This includes Claude sessions:** never print report text or evidence snippets, and never view MRI images in a Claude session. Anything shown there goes to Anthropic's API. Work from aggregates and metrics; Max looks at reports and images directly (Kaggle viewer notebook).
- Small models first (about 6–15 M parameters). Get an end-to-end baseline submitted before improving anything.
- One change at a time on fixed folds. Log every run in [experiments.md](experiments.md) (date, change, CV macro AUC, weakest 3 labels, runtime).
- Keep two final submissions in mind: **accurate** (main LB) and **efficient** (efficiency track).
- Code from reference repos without a licence file (01_wadakoki, 02_bartley, 05_siwooyong, 06_darragh, 08_kmat) is for reading only. The MIT/Apache ones may be adapted with attribution.

## Layout
- `docs/` — hand-written knowledge (keep it current).
- `reference/code/` — cloned lumbar solutions, git-ignored and about 220 MB. Re-clone the list in `reference/STUDY.md` if missing. `reference/nb2py.py` converts notebooks to greppable `.nb.py`.
- `.venv/` — Python with `kaggle`, `nbformat`, `pypdf`. Kaggle credentials are in `~/.kaggle/kaggle.json` (never print them).
- `external scan/` — one real, identifiable patient's knee MRI (DICOM from a hospital viewer), used only to test the model with `scripts/external_study.py` (`start/4_run_external.command`). Git-ignored. Not part of the submission: keep it out of any heads packaged for Kaggle. **Max allowed Claude to view this scan and read its radiology report (2026-10-02)**, e.g. to download it and to fill in `truth.csv`. That permission covers the external scan only; competition images and report text stay off-limits.

## Environment
- MacBook M3, 16 GB. Heavy work runs on Kaggle notebooks, or a rented GPU if needed.
- Also record progress in the Obsidian vault thread (`~/Downloads/MyVault/Memory/Open Threads.md`).
