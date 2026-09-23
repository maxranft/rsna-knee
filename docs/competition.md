# Competition facts

Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection (read 2026-09-22).
Host: Radiological Society of North America. Research code competition. $77,000 in prizes.

## Deadlines (all 23:59 UTC)
- **2026-10-15:** entry deadline. Accept the rules before this. Team merger deadline is the same day.
- **2026-10-22:** final submission deadline.
- **2026-11-05:** winners' deadline for code, a video and a method description. Winners must also open-source the code **and weights** and publish the final model.

## Task
Predict 12 binary findings per knee MRI study. See [labels.md](labels.md).
Submission `submission.csv` has the columns:
`StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`

## Metric
- **Main score:** the macro-averaged ROC AUC over the 12 labels. Only ranking matters; calibration and prevalence shift don't.
  - Each label counts equally.
  - Rank-average ensembles.
  - Watch the per-label AUC.
  - The all-0.5 sample submission scores 0.5.
- **Efficiency prize** (reconstructed from the page's maths rendering; the page is authoritative): `AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400`, lower is better.
  - Benchmark = 0.5, and maxAUC = the best private score.
  - If maxAUC ≈ 0.90, **1 hour of runtime ≈ 0.044 AUC**.
  - Prizes are $7k, $6k and $5k.
  - To be eligible, a submission must be one of the team's selected ones and must beat the sample submission.

## Data (569.76 GB, 819,640 files)
- **`train.csv`:** `StudyInstanceUID`, `Report` (free text, **multilingual**), plus the 12 labels. **Only a small subset of studies has labels**; the rest have only the report.
- **`train_series.csv`:** `StudyInstanceUID`, `SeriesInstanceUID`, and the flags `Fluid_Sensitive`, `Fat_Suppression` and `Anatomical_Plane` (Sagittal / Coronal / Axial). The two flags are identical on every training row (checked 2026-09-23, see data.md).
- **`train_series/<study>/<series>/<SOP>.dcm`:** one slice per file.
  - Slices per series: median 30, usually 20–45, with a tail up to a few hundred.
  - Mixed transfer syntaxes: uncompressed Explicit/Implicit VR, **JPEG Lossless, JPEG 2000**.
  - Only an **allowlist of 86 tags** is kept.
- **`test.csv` / `test_series.csv` / `test_series/`:** 3 example studies; the real test set is about **1,300 studies**. **No Report at test time.**
- The data covers about 22 sites across Europe, the Americas, Asia, Africa and Australia, with varied scanners and protocols.
- **Prevalence differs between train, public LB and private LB.**

## Code requirements
- Kaggle Notebook, CPU or GPU, **≤ 9 h** runtime, **internet off**.
- Free, publicly available external data and pretrained models are allowed. Check the discussion forum for rulings on specific datasets. MRNet and OAI need registration and are non-commercial.
- Offline dependencies (e.g. `pylibjpeg`, `gdcm` for JPEG 2000) must be attached as datasets.
