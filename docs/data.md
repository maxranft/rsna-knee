# Data: what is actually there (EDA, 2026-09-23)

Metadata CSVs are in `data/raw/` (git-ignored, about 9 MB, fetched with the Kaggle CLI). The DICOMs stay on
Kaggle. The DICOM findings come from the `rsna-knee-dicom-survey` kernel ([../kaggle/dicom-survey/](../kaggle/dicom-survey/)).

> **Report text never gets printed into a Claude session.** Anything printed there goes to Anthropic's API.
> Look only at aggregates (lengths, script / language counts, label rates). Read individual reports
> in a Kaggle notebook or locally, outside Claude.

## Studies and labels
- `train.csv`: **4,407 studies**, each with a report. **Only 58 have gold labels**, and they have all 12 labels (all 0/1).
  The other 4,349 have a report only.
- Gold positives (n = 58): ACL 24, MCL 9, Medial Meniscus 26, Lateral Meniscus 23, Medial OA 15, Lateral OA 11,
  PF OA 21, Effusion 35, Synovitis 27, Baker's 12, Contusion 19, Fracture 18.
  The gold set is clearly enriched: positives per study range from 1 to 9, and none has 0.
- **Consequence:** the AUC on 58 studies is very noisy. MCL has only 9 positives, and one flipped rank pair moves its
  AUC by about 0.002. Across all labels the standard error is about 0.05–0.08, and more for the rare ones. See "Validation" below.

## Series
- `train_series.csv`: 24,371 series. Per study: min 3, median 5, max 14.
- `Fluid_Sensitive == Fat_Suppression` on **every** row, so the two flags carry one bit between them.
- Share of studies with at least one series of each type:

  | | Sag | Cor | Ax |
  |---|---|---|---|
  | fluid-sensitive (FS) | 0.94 | 0.96 | **1.00** |
  | non-FS | 0.97 | 0.77 | 0.19 |

- **All 4 planned slots** (Sag FS, Cor FS, Ax FS, Sag non-FS) are present in 87% of studies, and the
  three FS planes in 91%. So the missing-slot mask is needed.
- 7–20% of studies have 2 or more series of the same type, so we need a rule to pick one (e.g. the most slices).

## Reports
- Length: median 977 characters (range 52–4,743).
- Script: Latin 3,866, Greek 321, Cyrillic 220. No Thai, CJK or Arabic.
- Rough stop-word language guess for the Latin-script reports: English ~1,700, Turkish ~530, Spanish ~365,
  German ~250, French ~240, unassigned ~770 (probably Italian, Croatian and others).
  So about 60% of reports are not in English.
- Gold subset: about half English, and the rest a similar mix.

## Folds
`meta/folds.csv` (committed, never regenerate) is made by `scripts/make_folds.py`, seed 42:
- 5 folds, about 881 studies each.
- The gold studies are multilabel-stratified: 10–13 per fold, with at least 1 positive per label per fold.
- The report-only studies are split at random.

CV = AUC on the **pooled out-of-fold predictions for the 58 gold studies**.

## Validation (open question for Max)
The rule is "validate only on gold". With 58 gold studies, a CV difference below about 0.02 macro AUC is noise.
Proposal: keep gold as the **final arbiter**, but also report AUC against the (validated) report labels on the
4,349 out-of-fold report-only studies. That is a large, lower-variance signal for choosing between models.
Use it only after the labeller's agreement with gold is measured.

## DICOM (survey kernel)
_Pending: surviving tags, transfer syntaxes, slices per series, decode speed._
From the first smoke test on one test series:
- `Laterality`, `BodyPartExamined`, `SeriesDescription`, `Manufacturer` and `MagneticFieldStrength`,
  plus the full geometry (`ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`), all survive.
- That series is 960×960 at 0.156 mm pixel spacing, with 3 mm slices.
