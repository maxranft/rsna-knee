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

## Validation (decided 2026-09-23: both, gold decides)
The rule is "validate only on gold". With 58 gold studies, a CV difference below about 0.02 macro AUC is noise.
Decision: keep gold as the **final arbiter**, but also report AUC against the (validated) report labels on the
4,349 out-of-fold report-only studies. That is a large, lower-variance signal for choosing between models.
Use it only after the labeller's agreement with gold is measured.

## DICOM (survey kernel, run 2026-09-23)
Source: [../kaggle/dicom-survey/survey.py](../kaggle/dicom-survey/survey.py). Outputs are in `data/survey/` (git-ignored).
The kernel listed all 24,371 series and read 4,428 headers (first and middle slice of every series in 400 random studies).
- **819,078 slices.** Per series: median 30, IQR 25–34, max 320. 749 series (3%) have more than 80 slices,
  mostly axial and sagittal (3D / thin-slice sequences), so the slot sampler must subsample them.
  Kaggle mounts the data at `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- **Every sampled file is uncompressed Explicit VR Little Endian.** There is no JPEG 2000 in the sample, despite the
  competition page mentioning it. Decoding takes under 1 ms per slice with any plugin.
  **I/O is the cost: about 14 ms median per file** from the Kaggle mount. Keep the JPEG plugins in the submission
  anyway, in case the hidden test set differs.
- **Throughput:** 218 files/s with 4 processes, cold I/O.
  - The whole training set is about 1 h, so preprocessing fits in one CPU session.
  - The test set, 4 slots with all slices, is about **12 min**, which is about 0.009 AUC on the efficiency score.
  - Threads may do better than processes, since the work is I/O-bound. Try that in the preprocessing kernel.
- **Laterality:** clean `L` or `R` tag in about **50% of studies**, conflicting (`L`+`R`, or `B`) in about 1.5%, missing in about 48%.
  `ImageLaterality` never appears. So we need another source:
  - **Train time:** the report, which usually says "right/left knee". Extract it in the S1 labeller.
  - **Test time:** the tag where present, otherwise a small L/R classifier trained on the tagged half, or
    anatomy (the fibular head is lateral). Canonicalisation only matters for the 4 Medial/Lateral labels. Until it
    is solved, just **never flip**.
- **Geometry:** pixel spacing median 0.31 mm (0.125–1.17); field of view median 160 mm (IQR 160–170); slice thickness
  median 3 mm. Matrix sizes range from 256² to 1024², with 512² the most common.
  So resample to a fixed mm grid (e.g. 160 mm → 256 px = 0.625 mm/px), not a fixed pixel count.
- **Scanners:** Siemens, GE, Philips and Toshiba/Canon, with vendor names in several spellings, so normalise
  them before using as metadata. Field strength: 1.5 T in about 60%, 3 T in about 37%.
  `PatientPosition` is FFS whenever present.
- **Tags worth using:** `SeriesDescription` (95%) for picking between duplicate series of the same type;
  `RepetitionTime`/`EchoTime`/`InversionTime` for the sequence type; `RescaleSlope`/`Intercept`
  (45%, Philips-style, so apply the modality LUT before normalising).
  `NumberOfFrames` is present in 20% of headers, so **check for multi-frame files** before trusting file
  count = slice count.
  `BodyPartExamined` is unreliable (HEAD, BRAIN and ILEUM appear on knee studies).
