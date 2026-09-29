"""Knee side (L/R) per study from the frozen-feature cache, for canonicalising medial vs lateral.

Why: image orientation is uniform (coronal/axial rows run along patient +X, sagittal slices are sorted along -X), so a
left knee shows its medial side where a right knee shows its lateral side. Medial/Lateral Meniscus and OA depend on it.

Classifier targets: the DICOM Laterality tag only (preprocess slots_shard*.csv, ~50% of studies). The report side
(labeller aux_right_knee / aux_left_knee) is only a cross-check: it agrees with the tag ~97%, the image probe ~99%.
Final side = tag if present, else the image probe, which is also what the submission can do. The probe sees only
left/right-antisymmetric features, which flip sign when the knee is mirrored:
  - coronal and axial: slice-mean 4x4 grid, column c minus column 3-c
  - sagittal and T1: mean of the first half of slices minus the second half
Logistic regression, 5-fold CV on the fixed folds. Writes meta/laterality.csv:
StudyInstanceUID, tag, report, p_left (out-of-fold), side (final L/R), side_source (tag/image), t1_plane (the T1 slot
is coronal or sagittal, which decides how it is mirrored).

    python scripts/laterality.py --feats data/features/dinov2s
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ["sag_fs", "cor_fs", "ax_fs", "t1"]
GRID = 4


def known_side():
    slots = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "data/pp-slots/slots_shard*.csv")))])
    tag = slots.groupby("StudyInstanceUID").laterality.apply(
        lambda x: "".join(sorted(set(x.dropna().astype(str).str.strip().str.upper().str[0]) - {"", "N"})))
    r = pd.read_csv(ROOT / "data/labeller/all-v2/report_scores_all.csv").set_index("StudyInstanceUID")
    rep = pd.Series(np.where(r.aux_right_knee > 0.5, "R", np.where(r.aux_left_knee > 0.5, "L", "")), index=r.index)
    j = pd.DataFrame({"tag": tag, "report": rep}).fillna("")
    j["tag"] = j.tag.where(j.tag.isin(["L", "R"]), "")  # "LR" / "B" conflicts count as missing
    series = pd.read_csv(ROOT / "data/raw/train_series.csv").set_index("SeriesInstanceUID").Anatomical_Plane
    t1 = slots[slots.slot == "t1"].set_index("StudyInstanceUID").SeriesInstanceUID
    j["t1_plane"] = t1.map(series).reindex(j.index).fillna("")
    return j


def study_features(feats_dir):
    """-> DataFrame index=StudyInstanceUID, antisymmetric features per slot (zeros if the slot is missing)."""
    index = pd.read_csv(feats_dir / "index.csv")
    out = {}
    for shard, g in index.groupby("shard"):
        a = np.load(feats_dir / f"feats_shard{shard}.npy", mmap_mode="r")
        for r in g.sort_values("start").itertuples():
            x = np.asarray(a[r.start:r.start + r.length], dtype=np.float32)  # [D, 17, C]
            if r.slot in ("cor_fs", "ax_fs"):
                grid = x[:, 1:].mean(0).reshape(GRID, GRID, -1)  # [row, col, C]
                f = np.concatenate([grid[:, c] - grid[:, GRID - 1 - c] for c in range(GRID // 2)]).ravel()
            else:
                h = len(x) // 2
                f = (x[:h].mean(0) - x[-h:].mean(0)).ravel()
            out.setdefault(r.StudyInstanceUID, {})[r.slot] = f
    dims = {s: next(len(v[s]) for v in out.values() if s in v) for s in SLOTS}
    rows = {u: np.concatenate([v.get(s, np.zeros(dims[s], np.float32)) for s in SLOTS]) for u, v in out.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", default="data/features/dinov2s")
    p.add_argument("--C", type=float, default=0.01)
    args = p.parse_args()

    side = known_side()
    X = study_features(Path(args.feats))
    folds = pd.read_csv(ROOT / "meta/folds.csv").set_index("StudyInstanceUID")
    side = side.reindex(X.index).fillna("")
    tagged = (side.tag != "").values
    y = (side.tag == "L").astype(int).values
    print(f"studies {len(X)}, tagged {tagged.sum()}, report side {(side.report != '').sum()}, features {X.shape[1]}")

    p_left = pd.Series(np.nan, index=X.index)
    fold = folds.loc[X.index, "fold"].values
    for f in range(5):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=2000))
        clf.fit(X.values[tagged & (fold != f)], y[tagged & (fold != f)])
        p_left[fold == f] = clf.predict_proba(X.values[fold == f])[:, 1]
    pred = np.where(p_left > 0.5, "L", "R")
    print(f"image probe vs tag (OOF): AUC {roc_auc_score(y[tagged], p_left[tagged]):.4f}  "
          f"accuracy {(pred[tagged] == side.tag.values[tagged]).mean():.4f}")
    rep = (side.report != "").values
    for name, m in [("tagged", tagged & rep), ("untagged", ~tagged & rep)]:
        print(f"  report side vs image probe, {name} studies n={m.sum()}: agreement {(pred[m] == side.report.values[m]).mean():.4f}")
    m = tagged & rep
    print(f"  report side vs tag n={m.sum()}: agreement {(side.report.values[m] == side.tag.values[m]).mean():.4f}")

    side["p_left"] = p_left.round(4)
    side["side"] = np.where(tagged, side.tag, pred)
    side["side_source"] = np.where(tagged, "tag", "image")
    side.index.name = "StudyInstanceUID"
    side.to_csv(ROOT / "meta/laterality.csv")
    print("wrote meta/laterality.csv", side.side.value_counts().to_dict(), side.side_source.value_counts().to_dict())


if __name__ == "__main__":
    main()
