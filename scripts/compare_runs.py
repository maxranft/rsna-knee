"""Compare runs side by side: per-label gold and report AUC, plus a paired bootstrap of the macro-AUC difference.

    python scripts/compare_runs.py runs/<baseline> runs/<candidate> [more runs...]

Every later run is compared with the first. The bootstrap resamples studies (same resample for both runs), so it
measures how much of the difference could be study-sampling noise: with 58 gold studies the 90% interval is wide,
which is why gold differences below ~0.02 are treated as noise. Labels that have no positive (or no negative) in a
resample are skipped for that resample.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def macro(y, p, idx):
    vals = [roc_auc_score(y[idx, j], p[idx, j]) for j in range(y.shape[1]) if 0 < y[idx, j].sum() < len(idx)]
    return np.mean(vals) if vals else np.nan


def paired_boot(y, pa, pb, n=500, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        d.append(macro(y, pb, idx) - macro(y, pa, idx))
    d = np.array(d)
    return np.nanpercentile(d, [5, 95]), np.nanmean(d > 0)


def main():
    runs = [Path(r) for r in sys.argv[1:]]
    ms = [json.loads((r / "metrics.json").read_text()) for r in runs]
    oofs = [pd.read_csv(r / "oof.csv").set_index("StudyInstanceUID") for r in runs]
    folds = pd.read_csv(ROOT / "meta/folds.csv").set_index("StudyInstanceUID")
    train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c != "Report").set_index("StudyInstanceUID")
    args0 = ms[0]["args"]
    scores = pd.read_csv(ROOT / args0["labels"]).set_index("StudyInstanceUID")
    lm = {l: l for l in LABELS} | json.loads(args0["label_map"])
    report = (scores[[lm[l] for l in LABELS]].set_axis(LABELS, axis=1) > 0.5).astype(int)

    common = sorted(set.intersection(*(set(o.index) for o in oofs)))
    gold = [u for u in common if folds.loc[u, "is_gold"]]
    rep = [u for u in common if not folds.loc[u, "is_gold"]]
    names = [r.name for r in runs]
    print(f"studies: gold {len(gold)}, report {len(rep)}")
    rows = []
    for l in LABELS:
        row = {"label": l}
        for n, m in zip(names, ms):
            row[f"gold {n[-6:]}"] = m["gold"].get(l, np.nan)
            row[f"rep {n[-6:]}"] = m["report"].get(l, np.nan)
        rows.append(row)
    t = pd.DataFrame(rows).set_index("label")
    t.loc["MACRO"] = t.mean()
    print(t.round(3).to_string())

    for name, ids, truth in [("gold", gold, train.loc[gold, LABELS].values.astype(int)),
                             ("report", rep, report.loc[rep].values)]:
        pa = oofs[0].loc[ids, LABELS].values
        for n, o in zip(names[1:], oofs[1:]):
            pb = o.loc[ids, LABELS].values
            (lo, hi), p_better = paired_boot(truth, pa, pb, n=300 if name == "report" else 1000)
            dm = macro(truth, pb, np.arange(len(ids))) - macro(truth, pa, np.arange(len(ids)))
            print(f"{name:6s} {n} vs {names[0]}: diff {dm:+.4f}  90% CI [{lo:+.4f}, {hi:+.4f}]  P(better) {p_better:.2f}")


if __name__ == "__main__":
    main()
