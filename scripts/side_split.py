"""Per-label report AUC split by knee side (meta/laterality.csv). If a model confuses medial with lateral, the
Medial/Lateral labels score lower on the minority side (left) than on the majority side (right).

    python scripts/side_split.py runs/<id> [runs/<id> ...]
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ML = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]
OTHER = ["ACL", "MCL", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

lat = pd.read_csv(ROOT / "meta/laterality.csv").set_index("StudyInstanceUID").side
folds = pd.read_csv(ROOT / "meta/folds.csv").set_index("StudyInstanceUID")
for r in map(Path, sys.argv[1:]):
    m = json.loads((r / "metrics.json").read_text())
    scores = pd.read_csv(ROOT / m["args"]["labels"]).set_index("StudyInstanceUID")
    lm = json.loads(m["args"]["label_map"])
    oof = pd.read_csv(r / "oof.csv").set_index("StudyInstanceUID")
    oof = oof[~folds.loc[oof.index, "is_gold"].values]
    rows = {}
    for side in ["R", "L"]:
        ids = oof.index[lat.loc[oof.index].values == side]
        y = scores.loc[ids]
        rows[side] = {l: roc_auc_score(y[lm.get(l, l)] > 0.5, oof.loc[ids, l]) for l in ML + OTHER}
    t = pd.DataFrame(rows)
    t["L-R"] = t.L - t.R
    print(f"== {r.name}  (report AUC by side; n R={sum(lat.loc[oof.index] == 'R')}, L={sum(lat.loc[oof.index] == 'L')})")
    print(t.round(3).to_string())
    print(f"mean L-R: medial/lateral labels {t.loc[ML, 'L-R'].mean():+.3f}   other labels {t.loc[OTHER, 'L-R'].mean():+.3f}")
