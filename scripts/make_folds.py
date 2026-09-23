"""Fixed 5-fold split over all training studies -> meta/folds.csv (committed; never regenerate).

Gold studies (the 58 with labels) are split with multilabel stratification so every fold gets
positives of each label; report-only studies are split at random. CV scores are computed on the
pooled out-of-fold predictions for the gold studies, not per fold (about 12 gold studies per fold is too few).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
SEED, K = 42, 5

train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c != "Report")
labels = [c for c in train.columns if c != "StudyInstanceUID"]
train["is_gold"] = train[labels].notna().all(axis=1)
train["fold"] = -1

gold = train.index[train.is_gold]
mskf = MultilabelStratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
for f, (_, va) in enumerate(mskf.split(gold, train.loc[gold, labels].values)):
    train.loc[gold[va], "fold"] = f

rest = train.index[~train.is_gold]
for f, (_, va) in enumerate(KFold(K, shuffle=True, random_state=SEED).split(rest)):
    train.loc[rest[va], "fold"] = f

assert (train.fold >= 0).all()
out = ROOT / "meta/folds.csv"
out.parent.mkdir(exist_ok=True)
train[["StudyInstanceUID", "fold", "is_gold"]].to_csv(out, index=False)
print(train.groupby("fold").agg(n=("is_gold", "size"), gold=("is_gold", "sum")).T.to_string())
g = train[train.is_gold]
print("gold positives per fold:\n", g.groupby("fold")[labels].sum().astype(int).to_string())
