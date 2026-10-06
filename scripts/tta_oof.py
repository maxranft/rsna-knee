"""Leave-one-slot-out test-time augmentation, scored out of fold. For every study, each run's fold head predicts the full
study and the study with each series slot removed (the heads train with slot dropout), and the mean is kept.

    python scripts/tta_oof.py runs/ens-20261005-222134             # members of an ensemble (or several run folders)
    -> runs/<id>/oof_tta_slot.csv per member (skipped when present), then the rank-averaged ensemble with and without
       TTA, compared like auto_improve.py (paired bootstrap on report and gold). CPU, about 1.5 min per run.

The submission kernel does the same when the heads dataset has tta.json (scripts/package_heads.py --tta).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee import dashboard as D  # noqa: E402
from knee import results as R  # noqa: E402
from knee.head import Head, Studies, collate, slot_variants  # noqa: E402

L = R.LABELS


def tta_run(run, index, feats, folds):
    rows = []
    cfg = json.loads((run / "config.json").read_text())
    for f in range(5):
        h = Head.from_config(cfg)
        h.load_state_dict(torch.load(run / f"head_fold{f}.pt", map_location="cpu"))
        h.eval()
        va = folds.StudyInstanceUID[folds.fold == f].tolist()
        ds = Studies(va, index, feats, {u: np.zeros(len(L), np.float32) for u in va})
        with torch.no_grad():
            for i, u in enumerate(va):
                x, slot, depth, _ = ds[i]
                xb, sb, db, pb = slot_variants(x, slot, depth)
                rows.append([u, *torch.sigmoid(h(xb, sb, db, pb)).mean(0).numpy(), f])
    return pd.DataFrame(rows, columns=["StudyInstanceUID"] + L + ["fold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    runs = []
    for a in args.runs:
        d = Path(a)
        runs += [ROOT / "runs" / m for m in json.loads((d / "args.json").read_text())["members"]] if d.name.startswith("ens-") else [d]
    cfg = json.loads((runs[0] / "config.json").read_text())
    fd = ROOT / cfg["feats"]
    index = pd.read_csv(fd / "index.csv")
    feats = {int(f.stem.split("shard")[1]): str(f) for f in sorted(fd.glob("feats_shard*.npy"))}
    folds = pd.read_csv(ROOT / "meta/folds.csv")
    folds = folds[folds.StudyInstanceUID.isin(set(index.StudyInstanceUID))]
    for r in runs:
        if not (r / "oof_tta_slot.csv").exists():
            tta_run(r, index, feats, folds).to_csv(r / "oof_tta_slot.csv", index=False)
            print(f"{r.name}: done", flush=True)

    m0 = json.loads((runs[0] / "metrics.json").read_text())
    fo, gold, report = D.truth(m0["args"]["labels"], m0["args"]["label_map"])
    plain = [pd.read_csv(r / "oof.csv").set_index("StudyInstanceUID") for r in runs]
    tta = [pd.read_csv(r / "oof_tta_slot.csv").set_index("StudyInstanceUID") for r in runs]
    ids = sorted(set.intersection(*(set(o.index) for o in plain + tta)))
    g = [u for u in ids if fo.loc[u, "is_gold"]]
    p = [u for u in ids if not fo.loc[u, "is_gold"]]
    ens = lambda oofs: sum(o.loc[ids, L].rank(pct=True) for o in oofs) / len(oofs)  # noqa: E731
    A, B = ens(plain), ens(tta)
    rep = D.boot_diff(report.loc[p].values, A.loc[p].values, B.loc[p].values, 300)
    gld = D.boot_diff(gold.loc[g].values.astype(int), A.loc[g].values, B.loc[g].values, 1000)
    macro = lambda P, ids, Y: D.macro(Y.loc[ids].values.astype(int), P.loc[ids].values, np.arange(len(ids)))  # noqa: E731
    print(f"{len(runs)}-run ensemble  plain: report {macro(A, p, report):.4f} gold {macro(A, g, gold):.4f}   "
          f"TTA: report {macro(B, p, report):.4f} gold {macro(B, g, gold):.4f}")
    print(f"TTA - plain: report {rep['diff']:+.4f} (90% CI {rep['lo']:+.4f}..{rep['hi']:+.4f}), "
          f"gold {gld['diff']:+.4f} (CI {gld['lo']:+.4f}..{gld['hi']:+.4f}, P(better) {gld['p_better']:.2f})")


if __name__ == "__main__":
    main()
