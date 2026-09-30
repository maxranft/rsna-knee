"""Rank-average the out-of-fold predictions of several runs and score the result like a run.

    python scripts/ensemble_oof.py runs/<a> runs/<b> [runs/<c> ...] --log "3-seed ensemble of PCA-128"

AUC is rank-based, so predictions are converted to ranks per label (pooled over all OOF studies) before averaging;
no calibration. Writes runs/ens-<timestamp>/{oof.csv, metrics.json, args.json} so the results viewer, compare_runs.py
and side_split.py treat it like any other run, and appends a row to experiments.md with --log.
At inference an ensemble costs one extra head pass per member; the frozen encoder, which dominates runtime, runs once.
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee import results as R  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--log", default="")
    args = p.parse_args()
    dirs = [Path(r) for r in args.runs]
    ms = [json.loads((d / "metrics.json").read_text()) for d in dirs]
    oofs = [pd.read_csv(d / "oof.csv").set_index("StudyInstanceUID") for d in dirs]
    common = sorted(set.intersection(*(set(o.index) for o in oofs)))
    ranks = [o.loc[common, R.LABELS].rank(pct=True) for o in oofs]
    ens = sum(ranks) / len(ranks)
    ens["fold"] = oofs[0].loc[common, "fold"]

    out = ROOT / "runs" / f"ens-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    ens.reset_index().to_csv(out / "oof.csv", index=False)
    base_args = dict(ms[0]["args"])
    run_args = base_args | {"log": args.log or f"rank-average of {', '.join(d.name for d in dirs)}",
                            "members": [d.name for d in dirs]}
    (out / "args.json").write_text(json.dumps(run_args, indent=1))

    folds, train, report = R._truth(pd.DataFrame({"labels_file": [base_args["labels"]],
                                                  "label_map": [base_args["label_map"]]}, index=["x"]), "x")
    gold_ids = [u for u in common if folds.loc[u, "is_gold"]]
    rep_ids = [u for u in common if not folds.loc[u, "is_gold"]]
    from sklearn.metrics import roc_auc_score

    def per_label(ids, truth):
        return {l: round(float(roc_auc_score(truth.loc[ids, l], ens.loc[ids, l])), 4) for l in R.LABELS
                if 0 < truth.loc[ids, l].sum() < len(ids)}

    gold, rep = per_label(gold_ids, train[R.LABELS]), per_label(rep_ids, report)
    m = {"run": out.name, "args": run_args, "device": "-", "minutes": round(sum(x["minutes"] for x in ms), 1),
         "gold_macro": round(float(np.mean(list(gold.values()))), 4), "gold": gold,
         "report_macro": round(float(np.mean(list(rep.values()))), 4), "report": rep,
         "n_gold": len(gold_ids), "n_report": len(rep_ids)}
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: m[k] for k in ("run", "gold_macro", "report_macro")}))
    for d, x in zip(dirs, ms):
        print(f"  member {d.name}: gold {x['gold_macro']}  report {x['report_macro']}")
    if args.log:
        weakest = ", ".join(f"{k} {v:.3f}" for k, v in sorted(gold.items(), key=lambda kv: kv[1])[:3])
        with open(ROOT / "experiments.md", "a") as fh:
            fh.write(f"| {date.today()} | {out.name} | S3/S4 | {args.log} | {m['gold_macro']} | {m['report_macro']} | "
                     f"{weakest} | {len(dirs)} heads at inference | members {', '.join(d.name for d in dirs)} |\n")


if __name__ == "__main__":
    main()
