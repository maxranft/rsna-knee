"""Replace a label's prediction by a weighted mean of ranks of several outputs, and score it like a run.

    python scripts/rank_mix.py runs/ens-20261005-222134 --mix '{"Synovitis": {"Synovitis": 1, "Effusion": 1}}' --log "..."
    -> runs/mix-<timestamp>/{oof.csv, metrics.json, args.json}; package_heads.py runs/mix-<id> ships the source's heads
       plus rank_mix.json, which the submission kernel applies over the whole test set after averaging the heads.

Why (2026-10-05): gold Synovitis (47% of the 58 gold studies) is not what reports call synovitis (20%). The model's own
Synovitis output ranks gold Synovitis at AUC ~0.53, its Effusion output at ~0.74 ("effusion-synovitis"). AUC only needs
ranks, so mixing ranks is not calibration. Other labels are untouched.
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee import results as R  # noqa: E402


def apply_mix(pred, mix):
    """pred: DataFrame with the label columns (one row per study, the whole pool) -> a copy with mixed columns."""
    rk, out = pred[R.LABELS].rank(pct=True), pred.copy()
    for target, w in mix.items():
        out[target] = sum(v * rk[src] for src, v in w.items()) / sum(w.values())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="a run or ens-* folder with oof.csv")
    p.add_argument("--mix", required=True, help='JSON {label: {output: weight, ...}}')
    p.add_argument("--log", default="")
    args = p.parse_args()
    src = Path(args.run)
    mix = json.loads(args.mix)
    assert all(t in R.LABELS and all(s in R.LABELS for s in w) for t, w in mix.items()), mix
    m0 = json.loads((src / "metrics.json").read_text())
    oof = pd.read_csv(src / "oof.csv").set_index("StudyInstanceUID")
    new = apply_mix(oof, mix)

    out = ROOT / "runs" / f"mix-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    new.reset_index().to_csv(out / "oof.csv", index=False)
    run_args = dict(m0["args"]) | {"log": args.log or f"rank mix {args.mix} on {src.name}", "source": src.name, "mix": mix}
    run_args.pop("members", None)
    (out / "args.json").write_text(json.dumps(run_args, indent=1))

    folds, train, report = R._truth(pd.DataFrame({"labels_file": [m0["args"]["labels"]],
                                                  "label_map": [m0["args"]["label_map"]]}, index=["x"]), "x")
    ids = list(new.index)
    gold_ids = [u for u in ids if folds.loc[u, "is_gold"]]
    rep_ids = [u for u in ids if not folds.loc[u, "is_gold"]]

    def per_label(ids, truth):
        return {l: round(float(roc_auc_score(truth.loc[ids, l], new.loc[ids, l])), 4) for l in R.LABELS
                if 0 < truth.loc[ids, l].sum() < len(ids)}

    gold, rep = per_label(gold_ids, train[R.LABELS]), per_label(rep_ids, report)
    m = {"run": out.name, "args": run_args, "device": "-", "minutes": m0.get("minutes", 0),
         "gold_macro": round(float(np.mean(list(gold.values()))), 4), "gold": gold,
         "report_macro": round(float(np.mean(list(rep.values()))), 4), "report": rep,
         "n_gold": len(gold_ids), "n_report": len(rep_ids)}
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: m[k] for k in ("run", "gold_macro", "report_macro")}))
    print(f"  source {src.name}: gold {m0['gold_macro']}  report {m0['report_macro']}")
    for t in mix:
        print(f"  {t}: gold {m0['gold'].get(t)} -> {gold.get(t)}, report {m0['report'].get(t)} -> {rep.get(t)}")
    if args.log:
        weakest = ", ".join(f"{k} {v:.3f}" for k, v in sorted(gold.items(), key=lambda kv: kv[1])[:3])
        with open(ROOT / "experiments.md", "a") as fh:
            fh.write(f"| {date.today()} | {out.name} | S1/S3 | {args.log} | {m['gold_macro']} | {m['report_macro']} | "
                     f"{weakest} | same heads as {src.name} | source {src.name}, mix {json.dumps(mix)} |\n")


if __name__ == "__main__":
    main()
