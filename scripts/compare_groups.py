"""Compare two configs trained with several seeds each (the reference vs a candidate), accounting for seed noise.

    python scripts/compare_groups.py --a runs/<ref seed42> runs/<ref seed43> --b runs/<cand seed42> runs/<cand seed43>

Reports (1) each run's macro AUC and the seed mean per group, (2) same-size rank-average ensembles of each group (first
n runs, n = the smaller group) compared with a paired bootstrap over studies, which is the comparison that matters for
the submission, and (3) the per-label change of the seed means (report labels). One seed per config moves report macro
by ~0.004 and gold by ~0.010 (2026-09-29), so single-run differences below that are noise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee import dashboard as D  # noqa: E402
from knee import results as R  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", nargs="+", required=True, help="reference runs (one per seed)")
    p.add_argument("--b", nargs="+", required=True, help="candidate runs (one per seed)")
    args = p.parse_args()
    A, B = [Path(r) for r in args.a], [Path(r) for r in args.b]
    m0 = json.loads((A[0] / "metrics.json").read_text())
    folds, gold, report = D.truth(m0["args"]["labels"], m0["args"]["label_map"])
    oof = {r: pd.read_csv(r / "oof.csv").set_index("StudyInstanceUID") for r in A + B}
    ids = sorted(set.intersection(*(set(o.index) for o in oof.values())))
    g = [u for u in ids if folds.loc[u, "is_gold"]]
    p_ = [u for u in ids if not folds.loc[u, "is_gold"]]
    Yg, Yr = gold.loc[g].values.astype(int), report.loc[p_].values

    def scores(o):
        return (D.macro(Yg, o.loc[g, R.LABELS].values, np.arange(len(g))),
                D.macro(Yr, o.loc[p_, R.LABELS].values, np.arange(len(p_))))

    print(f"{'run':24s} {'seed':>4s} {'gold':>7s} {'report':>7s}")
    means = {}
    for name, grp in [("A", A), ("B", B)]:
        s = []
        for r in grp:
            seed = json.loads((r / "metrics.json").read_text())["args"].get("seed")
            s.append(scores(oof[r]))
            print(f"{name} {r.name:22s} {seed:>4} {s[-1][0]:7.4f} {s[-1][1]:7.4f}")
        s = np.array(s)
        means[name] = s.mean(0)
        print(f"{name} seed mean {'':17s} {s[:, 0].mean():7.4f} {s[:, 1].mean():7.4f}   (range gold {np.ptp(s[:, 0]):.4f}, report {np.ptp(s[:, 1]):.4f})")
    print(f"seed-mean difference B - A: gold {means['B'][0] - means['A'][0]:+.4f}  report {means['B'][1] - means['A'][1]:+.4f}")

    n = min(len(A), len(B))
    ens = lambda grp: sum(oof[r].loc[ids, R.LABELS].rank(pct=True) for r in grp[:n]) / n
    ea, eb = ens(A), ens(B)
    print(f"\n{n}-run ensembles, B vs A (paired bootstrap, 90% interval):")
    for name, sub, Y, nb in [("gold", g, Yg, 1000), ("report", p_, Yr, 300)]:
        d = D.boot_diff(Y, ea.loc[sub].values, eb.loc[sub].values, nb)
        full = np.arange(len(sub))
        print(f"  {name:6s} A {D.macro(Y, ea.loc[sub].values, full):.4f}  B {D.macro(Y, eb.loc[sub].values, full):.4f}  "
              f"diff {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  P(better) {d['p_better']:.2f}")

    per = {}
    for l_i, l in enumerate(R.LABELS):
        va = np.mean([D.auc(Yr[:, l_i], oof[r].loc[p_, l].values) for r in A])
        vb = np.mean([D.auc(Yr[:, l_i], oof[r].loc[p_, l].values) for r in B])
        per[l] = (va, vb, vb - va)
    t = pd.DataFrame(per, index=["A", "B", "B - A"]).T.sort_values("B - A")
    print("\nper label, report AUC, seed means:")
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
