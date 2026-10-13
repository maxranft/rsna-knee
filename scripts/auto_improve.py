"""Recursive improvement loop for the study head (S3): try one change at a time on the current best config, 3 seeds
each, and keep a change only when it clearly helps. A kept change becomes the base for every later candidate.

    python scripts/auto_improve.py --plan                    # base config, candidates and their state; trains nothing
    python scripts/auto_improve.py --until 2026-10-13        # run pending candidates, then extra seeds of the base
    python scripts/auto_improve.py --add dinov2b "DINOv2-B features" --feats data/features/dinov2b-pca128

The rule from CLAUDE.md, made automatic: compare 3-run rank-average ensembles (seeds 42, 43, 44, fixed folds). Adopt
when report macro AUC improves with its paired-bootstrap 90% interval above zero AND gold macro AUC drops by no more
than 0.02 (gold differences below that are noise); otherwise reject. Candidates must keep the training labels fixed:
the report yardstick is computed from the base's labels, so label changes are judged separately, on gold.
Each verdict also states Cliff's delta between the candidate's seeds and the base's (share of seed pairs the candidate
wins minus the share it loses), as a seed-level check next to the ensemble comparison. It is not part of the rule.

Every run is logged by train_head.py in experiments.md, and the verdict goes into the first seed's Notes as
**Adopted** / **Rejected** (the dashboard reads those tags). State lives in runs/auto/: queue.json (candidates, read
before each one so new ones can be added while it runs) and state.json (base, verdicts). Finished runs are found by
their --log text and never retrained, so the loop resumes where it stopped. It never repackages heads or submits:
adopting the result for Kaggle stays a manual decision.
"""
import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee import dashboard as D  # noqa: E402
from knee import results as R  # noqa: E402

AUTO = ROOT / "runs/auto"
SEEDS = (42, 43, 44)
FIXED = ["--workers", "0", "--no-open"]
DEFAULTS = {"--lr": "3e-4", "--wd": "0.05", "--bs": "16", "--d": "256", "--layers": "1", "--drop": "0.1",
            "--epochs": "12", "--slot-drop": "0.1", "--ema": "0", "--label-smooth": "0", "--aux": "none"}  # train_head.py defaults
# on/off flags (train_head.py store_true) are given the value true in queue.json and passed without a value
# best single-model config (experiments.md, 2026-09-30) and its seed 42/43/44 runs
BASE = {"--feats": "data/features/dinov2s-pca128", "--labels": "data/labeller/all-v2/report_scores_all.csv",
        "--label-map": '{"Effusion": "Effusion@moderate"}', "--drop": "0.3", "--epochs": "16", "--slot-drop": "0.3"}
BASE_RUNS = ["20260930-055449", "20260930-061338", "20260930-063255"]
BASE_DESC = "best config (PCA-128, 16 ep, dropout 0.3, slot-drop 0.3)"
CANDIDATES = [  # most promising first; each is one change to whatever the base is when its turn comes
    ("ema", {"--ema": "0.999"}),
    ("wd0.1", {"--wd": "0.1"}),
    ("bs32", {"--bs": "32"}),
    ("lr2e-4", {"--lr": "2e-4"}),
    ("drop0.4", {"--drop": "0.4"}),
    ("ep20", {"--epochs": "20"}),
    ("smooth0.05", {"--label-smooth": "0.05"}),
    ("wd0.01", {"--wd": "0.01"}),
]


def load_state():
    AUTO.mkdir(parents=True, exist_ok=True)
    if not (AUTO / "queue.json").exists():
        (AUTO / "queue.json").write_text(json.dumps([{"key": k, "flags": f} for k, f in CANDIDATES], indent=1))
    if not (AUTO / "state.json").exists():
        (AUTO / "state.json").write_text(json.dumps({"base": BASE, "base_runs": BASE_RUNS, "base_desc": BASE_DESC,
                                                     "done": {}}, indent=1))
    return json.loads((AUTO / "queue.json").read_text()), json.loads((AUTO / "state.json").read_text())


def save_state(state):
    (AUTO / "state.json").write_text(json.dumps(state, indent=1))


def describe(flags, base):
    return ", ".join(f"{k.lstrip('-')}" + (" on" if v is True else f" {v}") + f" (was {base.get(k, DEFAULTS.get(k, 'off'))})"
                     for k, v in flags.items())


def find_run(log):
    """The finished run trained with this exact --log text, if any (newest first)."""
    for d in sorted((ROOT / "runs").glob("2*"), reverse=True):
        a = d / "args.json"
        if a.exists() and (d / "metrics.json").exists() and json.loads(a.read_text()).get("log") == log:
            return d
    return None


def train(flags, seed, log):
    """One train_head.py run (reused if it already finished) -> its run folder."""
    done = find_run(log)
    if done:
        print(f"  seed {seed}: already done ({done.name})", flush=True)
        return done
    args = [x for k, v in flags.items() for x in ((k,) if v is True else (k, v))]
    print(f"\n=== {datetime.now():%m-%d %H:%M}  seed {seed}: {log}", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/train_head.py"), *args, *FIXED, "--seed", str(seed),
                        "--log", log], cwd=ROOT)
    run = find_run(log)
    if r.returncode != 0 or run is None:
        raise RuntimeError(f"training failed (exit {r.returncode}): {log}")
    return run


def judge(a_runs, b_runs):
    """3-run rank-average ensembles, B vs A, paired bootstrap over studies (as scripts/compare_groups.py)."""
    a_runs, b_runs = [ROOT / "runs" / r for r in a_runs], [ROOT / "runs" / r for r in b_runs]
    m0 = json.loads((a_runs[0] / "metrics.json").read_text())
    folds, gold, report = D.truth(m0["args"]["labels"], m0["args"]["label_map"])
    oof = {r: pd.read_csv(r / "oof.csv").set_index("StudyInstanceUID") for r in a_runs + b_runs}
    ids = sorted(set.intersection(*(set(o.index) for o in oof.values())))
    g = [u for u in ids if folds.loc[u, "is_gold"]]
    p = [u for u in ids if not folds.loc[u, "is_gold"]]
    ens = lambda runs: sum(oof[r].loc[ids, R.LABELS].rank(pct=True) for r in runs) / len(runs)  # noqa: E731
    ea, eb = ens(a_runs), ens(b_runs)
    rep = D.boot_diff(report.loc[p].values, ea.loc[p].values, eb.loc[p].values, 300)
    gld = D.boot_diff(gold.loc[g].values.astype(int), ea.loc[g].values, eb.loc[g].values, 1000)
    return rep["lo"] > 0 and gld["diff"] > -0.02, rep, gld


def seed_cliff(a_runs, b_runs):
    """Cliff's delta of B's single-seed macro AUCs over A's: {"report": d, "gold": d}; None if a run is gone."""
    try:
        m = {r: json.loads((ROOT / "runs" / r / "metrics.json").read_text()) for r in [*a_runs, *b_runs]}
    except FileNotFoundError:
        return None
    return {k: D.cliff([m[r][f"{k}_macro"] for r in b_runs], [m[r][f"{k}_macro"] for r in a_runs]) for k in ("report", "gold")}


def note_verdict(run_ids, verdict, text):
    """Append the verdict to the first seed's Notes in experiments.md and point the other seeds at it."""
    md = ROOT / "experiments.md"
    lines = md.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        for j, rid in enumerate(run_ids):
            if f"| {rid} |" in line and line.rstrip().endswith("|"):
                add = f" **{verdict}** (auto) {text}" if j == 0 else f" Seed of the auto check; verdict on {run_ids[0]}"
                lines[i] = line.rstrip()[:-1].rstrip() + add + " |\n"
    md.write_text("".join(lines))


def run_candidate(c, state):
    base, key = state["base"], c["key"]
    flags = {**base, **c["flags"]}
    desc = describe(c["flags"], base)
    runs = [train(flags, s, f"auto: {desc}, seed {s}; vs {state['base_desc']}").name for s in SEEDS]
    adopt, rep, gld = judge(state["base_runs"], runs)
    verdict = "Adopted" if adopt else "Rejected"
    cl = seed_cliff(state["base_runs"], runs)
    text = (f"(3 seeds vs {state['base_desc']}, runs {', '.join(state['base_runs'])}): 3-run ensemble report "
            f"{rep['diff']:+.4f} (90% CI {rep['lo']:+.4f}..{rep['hi']:+.4f}), gold {gld['diff']:+.4f} "
            f"(P(better) {gld['p_better']:.2f}). Seed by seed, Cliff's delta: report {cl['report']:+.2f}, gold "
            f"{cl['gold']:+.2f}. Rule: report CI above 0 and gold no worse than -0.02")
    note_verdict(runs, verdict, text)
    state["done"][key] = {"verdict": verdict, "flags": c["flags"], "desc": desc, "runs": runs, "base_runs": state["base_runs"],
                          "report": rep, "gold": gld, "cliff": cl, "at": datetime.now().isoformat(timespec="seconds")}
    if adopt:
        state["base"], state["base_runs"] = flags, runs
        state["base_desc"] = f"{state['base_desc']} + {desc}"
    save_state(state)
    print(f"\n*** {key}: {verdict.upper()}  report {rep['diff']:+.4f} [{rep['lo']:+.4f}, {rep['hi']:+.4f}]  "
          f"gold {gld['diff']:+.4f}  Cliff's delta report {cl['report']:+.2f}, gold {cl['gold']:+.2f}", flush=True)


def fill_seeds(state, until, n):
    """After the candidates: more seeds of the (possibly improved) base, as ensemble members."""
    used = {json.loads((d / "args.json").read_text()).get("seed") for d in (ROOT / "runs").glob("2*") if (d / "args.json").exists()}
    seed = max([s for s in used if isinstance(s, int)] + [55]) + 1
    for _ in range(n):
        if date.today() > until:
            return
        train(state["base"], seed, f"auto: ensemble member, {state['base_desc']}, seed {seed}")
        seed += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--until", type=date.fromisoformat, default=date(2026, 10, 13), help="start nothing new after this day")
    ap.add_argument("--fill", type=int, default=6, help="extra base-config seeds once the candidates are done")
    ap.add_argument("--add", nargs=2, metavar=("KEY", "DESCRIPTION"), help="append a candidate; its flags follow")
    args, extra = ap.parse_known_args()
    queue, state = load_state()
    if args.add:
        flags, i = {}, 0
        while i < len(extra):  # "--flag value", or a bare "--flag" (an on/off switch)
            if i + 1 < len(extra) and not extra[i + 1].startswith("--"):
                flags[extra[i]], i = extra[i + 1], i + 2
            else:
                flags[extra[i]], i = True, i + 1
        queue.append({"key": args.add[0], "flags": flags, "note": args.add[1]})
        (AUTO / "queue.json").write_text(json.dumps(queue, indent=1))
        print(f"added {args.add[0]}: {flags}")
        return
    print(f"base: {state['base_desc']}\n  runs {', '.join(state['base_runs'])}")
    for c in queue:
        d = state["done"].get(c["key"])
        cl = d and (d.get("cliff") or seed_cliff(d["base_runs"], d["runs"]))  # verdicts from before it was stored
        print(f"  {c['key']:12s} {describe(c['flags'], state['base']):45s} "
              + (f"{d['verdict']} (report {d['report']['diff']:+.4f}, gold {d['gold']['diff']:+.4f}"
                 + (f"; Cliff's delta report {cl['report']:+.2f}, gold {cl['gold']:+.2f})" if cl else ")") if d else "pending"))
    if args.plan:
        return
    while True:
        queue, state = load_state()  # re-read: candidates may have been added meanwhile
        pending = [c for c in queue if c["key"] not in state["done"]]
        if not pending:
            break
        if date.today() > args.until:
            print(f"past {args.until}: stopping before {pending[0]['key']}")
            return
        run_candidate(pending[0], state)
    fill_seeds(state, args.until, args.fill)
    print(f"\ndone. base: {state['base_desc']}  runs {', '.join(state['base_runs'])}")


if __name__ == "__main__":
    main()
