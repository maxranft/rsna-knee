"""Results dashboard -> runs/dashboard.html: every run with its verdict and learning-curve delta and gamma, configs
compared across seeds (Cliff's delta), label betas across configs, the reference model's per-label efficacy (AUC with
90% intervals, ROC curves, label ceiling, knee side), the data, efficiency, and the run in progress. The 3D view draws
delta, gamma and the betas over accuracy, AUC and time: each run's path through (AUC, accuracy) by epoch (path3d) and
the plane accuracy = alpha + beta_auc * AUC + beta_time * minutes across configs (plane_section), with the minutes
taken from each run's epoch times so that pauses do not count (train_minutes).

Metrics and aggregates only: train.csv is always read without its Report column, and no image is touched.
Built by scripts/dashboard.py (and at the end of every training run). The page is one self-contained file; it
reloads itself every minute while a run is in progress. Bootstrap results are cached per run in
runs/<id>/dash_cache.json, keyed by the reference run and the OOF file's modification time (as is the accuracy of
runs from before train_head.py recorded it).
"""
import glob
import json
import re
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

from . import results as R

ROOT = R.ROOT
TEMPLATE = ROOT / "scripts/dashboard_ui/template.html"
OUT = ROOT / "runs/dashboard.html"
DEADLINES = [("Entry and team merger", "2026-10-15"), ("Final submission", "2026-10-22")]
CACHE_VERSION = 2
FLAT = 0.001  # validation macro AUC per epoch: a learning curve gaining less than this counts as flat
TEST_STUDIES = 1300  # docs/competition.md: about 1,300 hidden test studies


# ---------------------------------------------------------------- statistics
def auc(y, p):
    """Mann-Whitney AUC with average ranks for ties (equals sklearn's roc_auc_score); nan if one class is missing."""
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(Y, P, idx):
    return np.nanmean([auc(Y[idx, j], P[idx, j]) for j in range(Y.shape[1])])


def boot_diff(Y, Pa, Pb, n, seed=0):
    """Paired bootstrap of macro(Pb) - macro(Pa): (diff, lo 5%, hi 95%, P(b better))."""
    rng = np.random.default_rng(seed)
    full = np.arange(len(Y))
    d = np.array([macro(Y, Pb, i) - macro(Y, Pa, i) for i in (rng.integers(0, len(Y), len(Y)) for _ in range(n))])
    lo, hi = np.nanpercentile(d, [5, 95])
    return {"diff": float(macro(Y, Pb, full) - macro(Y, Pa, full)), "lo": float(lo), "hi": float(hi),
            "p_better": float(np.nanmean(d > 0))}


def cliff(x, y):
    """Cliff's delta of x over y: P(x > y) - P(x < y) over all pairs (-1 every x loses, 0 no difference, +1 every x
    wins). x and y are one value per seed, e.g. macro AUC of every seed of two configs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sign(x[:, None] - y[None, :]).mean())


def boot_cliff(x, y, n=2000, seed=0):
    """Cliff's delta with a 90% bootstrap interval, resampling the seeds within each group: (delta, lo, hi)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    X, Y = x[rng.integers(0, len(x), (n, len(x)))], y[rng.integers(0, len(y), (n, len(y)))]
    lo, hi = np.percentile(np.sign(X[:, :, None] - Y[:, None, :]).mean((1, 2)), [5, 95])
    return cliff(x, y), float(lo), float(hi)


def welch(x, y, level=0.90):
    """Welch's interval for mean(x) - mean(y), the t-interval for two small samples with unequal variances:
    [difference, lo, hi]; None when a sample has fewer than 2 values or nothing varies."""
    from scipy.stats import t
    x, y = np.asarray(x, float), np.asarray(y, float)
    if min(len(x), len(y)) < 2:
        return None
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    if vx + vy == 0:
        return None
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    half = float(t.ppf(0.5 + level / 2, df) * np.sqrt(vx + vy))
    d = float(x.mean() - y.mean())
    return [d, d - half, d + half]


def beta(y, x):
    """Beta of y on x, as in markets: cov(y, x) / var(x), how much y moves when x moves by 1."""
    c = np.cov(np.asarray(y, float), np.asarray(x, float))
    return float(c[0, 1] / c[1, 1])


def boot_beta(y, x, n=2000, seed=0):
    """Beta with a 90% bootstrap interval over the points: [beta, lo, hi]."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    i = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    X, Y = x[i] - x[i].mean(1, keepdims=True), y[i] - y[i].mean(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):  # a resample of one repeated point has no variance
        lo, hi = np.nanpercentile((X * Y).sum(1) / (X * X).sum(1), [5, 95])
    return [round(beta(y, x), 3), round(float(lo), 3), round(float(hi), 3)]


def greeks(hist, k=3):
    """A learning curve's sensitivities, named after the options Greeks. delta = validation macro AUC gained per epoch,
    gamma = the change in delta per epoch: differences within each fold, averaged over the folds, as a k-epoch mean.
    hist: {fold: [AUC per epoch]} (compact_history). -> delta from epoch 2 on, gamma from epoch 3 on, flat = the epoch
    from which delta stays below FLAT (None while it has not), now = [delta, gamma] of the newest fold on its own."""
    folds = [[v for v in f if v is not None] for f in hist.values()]

    def curve(fs):
        d = [np.mean([f[t] - f[t - 1] for f in fs if len(f) > t]) for t in range(1, max((len(f) for f in fs), default=0))]
        d = [round(float(np.mean(d[max(0, i - k + 1):i + 1])), 5) for i in range(len(d))]
        return d, [round(b - a, 5) for a, b in zip(d, d[1:])]

    delta, gamma = curve(folds)
    over = [i for i, v in enumerate(delta) if v >= FLAT]  # delta[i] is the gain going into epoch i + 2
    flat = None if not delta or (over and over[-1] == len(delta) - 1) else (over[-1] + 3 if over else 2)
    d, g = curve(folds[-1:])
    return {"delta": delta, "gamma": gamma, "flat": flat, "now": [d[-1] if d else None, g[-1] if g else None]}


def path3d(hist, g):
    """A run's learning curve as a path for the 3D view, one point per epoch: [[epoch, auc, acc, delta, gamma]]. auc and
    acc are the validation macro AUC and macro accuracy (0.5 threshold) averaged over the folds that reached the epoch,
    delta and gamma come from g = greeks() (None before epochs 2 and 3). hist: the history.jsonl records. Epochs without
    both an AUC and an accuracy are dropped; None when no epoch has an accuracy (runs trained before 2026-10-13)."""
    per = {}
    for r in hist:
        per.setdefault(r["epoch"], []).append((r.get("val_macro_auc"), r.get("val_macro_acc")))
    out = []
    for e in sorted(per):
        a, c = ([v for v in col if v is not None] for col in zip(*per[e]))
        if a and c:
            out.append([e, round(float(np.mean(a)), 4), round(float(np.mean(c)), 4),
                        g["delta"][e - 2] if 2 <= e < len(g["delta"]) + 2 else None,
                        g["gamma"][e - 3] if 3 <= e < len(g["gamma"]) + 3 else None])
    return out or None


def macro_acc(oof, folds, report_t):
    """Macro accuracy at the 0.5 threshold of out-of-fold predictions on the report-only studies against the report
    labels: the mean over the labels of mean((p > 0.5) == label), rounded as train_head.py's report_acc, which new runs
    carry in metrics.json. A display metric for the 3D view: AUC, which is rank-based, decides."""
    ids = oof.index[~oof.index.isin(folds.index[folds.is_gold])]
    hit = (oof.loc[ids, R.LABELS].values > 0.5) == (report_t.loc[ids, R.LABELS].values > 0.5)
    return round(float(np.mean(np.round(hit.mean(0), 4))), 4)


def boot_labels(Y, P, n, seed=0):
    """Per-label AUC with a 90% bootstrap interval: {label: [auc, lo, hi]}."""
    rng = np.random.default_rng(seed)
    out = {}
    for j, l in enumerate(R.LABELS):
        a = auc(Y[:, j], P[:, j])
        if a != a:
            continue
        b = [auc(Y[i, j], P[i, j]) for i in (rng.integers(0, len(Y), len(Y)) for _ in range(n))]
        lo, hi = np.nanpercentile(b, [5, 95])
        out[l] = [round(float(a), 4), round(float(lo), 4), round(float(hi), 4)]
    return out


# ---------------------------------------------------------------- inputs
def truth(labels_file, label_map):
    folds = pd.read_csv(ROOT / "meta/folds.csv").set_index("StudyInstanceUID")
    train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c != "Report").set_index("StudyInstanceUID")
    scores = pd.read_csv(ROOT / labels_file).set_index("StudyInstanceUID")
    lm = {l: l for l in R.LABELS} | json.loads(label_map)
    report = (scores[[lm[l] for l in R.LABELS]].set_axis(R.LABELS, axis=1) > 0.5).astype(int)
    return folds, train[R.LABELS], report


def verdicts():
    """experiments.md notes -> {run id: (verdict, notes)}. Verdicts are the bold tags written into the Notes column."""
    out = {}
    for line in (ROOT / "experiments.md").read_text().splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 9 or not re.match(r"\d{4}-\d{2}-\d{2}", cols[0]):
            continue
        notes = " | ".join(cols[8:])
        m = re.search(r"\*\*(Current reference|Adopted|Rejected|Neutral|Superseded|Baseline)", notes)
        v = {"Current reference": "reference"}.get(m.group(1), m.group(1).lower()) if m else ""
        out[cols[1]] = (v, re.sub(r"\*\*", "", notes))
    return out


def head_params(d):
    f = d / "head_fold0.pt"
    if not f.exists():
        return None
    import torch
    return int(sum(t.numel() for t in torch.load(f, map_location="cpu").values()))


def compact_history(hist):
    per = {}
    for r in hist:
        per.setdefault(str(r["fold"]), []).append(None if r["val_macro_auc"] is None else round(r["val_macro_auc"], 4))
    return per


def train_minutes(hist):
    """A run's training time in minutes with the pauses taken out: the median time per epoch times the epochs trained.
    metrics.json's minutes are wall-clock, so a run the Mac slept through, or that was paused, reads hours long there.
    hist: the history.jsonl records; None without any."""
    t = [h["elapsed_s"] for h in hist if h.get("elapsed_s") is not None]
    return float(np.median(np.diff([0] + t)) * len(t) / 60) if t else None


def read_cache(f):
    """A run's dash_cache.json; {} when it is missing or unreadable (cut short, empty, not an object), so it is rebuilt
    instead of stopping the build. An accuracy that is not a number is dropped, and so worked out again."""
    try:
        c = json.loads(f.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(c, dict):
        return {}
    a = c.get("acc")
    if "acc" in c and (isinstance(a, bool) or not isinstance(a, (int, float)) or not np.isfinite(a)):
        del c["acc"]
    return c


def stored_acc(d):
    """A finished run's report accuracy without recomputing it: metrics.json's report_acc, else the value runs_section
    cached for the current oof.csv; None when there is neither. Never raises (the training log calls it)."""
    try:
        a = (R._json(d / "metrics.json") or {}).get("report_acc")
        if a is None:
            c = read_cache(d / "dash_cache.json")
            a = c.get("acc") if str(c.get("key", "")).endswith(f"|{(d / 'oof.csv').stat().st_mtime}") else None
        return None if a is None else float(a)
    except (OSError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------- sections
def runs_section(runs, ref, V):
    folds, gold_t, report_t = truth(runs.loc[ref, "labels_file"], runs.loc[ref, "label_map"])
    ref_oof = pd.read_csv(runs.loc[ref, "dir"] / "oof.csv").set_index("StudyInstanceUID")
    rep_truth = {(runs.loc[ref, "labels_file"], runs.loc[ref, "label_map"]): report_t}  # report labels per labelling
    out = []
    for rid, r in runs.iterrows():
        d = r["dir"]
        args = json.loads((d / "args.json").read_text()) if (d / "args.json").exists() else {}
        kind = "ensemble" if rid.startswith(("ens-", "mix-")) else "single"
        cache_f = d / "dash_cache.json"
        oof_f = d / "oof.csv"
        key = f"v{CACHE_VERSION}|{ref}|{oof_f.stat().st_mtime if oof_f.exists() else 0}"
        cache = read_cache(cache_f)
        stale = cache.get("key") != key
        if stale:
            cache = {"key": key, "params": head_params(d)}
            if r.status == "done" and oof_f.exists() and rid != ref:
                o = pd.read_csv(oof_f).set_index("StudyInstanceUID")
                common = sorted(set(o.index) & set(ref_oof.index))
                g = [u for u in common if folds.loc[u, "is_gold"]]
                p = [u for u in common if not folds.loc[u, "is_gold"]]
                cache["vs_ref"] = {
                    "gold": boot_diff(gold_t.loc[g].values.astype(int), ref_oof.loc[g, R.LABELS].values,
                                      o.loc[g, R.LABELS].values, 1000),
                    "report": boot_diff(report_t.loc[p].values, ref_oof.loc[p, R.LABELS].values,
                                        o.loc[p, R.LABELS].values, 200)}
        # accuracy: new runs carry it in metrics.json; older ones get it from their OOF once, added to the cache under
        # the same key so the bootstraps above are not redone. Ensembles are rank averages, where 0.5 is no threshold.
        acc = (R._json(d / "metrics.json") or {}).get("report_acc")
        if acc is None and "acc" not in cache and kind == "single" and r.status == "done" and oof_f.exists():
            lab = (r.labels_file, r.label_map)  # a run on other labels is scored against its own, as its report AUC is
            try:
                if lab not in rep_truth:
                    rep_truth[lab] = truth(*lab)[2]
                cache["acc"] = macro_acc(pd.read_csv(oof_f).set_index("StudyInstanceUID"), folds, rep_truth[lab])
                stale = True
            except (OSError, KeyError, TypeError, ValueError):  # its labels are gone: no accuracy, the page still builds
                pass
        if stale and r.status == "done":
            tmp = cache_f.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            tmp.replace(cache_f)  # atomic: a build that is cut off never leaves half a cache behind
        hist = r["history"]
        curves = compact_history(hist)
        g = greeks(curves)
        n_done = len(hist)
        total = len(str(args.get("folds", "0,1,2,3,4")).split(",")) * int(args.get("epochs", 12) or 12)
        sec_per_epoch = hist[-1]["elapsed_s"] / n_done if n_done else None
        v, notes = V.get(rid, ("", ""))
        out.append({
            "id": rid, "status": r.status, "kind": kind, "change": r.change, "verdict": "reference" if rid == ref else v,
            "notes": notes, "gold": r.gold, "report": r.report, "gold_by_label": r.gold_by_label,
            "report_by_label": r.report_by_label, "minutes": r.minutes, "feats": r.feats, "canon": bool(r.canon),
            "slice_drop": float(r.slice_drop or 0), "epochs": args.get("epochs"), "seed": args.get("seed"),
            "lr": args.get("lr"), "d": args.get("d"), "layers": args.get("layers"), "drop": args.get("drop", 0.1),
            "members": args.get("members"), "params": cache.get("params"), "vs_ref": cache.get("vs_ref"),
            "history": curves, "greeks": g, "progress": [n_done, total],
            "acc": acc if acc is not None else cache.get("acc"), "path3d": path3d(hist, g),
            "eta_min": round(sec_per_epoch * (total - n_done) / 60, 1) if sec_per_epoch and r.status == "running" else None,
            "last": hist[-1] if hist else None,
            "started": datetime.strptime(rid[:15], "%Y%m%d-%H%M%S").isoformat() if re.match(r"\d{8}-\d{6}", rid) else None,
        })
    return out, (folds, gold_t, report_t, ref_oof)


NOT_CONFIG = ("seed", "log", "no_open", "workers")  # what may differ between seeds of one config
OFF = {"canon": "", "slice_drop": 0.0, "gold_repeat": 1, "bootstrap": False, "ema": 0.0, "label_smooth": 0.0, "aux": "",
       "teacher": "", "slot_proj": False, "local": False}  # train_head.py switches at their off value
DEPENDS = {"distill": "teacher", "aux_weight": "aux"}  # only used when the other one is on
MIN_SEEDS, CI_SEEDS = 3, 5  # a config is listed from 3 seeds; an interval needs 5 seeds in both configs
MIN_CONFIGS = 8  # betas are measured across configs, so they need a spread of them


def config_of(args):
    """The settings that define a config: train_head.py args without the seed and without switches left off, so runs
    from before a switch existed still match the later runs that leave it off."""
    a = {k: v for k, v in args.items() if k not in NOT_CONFIG and not (k in OFF and v in (OFF[k], "none"))}
    return {k: v for k, v in a.items() if k not in DEPENDS or DEPENDS[k] in a}


def config_groups(runs):
    """Finished single runs grouped into configs (same settings, different seeds): [{"config": ..., "runs": [ids]}]."""
    groups = {}
    for rid, r in runs.iterrows():
        a = r["dir"] / "args.json"
        if r.status != "done" or rid.startswith(("ens-", "mix-")) or not a.exists() or pd.isna(r.gold) or pd.isna(r.report):
            continue
        cfg = config_of(json.loads(a.read_text()))
        groups.setdefault(json.dumps(cfg, sort_keys=True), {"config": cfg, "runs": []})["runs"].append(rid)
    return list(groups.values())


def best_group(groups, ref):
    """The current best config: the auto-improve base (runs/auto/state.json), else the reference run's config."""
    state = ROOT / "runs/auto/state.json"
    anchors = (json.loads(state.read_text()).get("base_runs", []) if state.exists() else []) + [ref]
    return next((g for a in anchors for g in groups if a in g["runs"]), None)


def differs(cfg, base_cfg):
    """How a config differs from the base config, written like the train_head.py flags ("" if it does not)."""
    out = []
    for k in sorted(set(cfg) | set(base_cfg)):
        v = cfg.get(k)
        if v != base_cfg.get(k) and not (v is None and k in DEPENDS):  # "teacher off" already says it
            out.append(k.replace("_", "-") + (" off" if v is None else " on" if v is True else
                                              f" {Path(v).name}" if k in ("feats", "labels", "teacher", "canon") else f" {v}"))
    return " · ".join(out)


def configs_section(runs, groups, base):
    """Every config with enough seeds against the current best config: seed by seed (Cliff's delta) and as a difference
    in mean macro AUC with Welch's 90% interval (the forest plot). base_runs: the current best config's runs in the
    order they were trained (the control chart)."""
    if base is None:
        return {"min_seeds": MIN_SEEDS, "ci_seeds": CI_SEEDS, "rows": [], "base_runs": []}

    rows = []
    for g in groups:
        if g is not base and len(g["runs"]) < MIN_SEEDS:
            continue
        row = {"desc": differs(g["config"], base["config"]) or "current best config", "is_base": g is base, "n": len(g["runs"]),
               "first": g["runs"][0], "last": g["runs"][-1]}
        for k in ("report", "gold"):
            x, y = runs.loc[g["runs"], k].astype(float).values, runs.loc[base["runs"], k].astype(float).values
            row[k] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None}
            if g is not base:
                d, lo, hi = boot_cliff(x, y)
                row[k] |= {"cliff": d, "diff": welch(x, y)} | ({"lo": lo, "hi": hi} if min(len(x), len(y)) >= CI_SEEDS else {})
        rows.append(row)
    rows.sort(key=lambda r: (not r["is_base"], -r["report"]["mean"]))
    return {"min_seeds": MIN_SEEDS, "ci_seeds": CI_SEEDS, "rows": rows, "base_runs": base["runs"]}


def same_labels(groups, base):
    """The configs trained on the current best config's labels and label map: the ones that compare across configs."""
    return [g for g in groups if base and all(g["config"].get(k) == base["config"].get(k) for k in ("labels", "label_map"))]


def betas_section(runs, groups, base):
    """Betas across configs, one point per config (the mean over its seeds), for the configs trained on the current
    best config's labels. model: a label's report AUC against the mean report AUC of the other labels (which labels
    gain most from a better model). transfer: a label's gold AUC against its own report AUC, and the macro likewise
    (whether gains on the report labels show up on gold). Each is [beta, lo, hi], a 90% bootstrap interval. points,
    alpha and the macro beta are the characteristic line of gold against report, as in the CAPM."""
    same = same_labels(groups, base)
    if len(same) < MIN_CONFIGS:
        return None

    def per_label(col):
        return np.array([[np.mean([runs.loc[r, col].get(l, np.nan) for r in g["runs"]]) for l in R.LABELS] for g in same])

    def macro(col):
        return np.array([runs.loc[g["runs"], col].astype(float).mean() for g in same])

    rep, gold = per_label("report_by_label"), per_label("gold_by_label")
    mx, my = macro("report"), macro("gold")
    out = {"n": len(same), "model": {}, "transfer": {"Macro": boot_beta(my, mx)}, "r": float(np.corrcoef(my, mx)[0, 1]),
           "alpha": float(my.mean() - beta(my, mx) * mx.mean()),  # the characteristic line: gold = alpha + beta * report
           "points": [{"x": float(a), "y": float(b), "n": len(g["runs"]), "is_base": g is base,
                       "desc": differs(g["config"], base["config"]) or "current best config"} for g, a, b in zip(same, mx, my)]}
    for j, l in enumerate(R.LABELS):
        if not (np.isnan(rep[:, j]).any() or np.isnan(gold[:, j]).any()):
            out["model"][l] = boot_beta(rep[:, j], np.delete(rep, j, 1).mean(1))
            out["transfer"][l] = boot_beta(gold[:, j], rep[:, j])
    return out


def plane_section(runs, groups, base, acc):
    """The 3D view's plane: a two-factor model across configs, as a market factor model with two betas. Ordinary least
    squares of accuracy = alpha + beta_auc * report macro AUC + beta_time * training minutes, one point per config (the
    mean over its seeds), over the configs of betas_section. acc: {run id: accuracy at the 0.5 threshold or None}.
    Minutes are train_minutes(), not the wall-clock minutes of metrics.json: one config whose seeds sat through hours
    of sleep would otherwise set beta_time on its own. ci: a 90% bootstrap interval over the configs for both betas,
    as boot_beta. Configs without an accuracy or a time are skipped; None with fewer than MIN_CONFIGS left, or when
    the points span no plane (the accuracy or a factor never varies, or the two factors move as one)."""
    pts = []
    for g in same_labels(groups, base):
        a = [acc[r] for r in g["runs"] if acc.get(r) is not None]
        m = [v for v in (train_minutes(runs.loc[r, "history"]) for r in g["runs"]) if v is not None]
        if a and m:
            pts.append({"minutes": float(np.mean(m)), "auc": float(runs.loc[g["runs"], "report"].astype(float).mean()),
                        "acc": float(np.mean(a)), "n": len(g["runs"]), "is_base": g is base,
                        "desc": differs(g["config"], base["config"]) or "current best config"})
    if len(pts) < MIN_CONFIGS:
        return None
    X, y = np.array([[p["auc"], p["minutes"]] for p in pts]), np.array([p["acc"] for p in pts])
    if np.ptp(y) < 1e-9:
        return None

    def ols(X, y):
        """[beta_auc, beta_time], or None when the points span no plane. Centred and scaled before solving: AUC spans
        0.01s, minutes span 10s."""
        sd = X.std(0)
        if sd.min() < 1e-9:
            return None
        b, _, rank, _ = np.linalg.lstsq((X - X.mean(0)) / sd, y - y.mean(), rcond=1e-8)
        return b / sd if rank == 2 else None

    b = ols(X, y)
    if b is None:
        return None
    boot = [ols(X[i], y[i]) for i in np.random.default_rng(0).integers(0, len(y), (2000, len(y)))]
    boot = [v for v in boot if v is not None]  # a resample of points in one line has no plane
    lo, hi = np.percentile(boot, [5, 95], axis=0) if boot else (None, None)
    yc = y - y.mean()
    return {"n": len(pts), "points": pts,
            "coef": {"alpha": float(y.mean() - X.mean(0) @ b), "beta_auc": float(b[0]), "beta_time": float(b[1])},
            "ci": {"beta_auc": [float(lo[0]), float(hi[0])], "beta_time": [float(lo[1]), float(hi[1])]} if boot else None,
            "r2": float(1 - ((yc - (X - X.mean(0)) @ b) ** 2).sum() / (yc ** 2).sum())}


def reference_section(runs, ref, truths):
    folds, gold_t, report_t, oof = truths
    cache_f = runs.loc[ref, "dir"] / "dash_ref_cache.json"
    key = f"v{CACHE_VERSION}|{(runs.loc[ref, 'dir'] / 'oof.csv').stat().st_mtime}"
    if cache_f.exists() and json.loads(cache_f.read_text()).get("key") == key:
        return json.loads(cache_f.read_text())["data"]
    g = [u for u in oof.index if folds.loc[u, "is_gold"]]
    p = [u for u in oof.index if not folds.loc[u, "is_gold"]]
    Yg, Pg = gold_t.loc[g].values.astype(int), oof.loc[g, R.LABELS].values
    Yr, Pr = report_t.loc[p].values, oof.loc[p, R.LABELS].values
    grid = np.linspace(0, 1, 101)
    roc = {}
    for j, l in enumerate(R.LABELS):
        fr, tr, _ = roc_curve(Yr[:, j], Pr[:, j])
        fg, tg, _ = roc_curve(Yg[:, j], Pg[:, j]) if 0 < Yg[:, j].sum() < len(Yg) else ([], [], [])
        roc[l] = {"report": [[round(float(x), 3), round(float(y), 3)] for x, y in zip(grid, np.interp(grid, fr, tr))],
                  "gold": [[round(float(x), 3), round(float(y), 3)] for x, y in zip(fg, tg)],
                  "n_pos_report": int(Yr[:, j].sum()), "n_pos_gold": int(Yg[:, j].sum())}
    lat = pd.read_csv(ROOT / "meta/laterality.csv").set_index("StudyInstanceUID").side
    side = {}
    for l_side in ["R", "L"]:
        ids = [u for u in p if lat.get(u) == l_side]
        for j, l in enumerate(R.LABELS):
            side.setdefault(l, {})[l_side] = round(float(auc(report_t.loc[ids, l].values, oof.loc[ids, l].values)), 4)
            side[l][f"n{l_side}"] = len(ids)
    labeller = json.loads((ROOT / "data/labeller/all-v2/gold_eval_all.json").read_text())
    lm = json.loads(runs.loc[ref, "label_map"])
    data = {"ci": {"gold": boot_labels(Yg, Pg, 1000), "report": boot_labels(Yr, Pr, 200)}, "roc": roc, "side": side,
            "one_sided": R.ONE_SIDED,
            "labeller_vs_gold": {l: labeller.get(lm.get(l, l), {}).get("auc") for l in R.LABELS},
            "labeller_column": {l: lm.get(l, l) for l in R.LABELS}, "n_gold": len(g), "n_report": len(p)}
    cache_f.write_text(json.dumps({"key": key, "data": data}))
    return data


def normalise_vendor(m):
    m = str(m).lower()
    for k, v in [("siemens", "Siemens"), ("ge ", "GE"), ("gehc", "GE"), ("philips", "Philips"), ("toshiba", "Canon / Toshiba"),
                 ("canon", "Canon / Toshiba")]:
        if k in m + " ":
            return v
    return "Other / missing"


def data_section(truths):
    folds, gold_t, report_t, _ = truths
    report_only = [u for u in report_t.index if u in folds.index and not folds.loc[u, "is_gold"]]
    gold_ids = folds.index[folds.is_gold]
    index = pd.read_csv(ROOT / "data/features/dinov2s/index.csv")
    per_study = index.groupby("StudyInstanceUID").length.sum()
    slots = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "data/pp-slots/slots_shard*.csv")))])
    first = slots.drop_duplicates("StudyInstanceUID")
    fs = first.field_strength.round(1).map(lambda x: "1.5 T" if abs(x - 1.5) < 0.2 else "3 T" if abs(x - 3) < 0.2
                                           else "other / missing")
    lat = pd.read_csv(ROOT / "meta/laterality.csv")
    tagged = lat[lat.tag.isin(["L", "R"])]
    rep_tag = tagged[tagged.report.isin(["L", "R"])]
    caches = []
    for d in sorted((ROOT / "data/features").iterdir()):
        files = list(d.glob("feats_shard*.npy"))
        if not files:
            continue
        s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        shape = np.load(files[0], mmap_mode="r").shape
        caches.append({"name": d.name, "tokens": int(shape[1]), "dim": int(shape[2]),
                       "gb": round(sum(f.stat().st_size for f in files) / 1e9, 2),
                       "variance_kept": s.get("variance_kept"), "encoder": s.get("encoder", "DINOv2 ViT-S/14")})
    labeller = json.loads((ROOT / "data/labeller/all-v2/gold_eval_all.json").read_text())
    return {
        "studies": int(len(folds)), "gold_studies": int(folds.is_gold.sum()), "report_studies": len(report_only),
        "slices": int(index.length.sum()), "series": int(len(index)),
        "slices_per_study": {"median": float(per_study.median()), "p10": float(per_study.quantile(0.1)),
                             "p90": float(per_study.quantile(0.9))},
        "folds": [{"fold": int(f), "studies": int((folds.fold == f).sum()), "gold": int(folds[folds.fold == f].is_gold.sum())}
                  for f in sorted(folds.fold.unique())],
        "prevalence": {l: {"report": round(float(report_t.loc[report_only, l].mean()), 4),
                           "gold": round(float(gold_t.loc[gold_ids, l].mean()), 4),
                           "gold_pos": int(gold_t.loc[gold_ids, l].sum())} for l in R.LABELS},
        "slot_coverage": {s: int(n) for s, n in index.groupby("slot").StudyInstanceUID.nunique().items()},
        "slot_slices": {s: float(v) for s, v in index.groupby("slot").length.median().items()},
        "vendors": first.manufacturer.map(normalise_vendor).value_counts().to_dict(),
        "field_strength": fs.value_counts().to_dict(),
        "laterality": {"tag": {"R": int((lat.tag == "R").sum()), "L": int((lat.tag == "L").sum())},
                       "image": {"R": int(((lat.side_source == "image") & (lat.side == "R")).sum()),
                                 "L": int(((lat.side_source == "image") & (lat.side == "L")).sum())},
                       "probe_vs_tag": round(float(((tagged.p_left > 0.5) == (tagged.tag == "L")).mean()), 4),
                       "report_vs_tag": round(float((rep_tag.report == rep_tag.tag).mean()), 4),
                       "n_tagged": int(len(tagged)), "n_report_and_tag": int(len(rep_tag))},
        "caches": caches, "labeller_macro": labeller.get("macro_auc"),
    }


def efficiency_section(data):
    feat = json.loads((ROOT / "data/features/dinov2s/summary.json").read_text())
    slices_per_study = data["slices"] / data["studies"]
    encode_min = TEST_STUDIES * slices_per_study / feat["slices"] * feat["minutes"]
    preprocess_min = 12.0  # docs/data.md: 4 slots, all slices, cold Kaggle I/O
    setup_min = 1.5
    total = preprocess_min + encode_min + setup_min
    return {"test_studies": TEST_STUDIES, "preprocess_min": preprocess_min, "encode_min": round(encode_min, 1),
            "setup_min": setup_min, "heads_min": 0.1, "total_min": round(total + 0.1, 1),
            "auc_equiv_at_090": round((total + 0.1) * 60 / 32400 * 0.4, 4),
            "encoder_rate": round(feat["slices"] / feat["minutes"] / 60), "measured": False}


# ---------------------------------------------------------------- terminal summary
def analysis_text(run_id):
    """The statistical analysis of one finished run as plain text, for the training log: its scores next to the other
    seeds of its config, Cliff's delta against the current best config, the learning curve's delta and gamma, and the
    betas across configs beside each label's scores."""
    runs = R.load_runs()
    r = runs.loc[run_id]
    groups = config_groups(runs)
    base = best_group(groups, None)
    mine = next((g for g in groups if run_id in g["runs"]), None)
    acc = stored_acc(r["dir"])
    out = [f"=== statistical analysis: run {run_id} ===", f"macro AUC        gold {r.gold:.4f}   report {r.report:.4f}"
           + (f"   accuracy {acc:.3f} (0.5 threshold, report labels)" if acc is not None else "")]
    if mine:
        x = runs.loc[mine["runs"], ["gold", "report"]].astype(float)
        pm = {k: f"{x[k].mean():.4f}" + (f" ± {x[k].std():.4f}" if len(x) > 1 else "") for k in x}
        out.append(f"its config       {len(x)} seeds: gold {pm['gold']}, report {pm['report']}; this run is "
                   f"{int((x.report > r.report).sum()) + 1} of {len(x)} on report, {int((x.gold > r.gold).sum()) + 1} of {len(x)} on gold")
        if mine is base:
            out.append("vs current best  this is the current best config")
        elif base:
            y = runs.loc[base["runs"], ["gold", "report"]].astype(float)
            cl = {}
            for k in ("report", "gold"):
                d, lo, hi = boot_cliff(x[k].values, y[k].values)
                cl[k] = f"{d:+.2f}" + (f" [{lo:+.2f}, {hi:+.2f}]" if min(len(x), len(y)) >= CI_SEEDS else "")
            out.append(f"vs current best  Cliff's delta report {cl['report']}, gold {cl['gold']} ({len(x)} seeds against {len(y)}; "
                       f"+1 = every seed wins, -1 = every seed loses)")
    g = greeks(compact_history(r.history))
    out.append("learning curve   " + (f"flat from epoch {g['flat']} of {len(g['delta']) + 1}" if g["flat"] else "not flat by the last epoch")
               + (f"; at the end delta {round(g['delta'][-1], 4) + 0:+.4f} and gamma {round(g['gamma'][-1], 4) + 0:+.4f} AUC per epoch" if g["gamma"] else ""))
    b = betas_section(runs, groups, base)
    if b:
        m = b["transfer"]["Macro"]
        out.append(f"gold vs report   beta {m[0]:.2f} across {b['n']} configs (90% interval {m[1]:.2f} to {m[2]:.2f}, r {b['r']:.2f}): "
                   f"a report gain of 0.010 has meant about {m[0] * 0.01:.3f} on gold")
    out.append(f"{'per label':19s}{'gold':>7s}{'report':>8s}" + ("   model beta   transfer beta (90% interval)" if b else ""))
    for l in R.LABELS:
        line = f"  {l:17s}" + "".join(f"{v:{w}.3f}" if v is not None else " " * (w - 1) + "-"
                                    for v, w in ((r.gold_by_label.get(l), 7), (r.report_by_label.get(l), 8)))
        if b and l in b["model"]:
            t = b["transfer"][l]
            line += f"   {b['model'][l][0]:+10.2f}   {t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------- build
def clean(o):
    """JSON-safe: NaN / inf -> None, numpy scalars -> Python, Paths -> str."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def collect(ref=None):
    runs = R.load_runs()
    V = verdicts()
    done_single = [r for r in runs.index if runs.loc[r, "status"] == "done" and not r.startswith(("ens-", "mix-"))]
    if ref is None:
        tagged = [r for r in done_single if V.get(r, ("",))[0] == "reference"]
        ref = tagged[-1] if tagged else max(done_single, key=lambda r: runs.loc[r, "report"])
    run_rows, truths = runs_section(runs, ref, V)
    groups = config_groups(runs)
    base = best_group(groups, ref)
    data = data_section(truths)
    acc = {r["id"]: r["acc"] for r in run_rows}
    return {"built": datetime.now().isoformat(timespec="seconds"), "reference": ref, "labels": R.LABELS,
            "deadlines": [{"name": n, "date": d, "days": (date.fromisoformat(d) - date.today()).days} for n, d in DEADLINES],
            "runs": run_rows, "flat": FLAT, "configs": configs_section(runs, groups, base),
            "betas": betas_section(runs, groups, base), "plane": plane_section(runs, groups, base, acc),
            "ref": reference_section(runs, ref, truths), "data": data,
            "efficiency": efficiency_section(data)}


def build(ref=None, out=OUT):
    t = time.time()
    payload = json.dumps(clean(collect(ref)), separators=(",", ":"), allow_nan=False)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
    tmp = out.with_suffix(".tmp.html")
    tmp.write_text(html)
    tmp.replace(out)  # atomic: an open page never reloads a half-written file
    return out, time.time() - t
