"""Results viewer backend (notebooks/results.ipynb) and the compare / side-split CLIs. Metrics only: it reads
runs/<id>/{args.json, metrics.json, history.jsonl, oof.csv}, never report text or images.

Gold (58 studies) is the arbiter, report labels (4,349 studies) the steadier signal; gold differences below ~0.02 are
noise, which is why comparisons come with a paired bootstrap interval.
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
from sklearn.metrics import roc_auc_score

from .progress import AXIS, GRID, SURFACE, TEXT, TEXT_2

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# findings that sit on one side of the knee, so they land on opposite image sides for left and right knees
ONE_SIDED = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "MCL", "Baker's"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order (validated)
BASE_SHADE, CAND_SHADE = "#86b6ef", "#1c5cab"  # dumbbell: one hue, two shades (blue 250 / 550)
DIVERGING = LinearSegmentedColormap.from_list("worse_better", ["#e34948", "#f0efec", "#2a78d6"])  # red, gray, blue
STALE_MIN = 20  # no new epoch for this long and no metrics.json -> "stopped"


# ---------------------------------------------------------------- loading
def _json(p):
    return json.loads(p.read_text()) if p.exists() else None


def _history(d):
    p = d / "history.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def load_runs(include_smoke=False):
    """-> DataFrame, one row per run (index = run id), newest last. Smoke runs (fewer than 5 folds) are dropped unless
    include_smoke; synthetic-data runs (features under a temp dir) always are."""
    rows = []
    for d in sorted(p for p in (ROOT / "runs").iterdir() if p.is_dir()):
        m, hist = _json(d / "metrics.json"), _history(d)
        args = (m or {}).get("args") or _json(d / "args.json") or {}
        if not m and not hist:
            continue
        folds = [int(f) for f in str(args.get("folds", "0,1,2,3,4")).split(",")]
        synthetic = "/tmp/" in str(args.get("feats", "")) or "/private/" in str(args.get("feats", ""))
        if synthetic or (len(folds) < 5 and not include_smoke):
            continue
        age = (time.time() - (d / "history.jsonl").stat().st_mtime) / 60 if hist else 1e9
        status = "done" if m else ("running" if age < STALE_MIN else "stopped")
        done_epochs = len(hist)
        total = len(folds) * int(args.get("epochs", 12)) if args else None
        rows.append({
            "run": d.name, "status": status, "change": args.get("log") or "(no description)",
            "gold": (m or {}).get("gold_macro"), "report": (m or {}).get("report_macro"),
            "progress": f"{done_epochs}/{total}" if total else f"{done_epochs} epochs",
            "minutes": (m or {}).get("minutes") or (round(hist[-1]["elapsed_s"] / 60, 1) if hist else None),
            "feats": Path(str(args.get("feats", "?"))).name, "epochs": args.get("epochs"), "lr": args.get("lr"),
            "d": args.get("d"), "layers": args.get("layers"), "canon": bool(args.get("canon")),
            "slice_drop": args.get("slice_drop", 0.0), "label_map": args.get("label_map", "{}"),
            "gold_by_label": (m or {}).get("gold", {}), "report_by_label": (m or {}).get("report", {}),
            "labels_file": args.get("labels"), "history": hist, "dir": d,
        })
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()


def pick_baseline(runs, baseline=None):
    """The run id to compare against: the given one, else the first finished run whose description starts with
    'baseline', else the first finished run."""
    if baseline:
        return baseline
    done = runs[runs.status == "done"]
    base = [r for r in done.index if done.loc[r, "change"].lower().startswith("baseline")]
    return base[0] if base else (done.index[0] if len(done) else None)


def pick_candidate(runs, candidate=None, baseline=None):
    if candidate:
        return candidate
    done = [r for r in runs[runs.status == "done"].index if r != baseline]
    return done[-1] if done else None


# ---------------------------------------------------------------- tables
def _delta_style(v, scale):
    """Cell CSS for a delta: diverging red (worse) / gray / blue (better), ink always dark so it reads in both themes."""
    if v is None or v != v:
        return f"color: {TEXT_2}"
    c = to_hex(DIVERGING(0.5 + 0.5 * max(-1, min(1, v / scale)) * 0.7))  # 0.7: keep fills light enough for ink
    return f"background-color: {c}; color: {TEXT}"


def leaderboard(runs, baseline=None):
    """Styled table of all runs with deltas against the baseline (gold is the arbiter)."""
    b = pick_baseline(runs, baseline)
    t = runs[["status", "change", "gold", "report", "progress", "minutes", "feats", "canon", "slice_drop", "epochs"]].copy()
    if b is not None:
        t.insert(3, "Δ gold", t.gold - runs.loc[b, "gold"])
        t.insert(5, "Δ report", t.report - runs.loc[b, "report"])
    weakest = []
    for r in runs.index:
        g = runs.loc[r, "gold_by_label"]
        weakest.append(", ".join(f"{k} {v:.2f}" for k, v in sorted(g.items(), key=lambda kv: kv[1])[:3]) if g else "")
    t["weakest gold"] = weakest
    fmt = {"gold": "{:.4f}", "report": "{:.4f}", "Δ gold": "{:+.4f}", "Δ report": "{:+.4f}", "minutes": "{:.1f}",
           "slice_drop": "{:.2f}"}
    s = t.style.format(fmt, na_rep="–")
    if b is not None:
        s = s.map(lambda v: _delta_style(v, 0.03), subset=["Δ gold"]).map(lambda v: _delta_style(v, 0.01), subset=["Δ report"])
        s = s.set_caption(f"Baseline: {b}. Δ gold within ±0.02 is noise (58 studies); Δ report is the steadier signal.")
    return s


def label_table(runs, which="report", baseline=None):
    """Labels x finished runs: AUC per label, cells coloured by the delta against the baseline."""
    b = pick_baseline(runs, baseline)
    done = runs[runs.status == "done"]
    key = f"{which}_by_label"
    t = pd.DataFrame({r: pd.Series(done.loc[r, key]) for r in done.index}).reindex(LABELS)
    t.loc["macro"] = t.mean()
    s = t.style.format("{:.3f}", na_rep="–")
    if b in t.columns:
        scale = 0.05 if which == "gold" else 0.015
        for r in t.columns:
            if r != b:
                s = s.apply(lambda col, r=r: [_delta_style(col[i] - t.loc[i, b], scale) for i in col.index], subset=[r])
        s = s.set_caption(f"{which} AUC per label; colour = difference from {b} (blue better, red worse)")
    return s


# ---------------------------------------------------------------- statistics
def _macro(y, p, idx):
    vals = [roc_auc_score(y[idx, j], p[idx, j]) for j in range(y.shape[1]) if 0 < y[idx, j].sum() < len(idx)]
    return np.mean(vals) if vals else np.nan


def _truth(runs, run):
    """-> (folds, gold labels from train.csv, report labels thresholded at 0.5 with the run's label map)."""
    folds = pd.read_csv(ROOT / "meta/folds.csv").set_index("StudyInstanceUID")
    train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c != "Report").set_index("StudyInstanceUID")
    scores = pd.read_csv(ROOT / runs.loc[run, "labels_file"]).set_index("StudyInstanceUID")
    lm = {l: l for l in LABELS} | json.loads(runs.loc[run, "label_map"])
    report = (scores[[lm[l] for l in LABELS]].set_axis(LABELS, axis=1) > 0.5).astype(int)
    return folds, train, report


def paired_bootstrap(runs, a, b, n_gold=1000, n_report=300, seed=0):
    """Macro-AUC difference b - a with a 90% interval, resampling studies identically for both runs.
    -> DataFrame rows gold / report: diff, lo, hi, P(b better)."""
    folds, train, report = _truth(runs, a)
    oa = pd.read_csv(runs.loc[a, "dir"] / "oof.csv").set_index("StudyInstanceUID")
    ob = pd.read_csv(runs.loc[b, "dir"] / "oof.csv").set_index("StudyInstanceUID")
    common = sorted(set(oa.index) & set(ob.index))
    out = {}
    for name, n in [("gold", n_gold), ("report", n_report)]:
        ids = [u for u in common if bool(folds.loc[u, "is_gold"]) == (name == "gold")]
        y = (train.loc[ids, LABELS] if name == "gold" else report.loc[ids]).values.astype(int)
        pa, pb = oa.loc[ids, LABELS].values, ob.loc[ids, LABELS].values
        full = np.arange(len(ids))
        rng = np.random.default_rng(seed)
        d = []
        for _ in range(n):
            idx = rng.integers(0, len(ids), len(ids))
            d.append(_macro(y, pb, idx) - _macro(y, pa, idx))
        d = np.array(d)
        lo, hi = np.nanpercentile(d, [5, 95])
        out[name] = {"studies": len(ids), "diff": _macro(y, pb, full) - _macro(y, pa, full), "lo 5%": lo, "hi 95%": hi,
                     "P(better)": np.nanmean(d > 0)}
    return pd.DataFrame(out).T


def side_split(runs, run):
    """Report AUC per label for right vs left knees (meta/laterality.csv), report-only studies.
    A model that has not learnt the knee side scores lower on one side for the one-sided findings (2026-09-29, PCA run:
    MCL was 0.06 lower on left knees, 90% CI [-0.10, -0.03]; the menisci and tibiofemoral OA showed no gap)."""
    folds, _, report = _truth(runs, run)
    lat = pd.read_csv(ROOT / "meta/laterality.csv").set_index("StudyInstanceUID").side
    oof = pd.read_csv(runs.loc[run, "dir"] / "oof.csv").set_index("StudyInstanceUID")
    oof = oof[~folds.loc[oof.index, "is_gold"].values]
    t = {}
    for side in ["R", "L"]:
        ids = oof.index[lat.loc[oof.index].values == side]
        t[f"{side} (n={len(ids)})"] = {l: roc_auc_score(report.loc[ids, l], oof.loc[ids, l]) for l in LABELS}
    t = pd.DataFrame(t)
    t["L − R"] = t.iloc[:, 1] - t.iloc[:, 0]
    t["group"] = ["one-sided" if l in ONE_SIDED else "central / anywhere" for l in t.index]
    return t


# ---------------------------------------------------------------- charts
def _short(text, n=60):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0].rstrip(",;:·") + "…"


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fold_mean_curve(hist):
    """-> (epochs, mean val macro AUC over the folds that reached each epoch, per-fold curves)."""
    per = {}
    for r in hist:
        if r["val_macro_auc"] is not None:
            per.setdefault(r["fold"], {})[r["epoch"]] = r["val_macro_auc"]
    eps = sorted({e for c in per.values() for e in c})
    mean = [np.mean([c[e] for c in per.values() if e in c]) for e in eps]
    return eps, mean, per


def plot_curves(runs, run_ids, title="Validation macro AUC by epoch (report labels, mean over folds)"):
    """One line per run (fixed categorical order, at most 4), thin per-fold lines behind it."""
    run_ids = [r for r in run_ids if r in runs.index][:4]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=110, facecolor=SURFACE)
    _style(ax, title)
    for i, r in enumerate(run_ids):
        eps, mean, per = _fold_mean_curve(runs.loc[r, "history"])
        if not eps:
            continue
        for c in per.values():
            ax.plot(list(c), list(c.values()), color=SERIES[i], linewidth=0.8, alpha=0.25, zorder=1)
        name = _short(runs.loc[r, "change"])
        ax.plot(eps, mean, color=SERIES[i], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=f"{r} · {name}", zorder=3)
        best = int(np.argmax(mean))
        ax.annotate(f"{mean[-1]:.3f}", (eps[-1], mean[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=TEXT)
        if mean[best] - mean[-1] >= 0.003:  # only a real peak, not plateau jitter
            ax.annotate(f"peak {mean[best]:.3f}", (eps[best], mean[best]), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT_2)
    ax.set_xlabel("epoch", color=TEXT_2, fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("AUC", color=TEXT_2, fontsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="lower right")
    fig.tight_layout()
    return fig


def plot_dumbbell(runs, a, b):
    """Per label, baseline a -> candidate b, for gold and report AUC side by side. Sorted by the report change."""
    ga, gb = runs.loc[a, "gold_by_label"], runs.loc[b, "gold_by_label"]
    ra, rb = runs.loc[a, "report_by_label"], runs.loc[b, "report_by_label"]
    order = sorted(LABELS, key=lambda l: rb.get(l, np.nan) - ra.get(l, np.nan))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=110, facecolor=SURFACE, sharey=True)
    for ax, (name, va, vb) in zip(axes, [("Report AUC (4,349 studies)", ra, rb), ("Gold AUC (58 studies, noisy)", ga, gb)]):
        _style(ax, name)
        ax.grid(False, axis="y")
        for y, l in enumerate(order):
            x0, x1 = va.get(l, np.nan), vb.get(l, np.nan)
            ax.plot([x0, x1], [y, y], color=AXIS, linewidth=2, zorder=1, solid_capstyle="round")
            ax.scatter([x0], [y], s=55, color=BASE_SHADE, edgecolor=SURFACE, linewidth=1.5, zorder=2)
            ax.scatter([x1], [y], s=55, color=CAND_SHADE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if x1 == x1 and x0 == x0:
                ax.annotate(f"{x1 - x0:+.3f}", (max(x0, x1), y), xytext=(7, 0), textcoords="offset points",
                            va="center", fontsize=8, color=TEXT_2)
        ax.set_yticks(range(len(order)), order, fontsize=9, color=TEXT)
        ax.set_xlabel("AUC", color=TEXT_2, fontsize=9)
    handles = [axes[0].scatter([], [], s=55, color=BASE_SHADE, label=f"baseline {a}"),
               axes[0].scatter([], [], s=55, color=CAND_SHADE, label=f"candidate {b}")]
    fig.suptitle(_short(runs.loc[b, "change"], 90), x=0.02, y=0.98, ha="left", color=TEXT, fontsize=12)
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.985), ncol=2, frameon=False, fontsize=9,
               labelcolor=TEXT_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def live(runs):
    """Running / stopped runs: a status table and their learning curves."""
    act = runs[runs.status != "done"]
    if act.empty:
        print("No run in progress.")
        return None
    t = act[["status", "change", "progress", "minutes"]].copy()
    t["latest val AUC"] = [r["history"][-1]["val_macro_auc"] if r["history"] else None for _, r in act.iterrows()]
    t["fold / epoch"] = [f"{r['history'][-1]['fold']} / {r['history'][-1]['epoch']}" if r["history"] else ""
                         for _, r in act.iterrows()]
    from IPython.display import display
    display(t)
    return plot_curves(runs, list(act.index), title="In progress: validation macro AUC by epoch (mean over folds so far)")
