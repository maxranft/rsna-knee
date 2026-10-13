"""Results dashboard -> runs/dashboard.html: every run with its verdict, configs compared across seeds (Cliff's delta),
the reference model's per-label efficacy (AUC with 90% intervals, ROC curves, label ceiling, knee side), the data,
efficiency, and the run in progress.

Metrics and aggregates only: train.csv is always read without its Report column, and no image is touched.
Built by scripts/dashboard.py (and at the end of every training run). The page is one self-contained file; it
reloads itself every minute while a run is in progress. Bootstrap results are cached per run in
runs/<id>/dash_cache.json, keyed by the reference run and the OOF file's modification time.
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


# ---------------------------------------------------------------- sections
def runs_section(runs, ref, V):
    folds, gold_t, report_t = truth(runs.loc[ref, "labels_file"], runs.loc[ref, "label_map"])
    ref_oof = pd.read_csv(runs.loc[ref, "dir"] / "oof.csv").set_index("StudyInstanceUID")
    out = []
    for rid, r in runs.iterrows():
        d = r["dir"]
        args = json.loads((d / "args.json").read_text()) if (d / "args.json").exists() else {}
        cache_f = d / "dash_cache.json"
        oof_f = d / "oof.csv"
        key = f"v{CACHE_VERSION}|{ref}|{oof_f.stat().st_mtime if oof_f.exists() else 0}"
        cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
        if cache.get("key") != key:
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
            if r.status == "done":
                cache_f.write_text(json.dumps(cache))
        hist = r["history"]
        n_done = len(hist)
        total = len(str(args.get("folds", "0,1,2,3,4")).split(",")) * int(args.get("epochs", 12) or 12)
        sec_per_epoch = hist[-1]["elapsed_s"] / n_done if n_done else None
        v, notes = V.get(rid, ("", ""))
        kind = "ensemble" if rid.startswith(("ens-", "mix-")) else "single"
        out.append({
            "id": rid, "status": r.status, "kind": kind, "change": r.change, "verdict": "reference" if rid == ref else v,
            "notes": notes, "gold": r.gold, "report": r.report, "gold_by_label": r.gold_by_label,
            "report_by_label": r.report_by_label, "minutes": r.minutes, "feats": r.feats, "canon": bool(r.canon),
            "slice_drop": float(r.slice_drop or 0), "epochs": args.get("epochs"), "seed": args.get("seed"),
            "lr": args.get("lr"), "d": args.get("d"), "layers": args.get("layers"), "drop": args.get("drop", 0.1),
            "members": args.get("members"), "params": cache.get("params"), "vs_ref": cache.get("vs_ref"),
            "history": compact_history(hist), "progress": [n_done, total],
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


def config_of(args):
    """The settings that define a config: train_head.py args without the seed and without switches left off, so runs
    from before a switch existed still match the later runs that leave it off."""
    a = {k: v for k, v in args.items() if k not in NOT_CONFIG and not (k in OFF and v in (OFF[k], "none"))}
    return {k: v for k, v in a.items() if k not in DEPENDS or DEPENDS[k] in a}


def configs_section(runs, ref):
    """Finished single runs grouped into configs (same settings, different seeds), each compared seed by seed with the
    current best config: the auto-improve base (runs/auto/state.json), else the reference run's config."""
    groups = {}
    for rid, r in runs.iterrows():
        a = r["dir"] / "args.json"
        if r.status != "done" or rid.startswith(("ens-", "mix-")) or not a.exists() or pd.isna(r.gold) or pd.isna(r.report):
            continue
        cfg = config_of(json.loads(a.read_text()))
        groups.setdefault(json.dumps(cfg, sort_keys=True), {"config": cfg, "runs": []})["runs"].append(rid)
    state = ROOT / "runs/auto/state.json"
    anchors = (json.loads(state.read_text()).get("base_runs", []) if state.exists() else []) + [ref]
    base = next((g for a in anchors for g in groups.values() if a in g["runs"]), None)
    if base is None:
        return {"min_seeds": MIN_SEEDS, "ci_seeds": CI_SEEDS, "rows": []}

    def differs(cfg):  # how a config differs from the base, written like the train_head.py flags
        out = []
        for k in sorted(set(cfg) | set(base["config"])):
            v = cfg.get(k)
            if v != base["config"].get(k) and not (v is None and k in DEPENDS):  # "teacher off" already says it
                out.append(k.replace("_", "-") + (" off" if v is None else " on" if v is True else
                                                  f" {Path(v).name}" if k in ("feats", "labels", "teacher", "canon") else f" {v}"))
        return " · ".join(out)

    rows = []
    for g in groups.values():
        if g is not base and len(g["runs"]) < MIN_SEEDS:
            continue
        row = {"desc": differs(g["config"]) or "current best config", "is_base": g is base, "n": len(g["runs"]),
               "first": g["runs"][0], "last": g["runs"][-1]}
        for k in ("report", "gold"):
            x, y = runs.loc[g["runs"], k].astype(float).values, runs.loc[base["runs"], k].astype(float).values
            row[k] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None}
            if g is not base:
                d, lo, hi = boot_cliff(x, y)
                row[k] |= {"cliff": d} | ({"lo": lo, "hi": hi} if min(len(x), len(y)) >= CI_SEEDS else {})
        rows.append(row)
    rows.sort(key=lambda r: (not r["is_base"], -r["report"]["mean"]))
    return {"min_seeds": MIN_SEEDS, "ci_seeds": CI_SEEDS, "rows": rows}


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
    data = data_section(truths)
    return {"built": datetime.now().isoformat(timespec="seconds"), "reference": ref, "labels": R.LABELS,
            "deadlines": [{"name": n, "date": d, "days": (date.fromisoformat(d) - date.today()).days} for n, d in DEADLINES],
            "runs": run_rows, "configs": configs_section(runs, ref), "ref": reference_section(runs, ref, truths), "data": data,
            "efficiency": efficiency_section(data)}


def build(ref=None, out=OUT):
    t = time.time()
    payload = json.dumps(clean(collect(ref)), separators=(",", ":"), allow_nan=False)
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
    tmp = out.with_suffix(".tmp.html")
    tmp.write_text(html)
    tmp.replace(out)  # atomic: an open page never reloads a half-written file
    return out, time.time() - t
