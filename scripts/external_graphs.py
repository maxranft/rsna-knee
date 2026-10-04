"""Graphs of a real-world test, opened in Preview (scripts/external_study.py draws them at the end of every run).

    python scripts/external_graphs.py "external scan"             # redraw from <folder>/results/, then open
    python scripts/external_graphs.py "external scan" --no-open

Writes <folder>/results/graphs/:
  1_scores_vs_training.png      each finding's score against the training knees' out-of-fold scores; the dot is the
                                radiologist's call from truth.csv (red present, blue absent, grey not filled in)
  2_where_the_model_looked.png  slice occlusion evidence for every finding across every slice of every series
                                (red pushed toward present, blue toward absent)
  3_strongest_slices.png        for the radiologist's positives and the model's top findings, the slice that pushed
                                the score hardest toward present
Reads only the saved results, so it redraws in seconds (e.g. after editing truth.csv) without re-running the model.
Red always means present and blue absent, as in the review tool. The images show a real person's scan: for you only.
"""
import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# light chart surface and ink; present/absent validated as a pair (dataviz validate_palette.js, light: all pass)
SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
GRID, BOX, WHISKER = "#ebeae6", "#dcdbd6", "#b9b8b2"
PRESENT, ABSENT, UNKNOWN, NEUTRAL = "#e34948", "#2a78d6", "#a3a29d", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("absent_present", [ABSENT, NEUTRAL, PRESENT])
SHORT = {"sag_fs": "Sagittal fluid-sensitive", "cor_fs": "Coronal fluid-sensitive", "ax_fs": "Axial fluid-sensitive",
         "t1": "T1"}
NOTICEABLE = 0.05  # log-odds; as in scripts/review.py

plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK,
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "legend.frameon": False})


def ordinal(k):
    k = int(round(k))
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def truth_colour(t):
    return {1: PRESENT, 0: ABSENT}.get(t, UNKNOWN)


def truth_word(t):
    return {1: "present", 0: "absent"}.get(t, "not filled in")


def load(folder):
    res = folder / "results"
    meta = json.loads((res / "study.json").read_text())
    z = np.load(res / "study.npz")
    pred = pd.read_csv(res / "predictions.csv").set_index("finding")
    t = pd.read_csv(folder / "truth.csv", dtype=str) if (folder / "truth.csv").exists() else None
    truth = {} if t is None else {r.finding: int(r.present) for r in t.itertuples() if str(r.present).strip() in {"0", "1"}}
    return meta, z, pred, truth


def header(fig, title, subtitle):
    """Title and a subtitle wrapped to the figure's width (~14 characters per inch at 9.5 pt)."""
    width = int(fig.get_figwidth() * 14)
    lines = [textwrap.fill(part, width) for part in subtitle.split("\n")]
    fig.text(0.012, 0.975, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.975 - 0.42 / fig.get_figheight(), "\n".join(lines), fontsize=9.5, color=INK_2, va="top",
             linespacing=1.35)


def scores_vs_training(meta, z, pred, truth, order, out):
    labels, prob, ref = meta["labels"], z["prob"], z["reference"]
    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(left=0.17, right=0.86, top=0.82, bottom=0.17)
    for row, k in enumerate(order):
        lo, q1, med, q3, hi = np.percentile(ref[:, k], [5, 25, 50, 75, 95])
        ax.plot([lo, hi], [row, row], color=WHISKER, lw=1.2, solid_capstyle="round", zorder=1)
        ax.add_patch(Rectangle((q1, row - 0.2), q3 - q1, 0.4, facecolor=BOX, edgecolor="none", zorder=1))
        ax.plot([med, med], [row - 0.2, row + 0.2], color=INK_3, lw=1, zorder=2)
        ax.scatter(prob[k], row, s=110, color=truth_colour(truth.get(labels[k])), edgecolor=SURFACE, linewidth=2,
                   zorder=3)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.03, len(order) - 0.35, "Percentile", transform=tr, color=INK_3, fontsize=9, va="bottom")
    for row, k in enumerate(order):
        ax.text(1.03, row, ordinal(pred.loc[labels[k], "percentile"]), transform=tr, va="center", color=INK)
    ax.set_yticks(range(len(order)), [labels[k] for k in order])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Model score (probability)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    n_ref = len(ref)
    header(fig, "Where the external scan's scores fall among the training knees",
           f"Each row: its score (dot) against {n_ref:,} training knees (box = middle half, line = 5th to 95th percentile). "
           "Sorted by percentile.\nModel: " + meta.get("model", "packaged ensemble") + ". " + source_note(meta))
    handles = [Line2D([], [], marker="o", ls="", ms=9, mfc=PRESENT, mec=SURFACE, label="Radiologist: present"),
               Line2D([], [], marker="o", ls="", ms=9, mfc=ABSENT, mec=SURFACE, label="Radiologist: absent"),
               Patch(facecolor=BOX, label=f"Training knees (n = {n_ref:,})")]
    if any(labels[k] not in truth for k in order):
        handles.insert(2, Line2D([], [], marker="o", ls="", ms=9, mfc=UNKNOWN, mec=SURFACE, label="Not filled in"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=len(handles), fontsize=9,
              handletextpad=0.4, columnspacing=1.6, labelcolor=INK_2)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def where_the_model_looked(meta, z, truth, order, out):
    labels, ev, slots = meta["labels"], z["evidence"], meta["slots"]
    lengths = [len(z[f"stack_{s}"]) for s in slots]
    bounds = np.cumsum([0] + lengths)
    m = ev[:, order[::-1]].T  # top row = highest percentile, as in graph 1
    v = max(float(np.percentile(np.abs(ev), 99)), NOTICEABLE)
    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.012, 1], height_ratios=[1, 0.035], wspace=0.01, hspace=0.32,
                          left=0.135, right=0.985, top=0.76, bottom=0.14)
    tax, ax, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax.imshow(m, aspect="auto", cmap=DIVERGING, vmin=-v, vmax=v, interpolation="nearest")
    for b in bounds[1:-1]:
        ax.axvline(b - 0.5, color=SURFACE, lw=3)
    for s, a, b in zip(slots, bounds[:-1], bounds[1:]):
        ax.text((a + b - 1) / 2, -0.75, f"{SHORT[s]}\n{b - a} slices, first to last", ha="center", va="bottom",
                fontsize=9, color=INK_2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    names = [labels[k] for k in order[::-1]]
    tax.imshow(np.array([[0.0]] * len(names)), aspect="auto", cmap="gray", vmin=-1, vmax=0)  # sets the row grid
    for i, name in enumerate(names):
        tax.add_patch(Rectangle((-0.5, i - 0.42), 1, 0.84, color=truth_colour(truth.get(name))))
        tax.text(-1.0, i, name, ha="right", va="center", fontsize=10, color=INK, transform=tax.transData)
    tax.set_xlim(-0.5, 0.5)
    tax.axis("off")
    fig.legend(handles=[Patch(color=PRESENT, label="Radiologist: present"), Patch(color=ABSENT, label="Radiologist: absent")],
               loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=1, fontsize=9, labelcolor=INK_2, handlelength=1,
               handleheight=1)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-v, v)), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([])
    for x, ha, text in [(0, "left", "pushed toward absent"), (0.5, "center", "no effect"), (1, "right", "pushed toward present")]:
        cax.text(x, -0.6, text, transform=cax.transAxes, ha=ha, va="top", fontsize=9, color=INK_2)
    header(fig, "Where the model looked, slice by slice",
           "How much each slice moves each finding's score: the score with every slice minus the score without that "
           "slice (averaged over every head). Square at left: the radiologist's call.\nIt explains the model, not the knee. "
           + source_note(meta))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def strongest_slices(meta, z, pred, truth, order, out):
    """Unlike the report's top slice, this always shows one, flagged "weak" when no slice moves the score much."""
    labels, ev, slots = meta["labels"], z["evidence"], meta["slots"]
    bounds = np.cumsum([0] + [len(z[f"stack_{s}"]) for s in slots])
    by_pct = [labels[k] for k in order[::-1]]
    picks = list(dict.fromkeys([l for l in by_pct if truth.get(l) == 1] + by_pct[:3]))[:6]
    picks.sort(key=lambda l: -pred.loc[l, "percentile"])
    fig, axes = plt.subplots(1, len(picks), figsize=(max(3.1 * len(picks), 9), 5))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.66, bottom=0.11, wspace=0.06)
    for a, l in zip(axes, picks):
        e = ev[:, labels.index(l)]
        top = int(e.argmax())
        s_i = int(np.searchsorted(bounds, top, side="right") - 1)
        slot, idx = slots[s_i], top - int(bounds[s_i])
        stack = z[f"stack_{slot}"]
        a.imshow(stack[idx], cmap="gray", vmin=0, vmax=255)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor(truth_colour(truth.get(l)))
            sp.set_linewidth(3)
        a.set_title(f"{l}\nmodel: {ordinal(pred.loc[l, 'percentile'])} percentile\nradiologist: {truth_word(truth.get(l))}",
                    fontsize=10, color=INK, loc="left", pad=6)
        weak = "" if e[top] >= NOTICEABLE else "\nweak: no slice moves this score much"
        a.text(0, -0.04, f"{SHORT[slot]}, slice {idx + 1} of {len(stack)}{weak}", transform=a.transAxes, va="top",
               fontsize=8.5, color=INK_2)
    header(fig, "The slice that pushed each finding hardest toward present",
           "The radiologist's positives and the model's three highest findings. Frame: red = radiologist present, "
           "blue = absent. " + source_note(meta))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def source_note(meta):
    src = meta.get("source", "dicom")
    return "" if src == "dicom" else f"Approximate: {src}."


def write_graphs(folder):
    """-> paths of the graphs written to <folder>/results/graphs/."""
    meta, z, pred, truth = load(folder)
    if "reference" not in z.files:
        raise SystemExit(f'{folder}/results is from an older run; re-run: python scripts/external_study.py "{folder}"')
    order = list(np.argsort(pred.loc[meta["labels"], "percentile"].values))  # bottom row = lowest percentile
    out = folder / "results/graphs"
    out.mkdir(exist_ok=True)
    paths = [out / "1_scores_vs_training.png", out / "2_where_the_model_looked.png", out / "3_strongest_slices.png"]
    scores_vs_training(meta, z, pred, truth, order, paths[0])
    where_the_model_looked(meta, z, truth, order, paths[1])
    if not strongest_slices(meta, z, pred, truth, order, paths[2]):
        paths = paths[:2]
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    paths = write_graphs(args.folder.resolve())
    print(f"graphs: {paths[0].parent}")
    if not args.no_open:
        subprocess.run(["open", *map(str, paths)])


if __name__ == "__main__":
    main()
