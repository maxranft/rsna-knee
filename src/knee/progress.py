"""Per-epoch training progress: a JSONL history plus a dashboard PNG re-rendered after every epoch.

    tracker = ProgressTracker(run_dir, labels, folds=[0, 1, 2, 3, 4], epochs=12, title="baseline")
    tracker.log(fold, epoch, train_loss=..., val_loss=..., val_auc={label: auc, ...})
    tracker.finish(gold_macro=..., report_macro=...)

Re-draw a past run from its history:  python src/knee/progress.py runs/<id>

Open runs/<id>/progress.png in VS Code; the preview reloads each time the file is rewritten.
Panels: loss (train vs validation), validation macro AUC, per-label validation AUC at the latest epoch, status.
The current fold is drawn in colour; finished folds are grey for reference. Validation AUC here is against the
report labels on the fold's report-only studies. Per-fold gold AUC (about 12 studies) is too noisy to plot per epoch;
pooled gold AUC appears in the status panel when the run finishes.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
AXIS = "#cfcec8"
TRAIN = "#2a78d6"  # categorical slot 1 (validated pair with slot 2)
VALID = "#eb6834"  # categorical slot 2
EARLIER = "#bdbcb6"  # finished folds, de-emphasised


class ProgressTracker:
    def __init__(self, run_dir, labels, folds, epochs, title="", open_in_editor=True):
        self.dir, self.labels, self.folds, self.epochs, self.title = Path(run_dir), list(labels), list(folds), epochs, title
        self.png, self.jsonl = self.dir / "progress.png", self.dir / "history.jsonl"
        self.hist, self.t0, self.final = [], time.time(), None
        self.render()
        if open_in_editor:  # VS Code preview auto-reloads when the PNG changes
            try:
                subprocess.run(["code", "-r", str(self.png)], check=False, capture_output=True, timeout=10)
            except (OSError, subprocess.SubprocessError):
                pass

    # ---- recording ----
    def log(self, fold, epoch, train_loss, val_loss, val_auc):
        aucs = [v for v in val_auc.values() if v == v]
        rec = {"fold": fold, "epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
               "val_macro_auc": round(sum(aucs) / len(aucs), 4) if aucs else None, "val_auc": val_auc,
               "elapsed_s": round(time.time() - self.t0, 1)}
        self.hist.append(rec)
        with open(self.jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.render()
        best = self._best(fold)
        print(f"fold {fold} epoch {epoch}/{self.epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"val AUC {rec['val_macro_auc']:.3f} (best {best['val_macro_auc']:.3f} @ {best['epoch']})  "
              f"elapsed {rec['elapsed_s'] / 60:.1f} min  ETA {self._eta() / 60:.1f} min", flush=True)

    def finish(self, **final):
        self.final = final
        self.render()

    def _fold_hist(self, fold):
        return [r for r in self.hist if r["fold"] == fold]

    def _best(self, fold):
        h = [r for r in self._fold_hist(fold) if r["val_macro_auc"] is not None]
        return max(h, key=lambda r: r["val_macro_auc"]) if h else {"val_macro_auc": float("nan"), "epoch": "-"}

    def _eta(self):
        done = len(self.hist)
        total = len(self.folds) * self.epochs
        return (time.time() - self.t0) / done * (total - done) if done else 0.0

    # ---- drawing ----
    def render(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=110, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [1, 1.15]})
        fig.suptitle(f"Training progress · {self.title or self.dir.name}", x=0.06, ha="left", color=TEXT, fontsize=14)
        cur = self.hist[-1]["fold"] if self.hist else (self.folds[0] if self.folds else 0)
        self._curves(axes[0, 0], cur, "Loss (BCE)", [("train_loss", TRAIN, "train"), ("val_loss", VALID, "validation")])
        self._curves(axes[0, 1], cur, "Validation macro AUC (vs report labels)", [("val_macro_auc", VALID, "validation")],
                     chance=True)
        self._bars(axes[1, 0])
        self._status(axes[1, 1], cur)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        tmp = self.png.with_suffix(".tmp.png")
        fig.savefig(tmp, facecolor=SURFACE)
        plt.close(fig)
        os.replace(tmp, self.png)  # atomic: the preview never sees a half-written file

    def _style(self, ax, title):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", color=TEXT, fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=TEXT_2, labelsize=9)
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)

    def _curves(self, ax, cur, title, series, chance=False):
        self._style(ax, title)
        ax.set_xlabel("epoch", color=TEXT_2, fontsize=9)
        ax.set_xlim(0.5, self.epochs + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        drew_earlier = False
        for f in self.folds:
            h = [r for r in self._fold_hist(f) if r[series[-1][0]] is not None]
            if not h or f == cur:
                continue
            ax.plot([r["epoch"] for r in h], [r[series[-1][0]] for r in h], color=EARLIER, linewidth=1.25,
                    label="earlier folds (validation)" if not drew_earlier else None, zorder=1)
            drew_earlier = True
        h = self._fold_hist(cur)
        for key, color, name in series:
            pts = [(r["epoch"], r[key]) for r in h if r[key] is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                    marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=f"{name} (fold {cur})", zorder=3)
            ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        if chance:
            ax.axhline(0.5, color=AXIS, linewidth=1, zorder=0)
            ax.annotate("chance", (0.6, 0.5), xytext=(0, 3), textcoords="offset points", fontsize=8, color=TEXT_2)
            lo = min([0.5] + [r["val_macro_auc"] for r in self.hist if r["val_macro_auc"] is not None])
            ax.set_ylim(lo - 0.02, 1.0)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2, loc="best")

    def _bars(self, ax):
        last = self.hist[-1] if self.hist else None
        self._style(ax, f"Per-label validation AUC · fold {last['fold']}, epoch {last['epoch']}" if last
                    else "Per-label validation AUC")
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.grid(False, axis="y")
        if not last:
            ax.text(0.5, 0.5, "waiting for the first epoch…", ha="center", va="center", color=TEXT_2,
                    transform=ax.transAxes)
            ax.set_yticks([])
            return
        items = sorted(((k, v) for k, v in last["val_auc"].items() if v == v), key=lambda kv: kv[1])
        names, vals = [k for k, _ in items], [v for _, v in items]
        ys = range(len(items))
        ax.barh(ys, [v - 0.5 for v in vals], left=0.5, height=0.6, color=VALID, edgecolor=SURFACE, linewidth=2)
        for y, v in zip(ys, vals):
            ax.annotate(f"{v:.3f}", (v, y), xytext=(4 if v >= 0.5 else -4, 0), textcoords="offset points",
                        va="center", ha="left" if v >= 0.5 else "right", fontsize=8, color=TEXT)
        ax.set_yticks(list(ys), names, fontsize=9, color=TEXT)
        ax.axvline(0.5, color=AXIS, linewidth=1)
        ax.set_xlim(min(0.4, min(vals) - 0.03), 1.0)
        ax.set_xlabel("AUC (bars start at chance, 0.5; weakest at the bottom)", color=TEXT_2, fontsize=9)

    def _status(self, ax, cur):
        ax.axis("off")
        ax.set_facecolor(SURFACE)
        n_done = len(self.hist)
        total = len(self.folds) * self.epochs
        lines = [("Run", self.dir.name)]
        if self.hist:
            last = self.hist[-1]
            fi = self.folds.index(last["fold"]) + 1 if last["fold"] in self.folds else "?"
            best = self._best(cur)
            lines += [("Progress", f"fold {fi} of {len(self.folds)} · epoch {last['epoch']} of {self.epochs}  ({n_done}/{total})"),
                      ("Elapsed", f"{last['elapsed_s'] / 60:.1f} min"),
                      ("Remaining", f"about {self._eta() / 60:.1f} min" if n_done < total else "done"),
                      ("Best this fold", f"val AUC {best['val_macro_auc']:.3f} at epoch {best['epoch']}")]
            finished = [f for f in self.folds if len(self._fold_hist(f)) == self.epochs]
            if finished:
                lines.append(("Finished folds", ", ".join(f"{f}: {self._fold_hist(f)[-1]['val_macro_auc']:.3f}"
                                                            for f in finished)))
        else:
            lines.append(("Progress", "starting…"))
        if self.final:
            lines.append(("", ""))
            for k, v in self.final.items():
                lines.append((k.replace("_", " "), f"{v:.4f}" if isinstance(v, float) else str(v)))
        y = 0.95
        for k, v in lines:
            ax.text(0.0, y, k, color=TEXT_2, fontsize=10, transform=ax.transAxes, va="top")
            ax.text(0.32, y, v, color=TEXT, fontsize=10, transform=ax.transAxes, va="top")
            y -= 0.085


if __name__ == "__main__":  # re-render a finished or interrupted run from its history.jsonl
    import sys

    run = Path(sys.argv[1])
    hist = [json.loads(l) for l in (run / "history.jsonl").read_text().splitlines() if l.strip()]
    metrics = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else {}
    args = metrics.get("args", {})
    folds = [int(x) for x in args["folds"].split(",")] if "folds" in args else sorted({r["fold"] for r in hist})
    epochs = args.get("epochs", max(r["epoch"] for r in hist))
    t = ProgressTracker(run, list(hist[-1]["val_auc"]), folds, epochs, title=args.get("log") or run.name,
                        open_in_editor=False)
    t.hist = hist
    if metrics:
        t.finish(gold_macro_pooled=metrics.get("gold_macro"), report_macro_pooled=metrics.get("report_macro"),
                 n_gold=metrics.get("n_gold"))
    else:
        t.render()
    print(t.png)
