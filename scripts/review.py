"""Manual review: grade the model's predictions one (study, finding) at a time and record accept / deny.

    python scripts/review.py                       # latest run, biggest model-vs-label disagreements first
    python scripts/review.py --queue gold          # only the 58 gold studies
    python scripts/review.py --label "Lateral Meniscus" --n 20
    python scripts/review.py --demo                # synthetic data, to try the controls

For each item the window shows
  * the slice the model attended to most for that finding (its "reasoning": where it looked), with an attention
    strip across every slice of every series. Click the strip, or use the arrow keys, to browse slices;
  * the model's probability, the training label from the report labeller, and the gold label when there is one;
  * the report text, which is what the training label was derived from.
Keys: a accept (model is right) · d deny (model is wrong) · s skip · b back · arrows browse · q quit.

Decisions append to meta/reviews.csv (latest decision per study+finding wins); reviewed pairs leave the queue.
Images and reports appear only in this local window. Never paste them into a chat or external tool.
Stacks download from the preprocess kernels in the background (one batched call per shard) into data/stacks/.
"""
import argparse
import json
import re
import subprocess
import sys
import textwrap
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee.head import SLOT_NAMES, Head, Studies  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOT_TITLES = {"sag_fs": "Sagittal fluid-sensitive", "cor_fs": "Coronal fluid-sensitive",
               "ax_fs": "Axial fluid-sensitive", "t1": "T1 (not fluid-sensitive)"}
REVIEWS = ROOT / "meta/reviews.csv"
STACKS = ROOT / "data/stacks"
KAGGLE = str(ROOT / ".venv/bin/kaggle")
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
ATTN, CURRENT = "#2a78d6", "#eb6834"  # validated categorical slots 1 and 2


# ---------------------------------------------------------------- queue
def latest_run():
    runs = sorted(d for d in (ROOT / "runs").glob("*") if (d / "oof.csv").exists() and (d / "config.json").exists())
    if not runs:
        sys.exit("no finished training run in runs/ (needs oof.csv + config.json)")
    return runs[-1]


def reviewed_pairs():
    if not REVIEWS.exists():
        return set()
    r = pd.read_csv(REVIEWS)
    return set(zip(r.StudyInstanceUID, r.label))


def build_queue(run, labels_csv, mode, n, label, seed=0):
    cfg = json.loads((run / "config.json").read_text())
    oof = pd.read_csv(run / "oof.csv")
    scores = pd.read_csv(labels_csv).set_index("StudyInstanceUID")
    train = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c != "Report").set_index("StudyInstanceUID")
    gold = train.dropna(subset=LABELS)
    done = reviewed_pairs()
    rows = []
    for r in oof.to_dict("records"):
        u = r["StudyInstanceUID"]
        for l in ([label] if label else LABELS):
            if (u, l) in done:
                continue
            p = float(r[l])
            rep = float(scores.at[u, cfg["label_map"][l]])
            g = int(gold.at[u, l]) if u in gold.index else None
            rows.append({"StudyInstanceUID": u, "fold": int(r["fold"]), "label": l, "model_prob": p, "report_score": rep,
                         "gold": g, "gap": abs(p - (g if g is not None else rep))})
    q = pd.DataFrame(rows)
    if q.empty:
        return q
    if mode == "gold":
        q = q[q.gold.notna()].sort_values("gap", ascending=False)
    elif mode == "random":
        q = q.sample(frac=1, random_state=seed)
    else:  # disagree: rotate through findings so one label can't fill the queue
        q = q.sort_values("gap", ascending=False)
        q["rank"] = q.groupby("label").cumcount()
        q = q.sort_values(["rank", "gap"], ascending=[True, False]).drop(columns="rank")
    return q.head(n).reset_index(drop=True)


# ---------------------------------------------------------------- data
def stack_path(u):
    return STACKS / "stacks" / f"{u}.npz"


class StackFetcher(threading.Thread):
    """Downloads the queue's stacks in the background: one kaggle call per shard per chunk of studies
    (each call pages through the kernel's file list once, about 40 s, so batching matters)."""

    def __init__(self, studies, index, chunk=40):
        super().__init__(daemon=True)
        shard_of = index.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").shard
        self.jobs = {}
        for u in dict.fromkeys(studies):
            if not stack_path(u).exists():
                self.jobs.setdefault(int(shard_of[u]), []).append(u)
        self.chunk, self.errors = chunk, []

    def run(self):
        STACKS.mkdir(parents=True, exist_ok=True)
        for shard, uids in self.jobs.items():
            for i in range(0, len(uids), self.chunk):
                part = uids[i:i + self.chunk]
                pat = "(" + "|".join(re.escape(u) for u in part) + r")\.npz$"
                r = subprocess.run([KAGGLE, "kernels", "output", f"maxranft/rsna-knee-preprocess-shard{shard}",
                                    "-p", str(STACKS), "--file-pattern", pat, "-q"], capture_output=True, text=True)
                if r.returncode != 0:
                    self.errors.append((r.stdout + r.stderr)[-200:])


class Explainer:
    """Re-runs the study's fold model on its cached features to get each label's attention over slices."""

    def __init__(self, run, feats_dir):
        self.run, self.cfg = run, json.loads((run / "config.json").read_text())
        fd = Path(feats_dir)
        self.index = pd.read_csv(fd / "index.csv")
        self.feat_paths = {int(f.stem.split("shard")[1]): str(f) for f in fd.glob("feats_shard*.npy")}
        self.models = {}

    def model(self, fold):
        if fold not in self.models:
            c = self.cfg
            m = Head(c["tokens"], c["dim"], c["d"], len(LABELS), c["layers"])
            m.load_state_dict(torch.load(self.run / f"head_fold{fold}.pt", map_location="cpu"))
            self.models[fold] = m.eval()
        return self.models[fold]

    @torch.no_grad()
    def explain(self, u, fold, label):
        """-> (probability, attention [N], [(slot, slice index), ...] per token)."""
        ds = Studies([u], self.index, self.feat_paths, {u: np.zeros(len(LABELS), np.float32)})
        x, slot, depth, _ = ds[0]
        logits, attn = self.model(fold)(x[None], slot[None], depth[None], torch.zeros(1, len(x), dtype=torch.bool),
                                        return_attn=True)
        k = LABELS.index(label)
        tokens = [(r["slot"], i) for r in ds.rows[u] for i in range(r["length"])]
        return float(torch.sigmoid(logits[0, k])), attn[0, k].numpy(), tokens


# ---------------------------------------------------------------- window
class Reviewer:
    def __init__(self, queue, explain, load_stack, report_text, run_name, fetcher=None):
        import matplotlib.pyplot as plt
        for k in list(plt.rcParams):
            if k.startswith("keymap."):
                plt.rcParams[k] = []  # our keys, not matplotlib's (s = save, q = quit, arrows = history)
        self.plt, self.q, self.explain, self.load_stack, self.report_text = plt, queue, explain, load_stack, report_text
        self.run_name, self.fetcher, self.i, self.graded = run_name, fetcher, 0, 0
        self.fig = plt.figure(figsize=(15, 8.5), facecolor=SURFACE)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 0.16], hspace=0.28, wspace=0.08,
                                   left=0.03, right=0.98, top=0.93, bottom=0.07)
        self.ax_img, self.ax_strip = self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[1, 0])
        self.ax_txt = self.fig.add_subplot(gs[:, 1])
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.timer = self.fig.canvas.new_timer(interval=1500)
        self.timer.add_callback(self.poll)
        self.load()

    # -- state
    def load(self):
        if self.i >= len(self.q):
            self.item = None
            self.draw()
            return
        it = self.q.iloc[self.i]
        self.item = it
        self.prob, self.attn, self.tokens = self.explain(it.StudyInstanceUID, int(it.fold), it.label)
        self.cur = int(np.argmax(self.attn))
        self.top = self.cur
        self.stack = self.load_stack(it.StudyInstanceUID)
        if self.stack is None:
            self.timer.start()
        self.draw()

    def poll(self):
        if self.item is not None and self.stack is None:
            self.stack = self.load_stack(self.item.StudyInstanceUID)
            if self.stack is not None:
                self.timer.stop()
                self.draw()

    def record(self, decision):
        it = self.item
        slot, idx = self.tokens[self.top]
        row = pd.DataFrame([{"reviewed_at": datetime.now().isoformat(timespec="seconds"), "run": self.run_name,
                             "StudyInstanceUID": it.StudyInstanceUID, "fold": int(it.fold), "label": it.label,
                             "model_prob": round(self.prob, 4), "report_score": round(float(it.report_score), 4),
                             "gold": "" if pd.isna(it.gold) else int(it.gold), "decision": decision,
                             "top_slot": slot, "top_slice": idx}])
        REVIEWS.parent.mkdir(exist_ok=True)
        row.to_csv(REVIEWS, mode="a", header=not REVIEWS.exists(), index=False)
        self.graded += 1

    # -- input
    def on_key(self, ev):
        if self.item is None:
            if ev.key == "q":
                self.plt.close(self.fig)
            return
        if ev.key in ("a", "d"):
            self.record("accept" if ev.key == "a" else "deny")
            self.i += 1
            self.load()
        elif ev.key == "s":
            self.i += 1
            self.load()
        elif ev.key == "b" and self.i > 0:
            self.i -= 1  # re-grading appends a new row; the latest decision wins
            self.load()
        elif ev.key in ("left", "right"):
            s = self.tokens[self.cur][0]
            j = self.cur + (1 if ev.key == "right" else -1)
            if 0 <= j < len(self.tokens) and self.tokens[j][0] == s:
                self.cur = j
                self.draw()
        elif ev.key in ("up", "down"):
            slots = list(dict.fromkeys(t[0] for t in self.tokens))
            s = slots[(slots.index(self.tokens[self.cur][0]) + (1 if ev.key == "down" else -1)) % len(slots)]
            cand = [k for k, t in enumerate(self.tokens) if t[0] == s]
            self.cur = max(cand, key=lambda k: self.attn[k])  # jump to that series' most-attended slice
            self.draw()
        elif ev.key == "q":
            self.plt.close(self.fig)

    def on_click(self, ev):
        if ev.inaxes is self.ax_strip and self.item is not None and ev.xdata is not None:
            self.cur = int(np.clip(round(ev.xdata), 0, len(self.tokens) - 1))
            self.draw()

    # -- drawing
    def draw(self):
        for ax in (self.ax_img, self.ax_strip, self.ax_txt):
            ax.clear()
            ax.set_facecolor(SURFACE)
        self.ax_txt.axis("off")
        if self.item is None:
            self.ax_img.axis("off")
            self.ax_strip.axis("off")
            self.ax_txt.text(0, 0.9, f"Queue finished · {self.graded} graded this session.\nSaved to {REVIEWS.relative_to(ROOT)}"
                             "\n\nPress q to close.", color=TEXT, fontsize=13, va="top", transform=self.ax_txt.transAxes)
            self.fig.canvas.draw_idle()
            return
        it = self.item
        slot, idx = self.tokens[self.cur]
        n_slot = sum(1 for t in self.tokens if t[0] == slot)
        share = self.attn[self.cur] / self.attn.sum()
        self.fig.suptitle(f"{it.label}  ·  item {self.i + 1} of {len(self.q)}  ·  {self.graded} graded this session",
                          x=0.03, ha="left", color=TEXT, fontsize=14)
        # image
        self.ax_img.axis("off")
        if self.stack is None:
            ready = "" if self.fetcher is None else " (batched download from Kaggle, about 1 min per shard)"
            self.ax_img.text(0.5, 0.5, "downloading image…" + ready, ha="center", va="center", color=TEXT_2,
                             transform=self.ax_img.transAxes, wrap=True)
        else:
            self.ax_img.imshow(self.stack[slot][idx], cmap="gray", vmin=0, vmax=255)
        tag = "most-attended slice" if self.cur == self.top else "browsing"
        self.ax_img.set_title(f"{SLOT_TITLES[slot]} · slice {idx + 1} of {n_slot} · {share:.0%} of attention ({tag})",
                              loc="left", color=TEXT, fontsize=10)
        # attention strip: where the model looked, across all series
        xs = np.arange(len(self.tokens))
        self.ax_strip.bar(xs, self.attn, width=0.85, color=ATTN)
        self.ax_strip.bar([self.cur], [self.attn[self.cur]], width=0.85, color=CURRENT)
        starts = [k for k in xs if k == 0 or self.tokens[k][0] != self.tokens[k - 1][0]]
        for k in starts:
            if k:
                self.ax_strip.axvline(k - 0.5, color=TEXT_2, linewidth=0.8)
            self.ax_strip.text(k, 1.02, SLOT_TITLES[self.tokens[k][0]].split(" (")[0], transform=self.ax_strip.get_xaxis_transform(),
                               fontsize=8, color=TEXT_2, va="bottom")
        self.ax_strip.set_xlim(-0.5, len(xs) - 0.5)
        self.ax_strip.set_yticks([])
        self.ax_strip.set_xticks([])
        for s in ("top", "right", "left"):
            self.ax_strip.spines[s].set_visible(False)
        self.ax_strip.spines["bottom"].set_color(GRID)
        self.ax_strip.set_xlabel("attention per slice for this finding (click to view a slice; orange = shown)",
                                 color=TEXT_2, fontsize=8)
        # text panel
        verdict = lambda p: "PRESENT" if p >= 0.5 else "absent"  # noqa: E731
        gold = "—" if pd.isna(it.gold) else verdict(float(it.gold))
        agree = "agree" if verdict(self.prob) == verdict(float(it.report_score)) else "DISAGREE"
        lines = [("Study", "…" + it.StudyInstanceUID[-12:]), ("Finding", it.label),
                 ("Model", f"{self.prob:.0%}  → predicts {verdict(self.prob)}"),
                 ("Training label", f"{float(it.report_score):.0%}  → report says {verdict(float(it.report_score))}"),
                 ("Gold label", gold), ("Model vs report", agree), ("Fold", str(int(it.fold)))]
        y = 0.98
        for k, v in lines:
            self.ax_txt.text(0.0, y, k, color=TEXT_2, fontsize=11, va="top", transform=self.ax_txt.transAxes)
            self.ax_txt.text(0.30, y, v, color=TEXT, fontsize=11, va="top", transform=self.ax_txt.transAxes,
                             fontweight="bold" if k in ("Model", "Model vs report") else "normal")
            y -= 0.045
        report = self.report_text(it.StudyInstanceUID)
        wrapped = textwrap.wrap(" ".join(str(report).split()), 88)
        body = "\n".join(wrapped[:30]) + ("\n…" if len(wrapped) > 30 else "")
        self.ax_txt.text(0.0, y - 0.01, "Report (source of the training label)", color=TEXT_2, fontsize=10, va="top",
                         transform=self.ax_txt.transAxes)
        self.ax_txt.text(0.0, y - 0.045, body, color=TEXT, fontsize=8.5, va="top", family="monospace",
                         transform=self.ax_txt.transAxes)
        self.fig.text(0.03, 0.015, "a accept (model right)   d deny (model wrong)   s skip   b back   "
                      "←/→ slice   ↑/↓ series   click strip   q quit", color=TEXT_2, fontsize=10)
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------- entry
def demo_sources():
    rng = np.random.default_rng(0)
    q = pd.DataFrame([{"StudyInstanceUID": f"demo.{i:012d}", "fold": 0, "label": l, "model_prob": p, "report_score": r,
                       "gold": g, "gap": abs(p - r)} for i, (l, p, r, g) in
                      enumerate([("ACL", 0.91, 0.08, None), ("Effusion", 0.12, 0.95, 1), ("Lateral Meniscus", 0.55, 0.60, None)])])
    tokens = [(s, i) for s in SLOT_NAMES for i in range(20)]

    def explain(u, fold, label):
        a = rng.gamma(0.6, size=len(tokens))
        return float(q.loc[q.StudyInstanceUID == u, "model_prob"].iloc[0]), a / a.sum(), tokens

    def load_stack(u):  # smooth synthetic blobs, not patient data
        yy, xx = np.mgrid[:256, :256]
        img = lambda k: (np.clip(np.sin(xx / (17 + k)) * np.cos(yy / (23 + k)) * 127 + 128, 0, 255)).astype(np.uint8)  # noqa: E731
        return {s: np.stack([img(i + j) for i in range(20)]) for j, s in enumerate(SLOT_NAMES)}

    return q, explain, load_stack, lambda u: "DEMO REPORT. " + "Placeholder text standing in for the radiology report. " * 12


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="runs/<id> to review (default: latest finished run)")
    p.add_argument("--feats", default=str(ROOT / "data/features/dinov2s"))
    p.add_argument("--labels", default=str(ROOT / "data/labeller/all-v2/report_scores_all.csv"))
    p.add_argument("--queue", choices=["disagree", "gold", "random"], default="disagree")
    p.add_argument("--label", choices=LABELS, help="review one finding only")
    p.add_argument("--n", type=int, default=50, help="queue length")
    p.add_argument("--demo", action="store_true", help="synthetic data, to try the controls")
    p.add_argument("--snapshot", help="render the first item to this PNG and exit (layout check; use with --demo)")
    args = p.parse_args()

    if args.snapshot:
        import matplotlib
        matplotlib.use("Agg")
    if args.demo:
        q, explain, load_stack, report_text = demo_sources()
        fetcher, run_name = None, "demo"
        global REVIEWS
        REVIEWS = ROOT / "runs/demo_reviews.csv"
    else:
        run = Path(args.run) if args.run else latest_run()
        q = build_queue(run, args.labels, args.queue, args.n, args.label)
        if q.empty:
            sys.exit("nothing left to review for this selection")
        ex = Explainer(run, args.feats)
        missing = set(q.StudyInstanceUID) - set(ex.index.StudyInstanceUID)
        q = q[~q.StudyInstanceUID.isin(missing)].reset_index(drop=True)
        fetcher = StackFetcher(q.StudyInstanceUID, ex.index)
        fetcher.start()
        reports = pd.read_csv(ROOT / "data/raw/train.csv", usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID").Report

        def load_stack(u):
            f = stack_path(u)
            try:
                return dict(np.load(f)) if f.exists() else None
            except Exception:  # still being written
                return None

        explain, report_text, run_name = ex.explain, lambda u: reports.get(u, ""), run.name
        print(f"run {run.name}: {len(q)} items ({args.queue}); {len(fetcher.jobs)} shard(s) to download", flush=True)

    r = Reviewer(q, explain, load_stack, report_text, run_name, fetcher)
    if args.snapshot:
        r.fig.savefig(args.snapshot, facecolor=SURFACE, dpi=90)
        print("snapshot:", args.snapshot)
        return
    r.plt.show()
    print(f"{r.graded} decisions saved to {REVIEWS}")


if __name__ == "__main__":
    main()
