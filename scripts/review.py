"""Review tool: grade the model's predictions one (study, finding) at a time, in the browser.

    python scripts/review.py            # opens http://127.0.0.1:8765 (local only)
    python scripts/review.py --demo     # synthetic data, to try the controls
    python scripts/review.py --external "external scan"   # a real-world scan (scripts/external_study.py), port 8766

Each item shows the slice with the model's strongest evidence (large), a heat map of where on that slice the
evidence sits, an evidence strip across every slice of every series (up = toward present, down = toward absent),
the model's probability next to the training label and the gold label, a plain-English summary of the evidence,
and the report in English with the sentences about the finding highlighted.

Evidence is occlusion on the study head: how much the finding's score (log-odds) drops when one slice is removed, and,
for the slice on screen, where in the image: first a coarse 4x4 map (each cached region token blanked), then a finer
8x8 map (each 32-pixel block of the slice blanked to its mean intensity and the slice re-encoded by the frozen encoder
on the CPU, ~2 s). The finer map drives the red highlight and outline. It explains this model's decision; it is not a
diagnosis.

Decisions go to meta/reviews.csv with what they imply (the finding's true state, whether the model and the training
label were right). Graded items leave the queue. Images and reports stay on this machine: never paste them into a chat.
"""
import argparse
import io
import json
import re
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee.head import SLOT_NAMES, Head, Studies  # noqa: E402

UI = Path(__file__).resolve().parent / "review_ui"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOT_TITLES = {"sag_fs": "Sagittal fluid-sensitive", "cor_fs": "Coronal fluid-sensitive",
               "ax_fs": "Axial fluid-sensitive", "t1": "T1 (not fluid-sensitive)"}
SLOT_INLINE = {"sag_fs": "sagittal fluid-sensitive", "cor_fs": "coronal fluid-sensitive",
               "ax_fs": "axial fluid-sensitive", "t1": "T1"}  # mid-sentence form
QUESTIONS = {
    "ACL": ("Is the anterior cruciate ligament injured?", "ACL injury"),
    "MCL": ("Is the medial collateral ligament injured?", "MCL injury"),
    "Medial Meniscus": ("Is the medial meniscus torn?", "medial meniscus tear"),
    "Lateral Meniscus": ("Is the lateral meniscus torn?", "lateral meniscus tear"),
    "Medial OA": ("Is there osteoarthritis in the medial compartment?", "medial osteoarthritis"),
    "Lateral OA": ("Is there osteoarthritis in the lateral compartment?", "lateral osteoarthritis"),
    "PF OA": ("Is there osteoarthritis behind the kneecap (patellofemoral)?", "patellofemoral osteoarthritis"),
    "Effusion": ("Is there a joint effusion (more fluid than normal)?", "joint effusion"),
    "Synovitis": ("Is there synovitis?", "synovitis"),
    "Baker's": ("Is there a Baker's cyst?", "Baker's cyst"),
    "Contusion": ("Is there a bone bruise?", "bone bruise"),
    "Fracture": ("Is there a fracture?", "fracture"),
}
_OA_EN = r"osteoarthr|arthros|degenerat|cartilag|chondr|osteophyt|joint space"
_OA_XX = r"|arthrose|artroz|artros|hondro|hrskav|kıkırdak|cartílago|cartilagine|χόνδρ|хрущял|osteofit|ostéophyt"
KEYWORDS = {  # English terms (for translations) + common stems in the dataset's languages (for untranslated text)
    "ACL": r"anterior cruciate|\bACL\b|cruciate|kreuzband|križni|çapraz|crociato|cruzado|croisé|χιαστ|кръстн|крстаст",
    "MCL": r"collateral|\bMCL\b|kollateral|kolateral|collatéral|colateral|collaterale|yan bağ|πλάγι|колатерал",
    "Medial Meniscus": r"menisc|menisk|ménisque|μηνίσκ|мениск",
    "Lateral Meniscus": r"menisc|menisk|ménisque|μηνίσκ|мениск",
    "Medial OA": _OA_EN + _OA_XX,
    "Lateral OA": _OA_EN + _OA_XX,
    "PF OA": _OA_EN + _OA_XX + r"|patell|trochle|femoropatel|retropatel|rotul|rótul|επιγονατ|пател",
    "Effusion": r"effusion|fluid|erguss|izljev|izliv|efüzyon|mayi|versamento|derrame|épanchement|συλλογ|излив|течност",
    "Synovitis": r"synovi|sinovi|hoffa|συνοβ|синов",
    "Baker's": r"baker|popliteal|poplit|бейкър|μπέικερ",
    "Contusion": r"contusi|bruise|(o)?edema|kontuz|ödem|œdème|οίδημ|оток",
    "Fracture": r"fractur|avulsion|segond|broken|\bbreak|fraktur|frattur|kırık|κάταγμ|фрактур|счупв|prijelom|prelom|zlom",
}
REVIEWS = ROOT / "meta/reviews.csv"
STACKS = ROOT / "data/stacks"
TRANSLATIONS = ROOT / "data/labeller/translations/translations.json"
KAGGLE = str(ROOT / ".venv/bin/kaggle")


# ---------------------------------------------------------------- shared helpers
def verdict(p):
    return bool(p >= 0.5)


def consequences(label, model_p, label_p):
    """label_p None: no training label (a real-world scan), so the choice only says whether the model is right."""
    noun = QUESTIONS[label][1]
    m = verdict(model_p)
    state = lambda t: "present" if t else "absent"  # noqa: E731
    if label_p is None:
        return {"accept": {"title": f"Accept: {noun} {state(m)}", "truth": int(m), "detail": "The model is right."},
                "deny": {"title": f"Deny: {noun} {state(not m)}", "truth": int(not m), "detail": "The model is wrong."}}
    l = verdict(label_p)
    return {
        "accept": {"title": f"Accept: {noun} {state(m)}", "truth": int(m),
                   "detail": "The model is right. The training label is wrong." if m != l
                   else "The model and the training label are both right."},
        "deny": {"title": f"Deny: {noun} {state(not m)}", "truth": int(not m),
                 "detail": "The model is wrong. The training label is right." if m != l
                 else "The model and the training label are both wrong."},
    }


NOTICEABLE = 0.05  # log-odds change below which removing a slice doesn't really move the score


def summarise(prob, series):
    """Plain-English account of which slices carried the model's call."""
    flat = [(s["slot"], i, c) for s in series for i, c in enumerate(s["contrib"])]
    if max((abs(t[2]) for t in flat), default=0) < NOTICEABLE:
        return (f"The model says {'present' if verdict(prob) else 'absent'} ({prob:.0%}), but removing any single slice "
                "barely changes its score. The prediction reflects how common this finding is overall more than "
                "anything specific it saw in this knee.")
    says = verdict(prob)
    support = sorted((t for t in flat if (t[2] > 0) == says and t[2] != 0), key=lambda t: -abs(t[2]))
    oppose = sorted((t for t in flat if (t[2] > 0) != says and t[2] != 0), key=lambda t: -abs(t[2]))
    total = sum(abs(t[2]) for t in flat) or 1.0
    top5 = sum(abs(t[2]) for t in sorted(flat, key=lambda t: -abs(t[2]))[:5]) / total
    concentration = top5 / (5 / max(len(flat), 5))  # 1.0 = evidence spread evenly over all slices
    out = [f"The model says {'present' if says else 'absent'} ({prob:.0%})."]
    if not support:
        out.append("No slice pushes clearly in that direction.")
    elif concentration < 2.5:
        out.append("Its evidence is spread thinly across the whole study and no slice stands out, "
                   "so read this as a general impression rather than a specific finding.")
    else:
        slot, i0 = support[0][0], support[0][1]
        idx = sorted(i for s, i, _ in support[:6] if s == slot and abs(i - i0) <= 2)  # only close neighbours
        span = f"slice {idx[0] + 1}" if len(idx) == 1 else f"slices {idx[0] + 1}–{idx[-1] + 1}"
        out.append(f"The strongest evidence for that comes from {SLOT_INLINE[slot]} {span}.")
    if support and oppose and sum(abs(t[2]) for t in oppose) > 0.5 * sum(abs(t[2]) for t in support):
        s, i, _ = oppose[0]
        out.append(f"Some slices push the other way, most of all {SLOT_INLINE[s]} slice {i + 1}.")
    return " ".join(out)


def split_sentences(text):  # same rule as the translation kernel, for reports without a translation
    parts = [p.strip() for p in re.split(r"(?<=[.!?;])\s+|\n+", str(text)) if p.strip()]
    out = []
    for p in parts:
        if out and (len(p) < 4 or not re.search(r"[^\W\d_]{2}", p)):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def report_view(label, pairs, translated):
    rx = re.compile(KEYWORDS[label], re.I)
    sents = [{"src": s, "en": e, "hit": bool(rx.search(e if (translated and e) else s))} for s, e in pairs]
    return {"translated": translated, "sentences": sents, "hits": sum(s["hit"] for s in sents),
            "noun": QUESTIONS[label][1], "english_original": translated and all(s == e for s, e in pairs)}


def to_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- real data
def latest_run():
    runs = sorted(d for d in (ROOT / "runs").glob("*") if (d / "oof.csv").exists() and (d / "config.json").exists())
    if not runs:
        sys.exit("no finished training run in runs/ (needs oof.csv + config.json). Train one with scripts/train_head.py")
    return runs[-1]


def read_reviews():
    return pd.read_csv(REVIEWS) if REVIEWS.exists() else pd.DataFrame(columns=["StudyInstanceUID", "label"])


def stack_path(u):
    return STACKS / "stacks" / f"{u}.npz"


class StackFetcher(threading.Thread):
    """Downloads stacks in queue order: one kaggle call per shard per chunk (each call pages through the kernel's
    file list once, about 40 s, so batching matters)."""

    def __init__(self, studies, shard_of, chunk=40):
        super().__init__(daemon=True)
        self.jobs = {}
        for u in dict.fromkeys(studies):
            if not stack_path(u).exists():
                self.jobs.setdefault(int(shard_of[u]), []).append(u)
        self.chunk, self.errors = chunk, []

    def run(self):
        STACKS.mkdir(parents=True, exist_ok=True)
        for shard, uids in self.jobs.items():
            for i in range(0, len(uids), self.chunk):
                pat = "(" + "|".join(re.escape(u) for u in uids[i:i + self.chunk]) + r")\.npz$"
                r = subprocess.run([KAGGLE, "kernels", "output", f"maxranft/rsna-knee-preprocess-shard{shard}", "-p",
                                    str(STACKS), "--file-pattern", pat, "-q"], capture_output=True, text=True)
                if r.returncode != 0:
                    self.errors.append((r.stdout + r.stderr)[-200:])


class RealSource:
    def __init__(self, run, feats_dir, labels_csv):
        self.run, self.cfg = run, json.loads((run / "config.json").read_text())
        self.oof = pd.read_csv(run / "oof.csv")
        self.scores = pd.read_csv(labels_csv).set_index("StudyInstanceUID")
        train = pd.read_csv(ROOT / "data/raw/train.csv").set_index("StudyInstanceUID")
        self.reports = train.Report
        self.gold = train[LABELS].dropna()
        fd = Path(feats_dir)
        self.index = pd.read_csv(fd / "index.csv")
        self.feat_paths = {int(f.stem.split("shard")[1]): str(f) for f in fd.glob("feats_shard*.npy")}
        self.oof = self.oof[self.oof.StudyInstanceUID.isin(set(self.index.StudyInstanceUID))]
        self.shard_of = self.index.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").shard
        self.models, self.lock, self.stacks, self._tokens = {}, threading.Lock(), {}, {}
        self.feats_dir, self._enc, self._pca, self.enc_lock = fd, None, None, threading.Lock()
        self._tr, self._tr_mtime = {}, None
        self.fetch_translations()

    def fetch_translations(self):
        if TRANSLATIONS.exists():
            return
        def go():
            TRANSLATIONS.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([KAGGLE, "kernels", "output", "maxranft/rsna-knee-translate", "-p", str(TRANSLATIONS.parent),
                            "--file-pattern", r"^translations\.json$", "-q"], capture_output=True)
        threading.Thread(target=go, daemon=True).start()

    def translations(self):
        if TRANSLATIONS.exists() and TRANSLATIONS.stat().st_mtime != self._tr_mtime:
            try:
                self._tr, self._tr_mtime = json.loads(TRANSLATIONS.read_text()), TRANSLATIONS.stat().st_mtime
            except json.JSONDecodeError:  # still downloading
                pass
        return self._tr

    def queue(self, mode, label, n, seed=0):
        rv = read_reviews()
        done = set(zip(rv.StudyInstanceUID, rv.label))
        rows = []
        for r in self.oof.to_dict("records"):
            u = r["StudyInstanceUID"]
            for l in ([label] if label else LABELS):
                if (u, l) in done:
                    continue
                p, rep = float(r[l]), float(self.scores.at[u, self.cfg["label_map"][l]])
                g = int(self.gold.at[u, l]) if u in self.gold.index else None
                rows.append({"study": u, "fold": int(r["fold"]), "label": l, "model_prob": p, "report_score": rep,
                             "gold": g, "gap": abs(p - (g if g is not None else rep))})
        q = pd.DataFrame(rows)
        if q.empty:
            return q
        if mode == "gold":
            q = q[q.gold.notna()].sort_values("gap", ascending=False)
        elif mode == "random":
            q = q.sample(frac=1, random_state=seed)
        else:  # biggest disagreements, rotating through findings so one label can't fill the queue
            q = q.sort_values("gap", ascending=False)
            q["rank"] = q.groupby("label").cumcount()
            q = q.sort_values(["rank", "gap"], ascending=[True, False]).drop(columns="rank")
        q = q.head(n).reset_index(drop=True)
        StackFetcher(q.study, self.shard_of).start()
        return q

    def model(self, fold):
        if fold not in self.models:
            c = self.cfg
            m = Head.from_config(c)
            m.load_state_dict(torch.load(self.run / f"head_fold{fold}.pt", map_location="cpu"))
            self.models[fold] = m.eval()
        return self.models[fold]

    def tokens(self, u):
        if u not in self._tokens:
            ds = Studies([u], self.index, self.feat_paths, {u: np.zeros(len(LABELS), np.float32)})
            x, slot, depth, _ = ds[0]
            self._tokens[u] = (x.float()[None], slot[None], depth[None], ds.rows[u])
            if len(self._tokens) > 16:
                self._tokens.pop(next(iter(self._tokens)))
        return self._tokens[u]

    @torch.no_grad()
    def explain(self, u, fold, label, chunk=32):
        """Occlusion per slice -> (probability, per-series {n, contrib per slice}). contrib = score with every slice
        minus score without this slice (log-odds): positive = the slice pushed toward present."""
        x, slot, depth, rows = self.tokens(u)
        k, m, n = LABELS.index(label), self.model(fold), x.shape[1]
        with self.lock:
            full = m(x, slot, depth, torch.zeros(1, n, dtype=torch.bool))[0, k]
            deltas = []
            for s in range(0, n, chunk):
                idx = list(range(s, min(n, s + chunk)))
                pad = torch.zeros(len(idx), n, dtype=torch.bool)
                pad[range(len(idx)), idx] = True
                out = m(x.expand(len(idx), -1, -1, -1), slot.expand(len(idx), -1), depth.expand(len(idx), -1), pad)[:, k]
                deltas.append(full - out)
        contrib = torch.cat(deltas).numpy()
        series, pos = [], 0
        for r in rows:
            series.append({"slot": r["slot"], "title": SLOT_TITLES[r["slot"]], "n": r["length"],
                           "contrib": contrib[pos:pos + r["length"]].round(5).tolist()})
            pos += r["length"]
        return float(torch.sigmoid(full)), series

    @torch.no_grad()
    def region(self, u, fold, label, slot_name, idx):
        """Occlusion per image region of one slice -> 16 values (4x4, row-major): score minus score with that region
        replaced by the slice's average region."""
        x, slot, depth, rows = self.tokens(u)
        t = sum(r["length"] for r in rows[:[r["slot"] for r in rows].index(slot_name)]) + idx
        k, m, n = LABELS.index(label), self.model(fold), x.shape[1]
        xb = x.expand(16, -1, -1, -1).clone()
        neutral = x[0, t, 1:].mean(0)
        xb[torch.arange(16), t, torch.arange(16) + 1] = neutral
        with self.lock:
            full = m(x, slot, depth, torch.zeros(1, n, dtype=torch.bool))[0, k]
            out = m(xb, slot.expand(16, -1), depth.expand(16, -1), torch.zeros(16, n, dtype=torch.bool))[:, k]
        return (full - out).numpy().round(5).tolist()

    def encoder(self):
        """Frozen encoder in float32 on the CPU (training may be using the GPU), plus the run's PCA projection."""
        if self._enc is None:
            from knee.features import load_encoder
            torch.set_num_threads(4)
            self._enc = load_encoder("cpu", half=False)
            if self.cfg["dim"] != self._enc.num_features:
                z = np.load(Path(self.feats_dir) / "pca.npz")
                self._pca = (torch.as_tensor(z["mean"]), torch.as_tensor(z["components"]))
        return self._enc

    @torch.no_grad()
    def fine_region(self, u, fold, label, slot_name, idx, n=8):
        """Pixel occlusion on one slice -> n*n values (row-major), or None while its images download: score with the
        slice as is minus score with that block of the image set to the slice's mean intensity, re-encoded.
        Positive = the block pushed toward present. Re-encoding the untouched slice reproduces the cached features
        (cosine 1.0000, checked 2026-10-01)."""
        st = self.stack(u)
        if st is None:
            return None
        from knee.features import encode
        img = st[slot_name][int(idx)]
        b = img.shape[0] // n
        batch = np.repeat(img[None], n * n + 1, 0)
        fill = np.uint8(round(float(img.mean())))
        for k in range(n * n):
            r, c = divmod(k, n)
            batch[k + 1, r * b:(r + 1) * b, c * b:(c + 1) * b] = fill
        with self.enc_lock:
            enc = self.encoder()
            tok = encode(enc, batch, "cpu", batch=len(batch), half=False)
        if self._pca is not None:
            tok = (tok - self._pca[0]) @ self._pca[1].T
        x, slot, depth, rows = self.tokens(u)
        t = sum(r["length"] for r in rows[:[r["slot"] for r in rows].index(slot_name)]) + int(idx)
        k, m, N = LABELS.index(label), self.model(fold), x.shape[1]
        xb = x.expand(len(batch), -1, -1, -1).clone()
        xb[:, t] = tok
        with self.lock:
            out = m(xb, slot.expand(len(batch), -1), depth.expand(len(batch), -1), torch.zeros(len(batch), N, dtype=torch.bool))[:, k]
        return (out[0] - out[1:]).numpy().round(5).tolist()

    def report(self, u):
        tr = self.translations().get(u)
        if tr:
            return [(s, e if e else s) for s, e in tr], True
        return [(s, None) for s in split_sentences(self.reports.get(u, ""))], False

    def stack(self, u):
        if u not in self.stacks:
            f = stack_path(u)
            try:
                self.stacks[u] = dict(np.load(f)) if f.exists() else None
            except Exception:  # still being written
                return None
            if self.stacks[u] is None:
                del self.stacks[u]
                return None
            if len(self.stacks) > 12:
                self.stacks.pop(next(iter(self.stacks)))
        return self.stacks[u]

    def ready(self, studies):
        return sum(stack_path(u).exists() for u in set(studies))


# ---------------------------------------------------------------- demo data (synthetic, no patient data)
class DemoSource:
    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.run = Path("demo")

    def queue(self, mode, label, n, seed=0):
        rows = [("ACL", 0.91, 0.08, None), ("Fracture", 0.04, 1.0, None), ("Effusion", 0.12, 0.95, 1),
                ("Lateral Meniscus", 0.55, 0.60, None)]
        return pd.DataFrame([{"study": f"demo.{i:012d}", "fold": 0, "label": l, "model_prob": p, "report_score": r,
                              "gold": g, "gap": abs(p - r)} for i, (l, p, r, g) in enumerate(rows) if not label or l == label])

    def explain(self, u, fold, label):
        rng = np.random.default_rng(sum(map(ord, u + label)))
        series = []
        for j, s in enumerate(SLOT_NAMES):
            n = 18 + 4 * j
            base = np.exp(-((np.arange(n) - rng.integers(n)) ** 2) / 12) * rng.choice([-1, 1])
            c = base * 0.3 + rng.normal(0, 0.03, n)
            series.append({"slot": s, "title": SLOT_TITLES[s], "n": n, "contrib": c.round(5).tolist()})
        return 0.5, series

    def region(self, u, fold, label, slot_name, idx):
        rng = np.random.default_rng(sum(map(ord, u + label + slot_name)) + idx)
        g = np.zeros((4, 4))
        g[rng.integers(4), rng.integers(4)] = rng.choice([-1, 1]) * 0.2
        return (g + rng.normal(0, 0.02, (4, 4))).ravel().round(5).tolist()

    def fine_region(self, u, fold, label, slot_name, idx, n=8):
        rng = np.random.default_rng(sum(map(ord, u + label + slot_name)) + idx)
        cy, cx = rng.uniform(1.5, n - 2.5, 2)
        yy, xx = np.mgrid[:n, :n]
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 1.6) * 0.25
        return (blob - 0.04 * np.exp(-((yy - (n - 1 - cy)) ** 2 + (xx - (n - 1 - cx)) ** 2) / 2) + rng.normal(0, 0.004, (n, n))).ravel().round(5).tolist()

    def report(self, u):
        pairs = [("Primjer rečenice jedan.", "Example sentence one: the anterior cruciate ligament is continuous."),
                 ("Primjer rečenice dva.", "Example sentence two: no joint effusion."),
                 ("Primjer rečenice tri.", "Example sentence three: menisci without signs of tear."),
                 ("Primjer rečenice četiri.", "Example sentence four: small popliteal cyst, 14 mm.")]
        return pairs, True

    def stack(self, u):
        yy, xx = np.mgrid[:256, :256]
        img = lambda k: (np.clip(np.sin(xx / (17 + k)) * np.cos(yy / (23 + k)) * 127 + 128, 0, 255)).astype(np.uint8)  # noqa: E731
        return {s: np.stack([img(i + j) for i in range(18 + 4 * j)]) for j, s in enumerate(SLOT_NAMES)}

    def ready(self, studies):
        return len(set(studies))


# ---------------------------------------------------------------- a real-world scan (scripts/external_study.py)
class MeanHead(torch.nn.Module):
    """Several heads as one module: their mean logit. Occlusion differences are linear in it, so the region maps
    average over the heads the same way the slice evidence does."""

    def __init__(self, heads):
        super().__init__()
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, x, slot, depth, pad):
        return torch.stack([h(x, slot, depth, pad) for h in self.heads]).mean(0)


class ExternalSource(RealSource):
    """One scan from outside the competition, prepared by scripts/external_study.py (<folder>/results/study.npz):
    12 items, one per finding, most unusual first. There is no training label or report text; the truth is what Max
    entered in <folder>/truth.csv from the radiologist's report. Scores and slice evidence come from the report run (all
    heads). Region maps use RealSource's code over one head in five (21 of the packaged 105), about 2 s per fine map
    instead of 8.
    Every item stays in the queue, so the scan can be reopened any time; grades go to <folder>/results/reviews.csv."""

    def __init__(self, folder):
        from external_study import load_heads  # imported here: external_study imports this module
        res = folder / "results"
        if not (res / "study.npz").exists():
            sys.exit(f'nothing prepared in {res}. Run first: python scripts/external_study.py "{folder}"')
        meta = json.loads((res / "study.json").read_text())
        z = np.load(res / "study.npz")
        heads, self.cfg, self._pca, _ = load_heads(Path(meta["heads"]))
        assert self.cfg["labels"] == LABELS, self.cfg["labels"]
        self.folder, self.uid, self.run = folder, folder.name, Path(folder.name)
        self.head = MeanHead(heads[::max(1, len(heads) // 20)]).eval()
        self.prob, self.ev = z["prob"], z["evidence"]
        self.percentile = pd.read_csv(res / "predictions.csv").set_index("finding").percentile
        self.stacks = {self.uid: {s: z[f"stack_{s}"] for s in meta["slots"]}}
        rows = [{"slot": s, "length": len(self.stacks[self.uid][s])} for s in meta["slots"]]
        self._tokens = {self.uid: (torch.from_numpy(z["tokens"]).float()[None], torch.from_numpy(z["slot"]).long()[None],
                                   torch.from_numpy(z["depth"]).float()[None], rows)}
        self.lock, self.enc_lock, self._enc = threading.Lock(), threading.Lock(), None
        threading.Thread(target=lambda: self.enc_lock.acquire() and (self.encoder(), self.enc_lock.release()),
                         daemon=True).start()  # load the encoder now, not on the first fine map
        t = pd.read_csv(folder / "truth.csv", dtype=str) if (folder / "truth.csv").exists() else None
        self.truth = {} if t is None else {r.finding: int(r.present) for r in t.itertuples()
                                           if str(r.present).strip() in {"0", "1"}}

    def queue(self, mode, label, n, seed=0):
        q = pd.DataFrame([{"study": self.uid, "fold": 0, "label": l, "model_prob": float(self.prob[k]),
                           "report_score": np.nan, "gold": self.truth.get(l), "gap": 0.0}
                          for k, l in enumerate(LABELS) if not label or l == label])
        q["order"] = q.label.map(self.percentile)
        return q.sort_values("order", ascending=False).drop(columns="order").head(n).reset_index(drop=True)

    def model(self, fold):
        return self.head

    def tokens(self, u):
        return self._tokens[u]

    def explain(self, u, fold, label):
        k, series, pos = LABELS.index(label), [], 0
        for r in self._tokens[u][3]:
            series.append({"slot": r["slot"], "title": SLOT_TITLES[r["slot"]], "n": r["length"],
                           "contrib": self.ev[pos:pos + r["length"], k].round(5).tolist()})
            pos += r["length"]
        return float(self.prob[k]), series

    def encoder(self):
        if self._enc is None:
            from knee.features import load_encoder
            torch.set_num_threads(4)
            self._enc = load_encoder("cpu", half=False)
        return self._enc

    def report(self, u):
        return [], False

    def stack(self, u):
        return self.stacks.get(u)

    def ready(self, studies):
        return len(set(studies))


# ---------------------------------------------------------------- app state
class App:
    def __init__(self, source, reviews_path, n):
        self.src, self.reviews, self.n = source, reviews_path, n
        self.mode, self.label = "disagree", ""
        self.lock = threading.Lock()
        self.load_queue("disagree", "", n)

    def load_queue(self, mode, label, n):
        with self.lock:
            self.mode, self.label, self.n = mode, label or "", n
            self.q = self.src.queue(mode, label or None, n)
            self.cache, self.decisions, self.flags, self.fine_cache = {}, {}, {}, {}

    def session(self):
        q = self.q
        items = [] if q.empty else [{"i": i, "label": r.label, "decision": self.decisions.get(i)} for i, r in enumerate(q.itertuples())]
        first = next((it["i"] for it in items if it["decision"] is None), len(items))
        return {"run": self.src.run.name, "external": isinstance(self.src, ExternalSource), "mode": self.mode, "label": self.label, "labels": LABELS, "items": items,
                "first_open": first, "tally": self.tally(), "ready": self.src.ready(q.study) if len(q) else 0,
                "studies": int(q.study.nunique()) if len(q) else 0}

    def tally(self):
        d = list(self.decisions.values())
        return {"graded": sum(x != "unsure" for x in d), "label_wrong": sum(f[0] for f in self.flags.values()),
                "model_wrong": sum(f[1] for f in self.flags.values())}

    def item(self, i):
        if i in self.cache:
            return self.cache[i]
        r = self.q.iloc[i]
        prob, series = self.src.explain(r.study, int(r.fold), r.label)
        if isinstance(self.src, DemoSource):
            prob = float(r.model_prob)
        flat = [(s["slot"], k, abs(c)) for s in series for k, c in enumerate(s["contrib"])]
        top = max(flat, key=lambda t: t[2])
        pairs, translated = self.src.report(r.study)
        rep = None if pd.isna(r.report_score) else float(r.report_score)
        out = {"i": i, "n": len(self.q), "study_short": r.study[-10:], "label": r.label,
               "question": QUESTIONS[r.label][0], "noun": QUESTIONS[r.label][1],
               "model": prob, "report_label": rep, "gold": None if pd.isna(r.gold) else int(r.gold),
               "series": series, "start": {"slot": top[0], "idx": top[1]},
               "scale": max(max((t[2] for t in flat), default=0.0), NOTICEABLE),
               "negligible": max((t[2] for t in flat), default=0.0) < NOTICEABLE,
               "summary": summarise(prob, series), "report": report_view(r.label, pairs, translated),
               "choices": consequences(r.label, prob, rep), "decision": self.decisions.get(i)}
        self.cache[i] = out
        return out

    def region(self, i, slot, idx):
        r = self.q.iloc[i]
        return {"grid": self.src.region(r.study, int(r.fold), r.label, slot, idx)}

    def fine(self, i, slot, idx):
        key = (i, slot, int(idx))
        if key not in self.fine_cache:
            r = self.q.iloc[i]
            grid = self.src.fine_region(r.study, int(r.fold), r.label, slot, int(idx))
            if grid is None:
                return None
            self.fine_cache[key] = {"grid": grid, "n": int(round(len(grid) ** 0.5))}
            if len(self.fine_cache) > 300:
                self.fine_cache.pop(next(iter(self.fine_cache)))
        return self.fine_cache[key]

    def image(self, i, slot, idx):
        st = self.src.stack(self.q.iloc[i].study)
        if st is None:
            return None
        return to_png(st[slot][int(np.clip(idx, 0, len(st[slot]) - 1))])

    def decide(self, i, decision):
        it, r = self.item(i), self.q.iloc[i]
        m, rep = verdict(it["model"]), it["report_label"]
        truth = it["choices"][decision]["truth"] if decision in ("accept", "deny") else None
        label_wrong = int(truth is not None and rep is not None and truth != verdict(rep))
        model_wrong = int(truth is not None and truth != m)
        row = {"reviewed_at": datetime.now().isoformat(timespec="seconds"), "run": self.src.run.name,
               "StudyInstanceUID": r.study, "fold": int(r.fold), "label": r.label, "model_prob": round(it["model"], 4),
               "report_score": "" if rep is None else round(rep, 4), "gold": "" if it["gold"] is None else it["gold"],
               "decision": decision, "truth": "" if truth is None else truth, "model_right": "" if truth is None else 1 - model_wrong,
               "label_right": "" if truth is None or rep is None else 1 - label_wrong,
               "strongest_slot": it["start"]["slot"], "strongest_slice": it["start"]["idx"],
               "translated": int(it["report"]["translated"])}
        with self.lock:
            old = pd.read_csv(self.reviews) if self.reviews.exists() else pd.DataFrame()
            self.reviews.parent.mkdir(parents=True, exist_ok=True)
            pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(self.reviews, index=False)
            self.decisions[i] = decision
            self.flags[i] = (label_wrong, model_wrong)
            self.cache.get(i, {})["decision"] = decision
        return {"tally": self.tally(), "saved": it["choices"][decision]["title"] if truth is not None else "Marked unsure"}


# ---------------------------------------------------------------- http
def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            p = urlparse(self.path).path
            try:
                if p in ("/", "/index.html"):
                    return self.send(200, (UI / "index.html").read_bytes(), "text/html; charset=utf-8")
                if p.startswith("/static/"):
                    f = (UI / p[len("/static/"):]).resolve()
                    if UI.resolve() in f.parents and f.exists():
                        ctype = {"css": "text/css", "js": "text/javascript"}.get(f.suffix[1:], "application/octet-stream")
                        return self.send(200, f.read_bytes(), ctype + "; charset=utf-8")
                if p == "/api/session":
                    return self.send(200, app.session())
                m = re.fullmatch(r"/api/item/(\d+)", p)
                if m:
                    return self.send(200, app.item(int(m[1])))
                m = re.fullmatch(r"/api/region/(\d+)/(\w+)/(\d+)", p)
                if m:
                    return self.send(200, app.region(int(m[1]), m[2], int(m[3])))
                m = re.fullmatch(r"/api/fine/(\d+)/(\w+)/(\d+)", p)
                if m:
                    d = app.fine(int(m[1]), m[2], int(m[3]))
                    return self.send(200, d) if d is not None else self.send(202, {"pending": True})
                m = re.fullmatch(r"/api/image/(\d+)/(\w+)/(\d+)\.png", p)
                if m:
                    png = app.image(int(m[1]), m[2], int(m[3]))
                    if png is None:
                        q = app.q
                        return self.send(202, {"pending": True, "ready": app.src.ready(q.study), "studies": int(q.study.nunique())})
                    return self.send(200, png, "image/png")
                self.send(404, {"error": f"no route {p}"})
            except Exception as e:  # show the reason in the page instead of a dead request
                self.send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            p = urlparse(self.path).path
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if p == "/api/decision":
                    return self.send(200, app.decide(int(body["i"]), body["decision"]))
                if p == "/api/queue":
                    app.load_queue(body.get("mode", "disagree"), body.get("label", ""), int(body.get("n", app.n)))
                    return self.send(200, app.session())
                self.send(404, {"error": f"no route {p}"})
            except Exception as e:
                self.send(500, {"error": f"{type(e).__name__}: {e}"})
    return H


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="runs/<id> to review (default: latest finished run)")
    p.add_argument("--feats", help="feature cache (default: the one the run was trained on, from its config.json)")
    p.add_argument("--labels", default=str(ROOT / "data/labeller/all-v2/report_scores_all.csv"))
    p.add_argument("--n", type=int, default=50, help="queue length")
    p.add_argument("--port", type=int, help="default 8765, or 8766 with --external (both can run at once)")
    p.add_argument("--demo", action="store_true", help="synthetic data, to try the controls")
    p.add_argument("--external", type=Path, help='a real-world scan prepared by scripts/external_study.py ("external scan")')
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    if args.demo:
        app = App(DemoSource(), ROOT / "runs/demo_reviews.csv", args.n)
    elif args.external:
        folder = args.external.resolve()
        app = App(ExternalSource(folder), folder / "results/reviews.csv", args.n)
    else:
        run = Path(args.run) if args.run else latest_run()
        feats = args.feats or str(ROOT / json.loads((run / "config.json").read_text())["feats"])
        app = App(RealSource(run, feats, args.labels), REVIEWS, args.n)
    port = args.port or (8766 if args.external else 8765)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{port}"
    print(f"review tool on {url} (run {app.src.run.name}, {len(app.q)} items). Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped. Decisions are in {app.reviews}")


if __name__ == "__main__":
    main()
