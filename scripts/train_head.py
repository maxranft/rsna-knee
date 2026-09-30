"""Train the study head on cached features with the fixed folds; report gold and report-label AUC.

    python scripts/train_head.py --feats data/features/dinov2s --labels data/labeller/all-v2/report_scores_all.csv \
        --log "baseline: DINOv2-S cache, 1-layer head"

Targets: gold labels for the 58 gold studies, soft report scores (P(yes)) for everyone else.
Metrics (pooled out-of-fold): gold macro AUC on the 58 gold studies (the arbiter) and macro AUC against the
report labels (score > 0.5) on the report-only studies. Writes runs/<id>/{oof.csv, metrics.json}.
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee.head import Head, Studies, collate  # noqa: E402
from knee.progress import ProgressTracker  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def aucs(oof, ids, truth):
    o = oof.set_index("StudyInstanceUID").loc[ids]
    res = {}
    for l in LABELS:
        y = truth.loc[ids, l].values
        if 0 < y.sum() < len(y):
            res[l] = round(float(roc_auc_score(y, o[l].values)), 4)
    return res


def refresh_dashboard():
    """Rebuild runs/dashboard.html in the background; never blocks or fails training."""
    try:
        import subprocess
        subprocess.Popen([sys.executable, str(ROOT / "scripts/dashboard.py"), "--quiet"], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        pass


def evaluate(model, dl, dev):
    """-> (sigmoid predictions [n, 12] in dataset order, mean BCE against the training targets)."""
    lossf = torch.nn.BCEWithLogitsLoss(reduction="sum")
    model.eval()
    preds, tot, n = [], 0.0, 0
    with torch.no_grad():
        for x, slot, depth, pad, y in dl:
            logits = model(x.to(dev), slot.to(dev), depth.to(dev), pad.to(dev))
            tot += lossf(logits, y.to(dev)).item()
            n += y.numel()
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds), tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", required=True, help="dir with index.csv and feats_shard*.npy")
    p.add_argument("--labels", required=True, help="labeller report_scores_all.csv")
    p.add_argument("--label-map", default="{}", help='JSON {label: column} to use a variant column, e.g. {"Effusion": "Effusion@moderate"}')
    p.add_argument("--canon", default="", help="meta/laterality.csv: mirror left knees into right-knee layout (medial on one side)")
    p.add_argument("--slice-drop", type=float, default=0.0, help="training: drop each slice with this probability")
    p.add_argument("--slot-drop", type=float, default=0.1, help="training: drop each series slot with this probability")
    p.add_argument("--gold-repeat", type=int, default=1, help="training: list each gold study this many times per epoch")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--drop", type=float, default=0.1, help="dropout in the head (input projection, transformer, pooling)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log", default="", help="append a row to experiments.md with this change description")
    p.add_argument("--no-open", action="store_true", help="don't open the live progress.png in VS Code")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))  # lets the results viewer label a run mid-training

    # ---- data ----
    fd = Path(args.feats)
    index = pd.read_csv(fd / "index.csv")
    feats = {int(f.stem.split("shard")[1]): str(f) for f in sorted(fd.glob("feats_shard*.npy"))}
    index = index[index.shard.isin(feats)]  # train on whichever shards are present (e.g. mid-download)
    T, C = np.load(next(iter(feats.values())), mmap_mode="r").shape[1:]
    folds = pd.read_csv(ROOT / "meta/folds.csv")
    train_csv = pd.read_csv(ROOT / "data/raw/train.csv", usecols=lambda c: c != "Report").set_index("StudyInstanceUID")
    scores = pd.read_csv(args.labels).set_index("StudyInstanceUID")
    label_map = {l: l for l in LABELS} | json.loads(args.label_map)
    report = scores[[label_map[l] for l in LABELS]].set_axis(LABELS, axis=1)
    gold_ids = set(folds.StudyInstanceUID[folds.is_gold])
    have = set(index.StudyInstanceUID)
    targets = {u: (train_csv.loc[u, LABELS].values.astype(np.float32) if u in gold_ids else report.loc[u].values.astype(np.float32))
               for u in folds.StudyInstanceUID if u in have and (u in gold_ids or u in report.index)}
    folds = folds[folds.StudyInstanceUID.isin(targets)]
    left = None
    if args.canon:
        lat = pd.read_csv(ROOT / args.canon)
        left = dict(lat.loc[lat.side == "L", ["StudyInstanceUID", "t1_plane"]].itertuples(index=False))
        print(f"canonicalising {len(left)} left knees into right-knee layout", flush=True)
    print(f"device={dev} studies={len(folds)} gold={folds.is_gold.sum()} feats={T}x{C}", flush=True)

    # ---- train / predict per fold ----
    oof = []
    t0 = time.time()
    fold_list = [int(x) for x in args.folds.split(",")]
    tracker = ProgressTracker(out, LABELS, fold_list, args.epochs, title=args.log or "test run", open_in_editor=not args.no_open)
    print(f"live progress: {out / 'progress.png'}", flush=True)
    for f in fold_list:
        tr = folds.StudyInstanceUID[folds.fold != f].tolist()
        tr += [u for u in tr if u in gold_ids] * (args.gold_repeat - 1)  # upweight gold labels by repetition
        va = folds.StudyInstanceUID[folds.fold == f].tolist()
        dl_tr = torch.utils.data.DataLoader(Studies(tr, index, feats, targets, train=True, seed=args.seed + f, left=left,
                                                    slice_drop=args.slice_drop, slot_drop=args.slot_drop), batch_size=args.bs,
                                            shuffle=True, collate_fn=collate, num_workers=args.workers, persistent_workers=args.workers > 0)
        dl_va = torch.utils.data.DataLoader(Studies(va, index, feats, targets, left=left), batch_size=args.bs * 2, collate_fn=collate,
                                            num_workers=args.workers)
        model = Head(T, C, args.d, len(LABELS), args.layers, drop=args.drop).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        steps = args.epochs * len(dl_tr)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
        lossf = torch.nn.BCEWithLogitsLoss()
        for ep in range(args.epochs):
            model.train()
            tot = 0.0
            for x, slot, depth, pad, y in dl_tr:
                loss = lossf(model(x.to(dev), slot.to(dev), depth.to(dev), pad.to(dev)), y.to(dev))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item() * len(y)
            preds, val_loss = evaluate(model, dl_va, dev)
            fold_oof = pd.DataFrame(preds, columns=LABELS).assign(StudyInstanceUID=va)
            va_rep = [u for u in va if u not in gold_ids]
            tracker.log(f, ep + 1, train_loss=tot / len(tr), val_loss=val_loss,
                        val_auc=aucs(fold_oof, va_rep, (report > 0.5).astype(int)))
        torch.save(model.state_dict(), out / f"head_fold{f}.pt")
        oof.append(fold_oof.assign(fold=f))
        refresh_dashboard()
    oof = pd.concat(oof)
    oof.to_csv(out / "oof.csv", index=False)
    (out / "config.json").write_text(json.dumps({"tokens": int(T), "dim": int(C), "d": args.d, "layers": args.layers, "drop": args.drop,
                                                 "labels": LABELS, "label_map": label_map, "feats": str(fd),
                                                 "canon": bool(args.canon)}, indent=1))


    # ---- metrics ----
    gold_eval = [u for u in oof.StudyInstanceUID if u in gold_ids]
    rep_eval = [u for u in oof.StudyInstanceUID if u not in gold_ids]
    gold_auc = aucs(oof, gold_eval, train_csv[LABELS]) if gold_eval else {}
    rep_auc = aucs(oof, rep_eval, (report > 0.5).astype(int))
    m = {"run": run_id, "args": vars(args), "device": dev, "minutes": round((time.time() - t0) / 60, 1),
         "gold_macro": round(float(np.mean(list(gold_auc.values()))), 4) if gold_auc else None, "gold": gold_auc,
         "report_macro": round(float(np.mean(list(rep_auc.values()))), 4), "report": rep_auc,
         "n_gold": len(gold_eval), "n_report": len(rep_eval)}
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    tracker.finish(gold_macro_pooled=m["gold_macro"], report_macro_pooled=m["report_macro"], n_gold=len(gold_eval))
    print(json.dumps(m, indent=1))

    if args.log:
        weakest = ", ".join(f"{k} {v:.3f}" for k, v in sorted(gold_auc.items(), key=lambda kv: kv[1])[:3])
        row = (f"| {date.today()} | {run_id} | S3 | {args.log} | {m['gold_macro']} | {m['report_macro']} | {weakest} | "
               f"train {m['minutes']} min ({dev}) | folds {args.folds}, {args.epochs} ep |\n")
        with open(ROOT / "experiments.md", "a") as fh:
            fh.write(row)
        print("logged to experiments.md")
    refresh_dashboard()


if __name__ == "__main__":  # DataLoader workers are spawned on macOS and re-import this file
    main()
