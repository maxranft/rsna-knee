"""Teacher predictions for distillation (S4): for each fold k, the mean probability of the fold-k heads of several runs on
that fold's TRAINING studies (folds != k). Those heads never saw fold k, so a student trained on these targets for fold k
stays leak-free (an out-of-fold teacher would leak: its predictions on fold j come from heads trained on fold k).

    python scripts/teacher_preds.py runs/ens-20261005-222134 --out runs/teacher-best23
    -> runs/teacher-best23/teacher_fold{k}.csv (StudyInstanceUID + 12 label probabilities), members.txt
Then: train_head.py ... --teacher runs/teacher-best23 --distill 0.5

Runs on the CPU by default, so it can share the Mac with one MPS training (it is slower there, about 1 s per batch).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from knee.head import Head, Studies, collate  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders, or an ens-* folder (its members)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only this many studies per fold")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    runs = []
    for a in args.runs:
        d = ROOT / a if not Path(a).is_absolute() else Path(a)
        runs += ([ROOT / "runs" / m for m in json.loads((d / "args.json").read_text())["members"]]
                 if d.name.startswith("ens-") else [d])
    cfgs = [json.loads((r / "config.json").read_text()) for r in runs]
    assert len({c["feats"] for c in cfgs}) == 1, "teachers must share one feature cache"
    fd = ROOT / cfgs[0]["feats"]
    index = pd.read_csv(fd / "index.csv")
    feats = {int(f.stem.split("shard")[1]): str(f) for f in sorted(fd.glob("feats_shard*.npy"))}
    folds = pd.read_csv(ROOT / "meta/folds.csv")
    folds = folds[folds.StudyInstanceUID.isin(set(index.StudyInstanceUID))]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "members.txt").write_text("\n".join(r.name for r in runs) + "\n")
    t0 = time.time()
    for f in range(5):
        if (out / f"teacher_fold{f}.csv").exists():
            print(f"fold {f}: done already", flush=True)
            continue
        heads = []
        for r, c in zip(runs, cfgs):
            h = Head.from_config(c)
            h.load_state_dict(torch.load(r / f"head_fold{f}.pt", map_location="cpu"))
            heads.append(h.to(args.device).eval())
        tr = folds.StudyInstanceUID[folds.fold != f].tolist()
        tr = tr[:args.limit] if args.limit else tr
        dummy = {u: np.zeros(len(LABELS), np.float32) for u in tr}
        dl = torch.utils.data.DataLoader(Studies(tr, index, feats, dummy), batch_size=32, collate_fn=collate)
        preds = []
        with torch.no_grad():
            for x, slot, depth, pad, _ in dl:
                x, slot, depth, pad = (t.to(args.device) for t in (x, slot, depth, pad))
                preds.append(torch.stack([torch.sigmoid(h(x, slot, depth, pad)) for h in heads]).mean(0).cpu().numpy())
        pd.DataFrame(np.concatenate(preds), columns=LABELS).assign(StudyInstanceUID=tr)[["StudyInstanceUID"] + LABELS] \
            .to_csv(out / f"teacher_fold{f}.csv", index=False)
        print(f"fold {f}: {len(tr)} studies x {len(heads)} heads  {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
