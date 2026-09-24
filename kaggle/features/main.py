# ---- kernel entry point (appended to src/knee/features.py by scripts/build_kernels.py) ----
# Reads the preprocess-shard outputs (attached as kernel sources), writes one float16 memmap per shard plus index.csv:
#   feats_shard{k}.npy  [n_slices_in_shard, 1 + GRID*GRID, dim]
#   index.csv           StudyInstanceUID, slot, shard, start, length
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("OUT", "/kaggle/working"))
LIMIT = int(os.environ.get("LIMIT", "0"))
DEVICE = os.environ.get("DEVICE", "cuda")
SLOT_ORDER = ["sag_fs", "cor_fs", "ax_fs", "t1"]


def find_stack_dirs():
    found = {}
    for pat in ["*/stacks", "*/*/stacks", "*/*/*/stacks"]:
        for d in Path(os.environ.get("INPUT_ROOT", "/kaggle/input")).glob(pat):
            k = next((int(c[-1]) for c in d.parent.parts if "preprocess-shard" in c), None)
            if k is not None:
                found[k] = d
    return dict(sorted(found.items()))


def load_npz(f):
    z = np.load(f)
    return f.stem, {k: z[k] for k in SLOT_ORDER if k in z.files}


t0 = time.time()
dirs = find_stack_dirs()
print("stack dirs:", {k: str(v) for k, v in dirs.items()}, flush=True)
assert dirs, "attach the rsna-knee-preprocess-shard* kernels as inputs"
model = load_encoder(DEVICE)
dim = model.num_features
index = []
for k, d in dirs.items():
    files = sorted(d.glob("*.npz"))[: LIMIT or None]
    slots = pd.read_csv(d.parent / f"slots_shard{k}.csv")  # n_kept per slot -> memmap size without decompressing
    total = int(slots[slots.StudyInstanceUID.isin({f.stem for f in files})].n_kept.sum())
    mm = np.lib.format.open_memmap(OUT / f"feats_shard{k}.npy", mode="w+", dtype=np.float16,
                                   shape=(total, 1 + GRID * GRID, dim))
    pos = 0
    with ThreadPoolExecutor(6) as ex:
        for i, (uid, arrays) in enumerate(ex.map(load_npz, files), 1):
            if not arrays:
                continue
            feats = encode(model, np.concatenate(list(arrays.values())), DEVICE).numpy()
            mm[pos:pos + len(feats)] = feats
            off = pos
            for s, a in arrays.items():
                index.append({"StudyInstanceUID": uid, "slot": s, "shard": k, "start": off, "length": len(a)})
                off += len(a)
            pos += len(feats)
            if i % 250 == 0 or i == len(files):
                print(f"shard {k}: {i}/{len(files)} studies, {pos} slices, {(time.time() - t0) / 60:.1f} min", flush=True)
    assert pos == total, (k, pos, total)
    mm.flush()
    del mm
pd.DataFrame(index).to_csv(OUT / "index.csv", index=False)
summary = {"encoder": ENCODER, "img": IMG, "grid": GRID, "dim": dim,
           "studies": len({r["StudyInstanceUID"] for r in index}), "slices": sum(r["length"] for r in index),
           "minutes": round((time.time() - t0) / 60, 1),
           "gb": round(sum(f.stat().st_size for f in OUT.glob("feats_shard*.npy")) / 1e9, 2)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
