# ---- kernel entry point (appended to src/knee/features.py by scripts/build_kernels.py) ----
# Reads the preprocess-shard outputs (attached as kernel sources), writes one float16 memmap per shard plus index.csv:
#   feats_shard{k}.npy  [n_slices_in_shard, 1 + GRID*GRID, dim]
#   index.csv           StudyInstanceUID, slot, shard, start, length
# With PCA_K > 0 (bigger encoders) the full features go to /tmp, one PCA is fitted on a sample of slices (as
# scripts/pca_cache.py) and only the projected cache is saved, already split into ~200 MB parts that
# scripts/fetch_features.py downloads directly: feats_shard{k}_part{j}.npy, parts.json, pca.npz.
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
HALF = DEVICE != "cpu"  # float16 only on the GPU; on a CPU (local tests) it is very slow
SLOT_ORDER = ["sag_fs", "cor_fs", "ax_fs", "t1"]
PCA_K = int("{pca_k}")
FULL = Path("/tmp/feats") if PCA_K else OUT
FULL.mkdir(parents=True, exist_ok=True)


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
model = load_encoder(DEVICE, half=HALF)
dim = model.num_features
index = []
for k, d in dirs.items():
    files = sorted(d.glob("*.npz"))[: LIMIT or None]
    slots = pd.read_csv(d.parent / f"slots_shard{k}.csv")  # n_kept per slot -> memmap size without decompressing
    total = int(slots[slots.StudyInstanceUID.isin({f.stem for f in files})].n_kept.sum())
    mm = np.lib.format.open_memmap(FULL / f"feats_shard{k}.npy", mode="w+", dtype=np.float16,
                                   shape=(total, 1 + GRID * GRID, dim))
    pos = 0
    with ThreadPoolExecutor(6) as ex:
        for i, (uid, arrays) in enumerate(ex.map(load_npz, files), 1):
            if not arrays:
                continue
            feats = encode(model, np.concatenate(list(arrays.values())), DEVICE, half=HALF).numpy().astype(np.float16)
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
           "gb": round(sum(f.stat().st_size for f in FULL.glob("feats_shard*.npy")) / 1e9, 2)}
if PCA_K:
    shards = sorted(FULL.glob("feats_shard*.npy"))
    arrs = [np.load(f, mmap_mode="r") for f in shards]
    rng = np.random.default_rng(0)
    per = min(20000 // len(arrs), min(len(a) for a in arrs))  # fewer on tiny test runs
    X = np.concatenate([a[np.sort(rng.choice(len(a), per, replace=False))].reshape(-1, a.shape[-1]).astype(np.float32)
                        for a in arrs])
    mean = X.mean(0)
    _, sv, vt = np.linalg.svd(X - mean, full_matrices=False)
    cum = np.cumsum(sv ** 2) / (sv ** 2).sum()
    comp = vt[:PCA_K].astype(np.float32)
    np.savez(OUT / "pca.npz", mean=mean, components=comp)
    rows = 200_000_000 // (arrs[0].shape[1] * PCA_K * 2)  # ~200 MB per part
    parts = {}
    for f, a in zip(shards, arrs):
        names = []
        for j, i in enumerate(range(0, len(a), rows)):
            x = np.asarray(a[i:i + rows], dtype=np.float32)
            names.append(f"{f.stem}_part{j:03d}.npy")
            np.save(OUT / names[-1], ((x - mean) @ comp.T).astype(np.float16))
        parts[f.stem] = {"shape": [len(a), a.shape[1], PCA_K], "parts": names}
        print(f.stem, "projected into", len(names), "parts", flush=True)
    (OUT / "parts.json").write_text(json.dumps(parts, indent=1))
    summary |= {"pca_k": PCA_K, "variance_kept": round(float(cum[PCA_K - 1]), 4), "fit_slices": per * len(arrs),
                "minutes_total": round((time.time() - t0) / 60, 1)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
