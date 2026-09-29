"""Shrink a feature cache by projecting every token onto its top-k principal components (S3 iteration speed, S4 size).

The DINOv2-S cache is 6.6 GB float16 [slices, 17, 384]; on the 16 GB Mac it does not stay in the page cache next to a
training run, so every epoch re-reads it from disk. At k=128 it is 2.2 GB. One PCA is fitted over all token positions
(CLS + pooled patches) on a random sample of slices; it is unsupervised, so fitting on every training study leaks no
labels. The projection (mean, components) is saved as pca.npz for the submission notebook: x_k = (x - mean) @ components.T

    python scripts/pca_cache.py data/features/dinov2s data/features/dinov2s-pca128 --k 128
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--sample", type=int, default=20000, help="slices sampled to fit the PCA")
    p.add_argument("--chunk", type=int, default=8192)
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("feats_shard*.npy"))
    arrs = [np.load(f, mmap_mode="r") for f in shards]

    rng = np.random.default_rng(0)
    per = args.sample // len(arrs)
    X = np.concatenate([a[np.sort(rng.choice(len(a), per, replace=False))].reshape(-1, a.shape[-1]).astype(np.float32)
                        for a in arrs])
    mean = X.mean(0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    cum = np.cumsum(var)
    print("variance kept:", {k: round(float(cum[k - 1]), 4) for k in (32, 64, 96, 128, 192, 256)}, flush=True)
    comp = vt[:args.k].astype(np.float32)
    np.savez(dst / "pca.npz", mean=mean, components=comp)

    for f, a in zip(shards, arrs):
        out = np.lib.format.open_memmap(dst / f.name, mode="w+", dtype=np.float16, shape=(*a.shape[:2], args.k))
        for i in range(0, len(a), args.chunk):
            x = np.asarray(a[i:i + args.chunk], dtype=np.float32)
            out[i:i + len(x)] = ((x - mean) @ comp.T).astype(np.float16)
        out.flush()
        del out
        print(f.name, "done", flush=True)
    shutil.copy(src / "index.csv", dst / "index.csv")
    (dst / "summary.json").write_text(json.dumps({"from": str(src), "k": args.k, "variance_kept": round(float(cum[args.k - 1]), 4),
                                                  "fit_slices": per * len(arrs)}, indent=1))


if __name__ == "__main__":
    main()
