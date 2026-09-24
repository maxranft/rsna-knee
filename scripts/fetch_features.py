"""Download a feature cache that a rechunk kernel split into ~200 MB parts, then reassemble feats_shard{k}.npy.
The Kaggle CLI drops multi-GB downloads and cannot resume, so each part is fetched separately with retries.
Re-running skips parts (and shards) that are already complete.

    python scripts/fetch_features.py maxranft/rsna-knee-features-dinov2s-parts data/features/dinov2s
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = str(ROOT / ".venv/bin/kaggle")
kernel, dest = sys.argv[1], Path(sys.argv[2])
parts_dir = dest / "parts"
parts_dir.mkdir(parents=True, exist_ok=True)


def fetch(pattern, where, tries=6):
    for t in range(tries):
        r = subprocess.run([KAGGLE, "kernels", "output", kernel, "-p", str(where), "--file-pattern", pattern, "-o", "-q"],
                           capture_output=True, text=True)
        if r.returncode == 0 and "Connection broken" not in r.stdout + r.stderr:
            return True
        print(f"  retry {t + 1} for {pattern}: {(r.stdout + r.stderr).strip()[-120:]}", flush=True)
        time.sleep(5 * (t + 1))
    return False


def valid(f, rows=None):
    try:
        a = np.load(f, mmap_mode="r")
        return rows is None or len(a) == rows
    except Exception:
        return False


fetch(r"^(parts\.json|index\.csv|summary\.json)$", dest)
meta = json.loads((dest / "parts.json").read_text())
for stem, m in meta.items():
    shape, target = tuple(m["shape"]), dest / f"{stem}.npy"
    if valid(target) and np.load(target, mmap_mode="r").shape == shape:
        print(stem, "already complete", flush=True)
        continue
    for name in m["parts"]:
        f = parts_dir / name
        if not valid(f):
            f.unlink(missing_ok=True)
            assert fetch(f"^{name}$", parts_dir) and valid(f), f"failed to fetch {name}"
    out = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16, shape=shape)
    pos = 0
    for name in m["parts"]:
        a = np.load(parts_dir / name)
        out[pos:pos + len(a)] = a
        pos += len(a)
    assert pos == shape[0], (stem, pos, shape)
    out.flush()
    del out
    for name in m["parts"]:
        (parts_dir / name).unlink()
    print(stem, shape, "assembled", flush=True)
print("done")
