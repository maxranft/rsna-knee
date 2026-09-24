# ---- standalone CPU kernel: split feats_shard{k}.npy from a feature kernel into ~200 MB parts ----
# The Kaggle CLI cannot resume downloads and drops multi-GB files; scripts/fetch_features.py reassembles the parts.
import json
from pathlib import Path

import numpy as np

ROWS = 15000  # 15000 x 17 x 384 x 2 bytes ~= 196 MB
src = next(d for pat in ["*", "*/*", "*/*/*"] for d in Path("/kaggle/input").glob(pat) if (d / "index.csv").exists())
out = Path("/kaggle/working")
parts = {}
for f in sorted(src.glob("feats_shard*.npy")):
    a = np.load(f, mmap_mode="r")
    names = []
    for j, i in enumerate(range(0, len(a), ROWS)):
        name = f"{f.stem}_part{j:03d}.npy"
        np.save(out / name, np.ascontiguousarray(a[i:i + ROWS]))
        names.append(name)
    parts[f.stem] = {"shape": list(a.shape), "parts": names}
    print(f.stem, a.shape, len(names), "parts", flush=True)
(out / "parts.json").write_text(json.dumps(parts, indent=1))
for extra in ["index.csv", "summary.json"]:
    (out / extra).write_bytes((src / extra).read_bytes())
