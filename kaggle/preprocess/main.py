# ---- kernel entry point (appended to src/knee/preprocess.py by scripts/build_preprocess.py) ----
import json
import os
import time
from multiprocessing import get_context

import pandas as pd

SHARD, N_SHARDS = int(os.environ.get("SHARD", "{shard}")), int(os.environ.get("N_SHARDS", "{n_shards}"))
LIMIT = int(os.environ.get("LIMIT", "0"))  # >0: smoke test on the first LIMIT studies of the shard
SPLIT = os.environ.get("SPLIT", "train")
OUT = Path(os.environ.get("OUT", "/kaggle/working"))


def find_root():
    if os.environ.get("DATA_ROOT"):
        return Path(os.environ["DATA_ROOT"])
    for p in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/rsna-knee-abnormality-detection")]:
        if (p / "train_series.csv").exists():
            return p
    raise FileNotFoundError("competition data not mounted")


ROOT = find_root()
series = pd.read_csv(ROOT / f"{SPLIT}_series.csv")
by_study = {u: g[["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]].to_dict("records")
            for u, g in series.groupby("StudyInstanceUID")}
studies = sorted(by_study)[SHARD::N_SHARDS]
if LIMIT:
    studies = studies[:LIMIT]
(OUT / "stacks").mkdir(parents=True, exist_ok=True)
print(f"shard {SHARD}/{N_SHARDS}: {len(studies)} studies from {ROOT}", flush=True)


def work(u):
    try:
        arrays, infos = process_study(u, by_study[u], ROOT / f"{SPLIT}_series")
        np.savez_compressed(OUT / "stacks" / f"{u}.npz", **arrays)
        return infos, None
    except Exception as e:  # keep going; failures are listed in errors.json
        return [], f"{u}: {type(e).__name__}: {e}"


t0, rows, errors = time.time(), [], []
with get_context("fork").Pool(os.cpu_count()) as pool:
    for i, (infos, err) in enumerate(pool.imap_unordered(work, studies, chunksize=2), 1):
        rows += infos
        if err:
            errors.append(err)
        if i % 100 == 0 or i == len(studies):
            el = time.time() - t0
            print(f"{i}/{len(studies)} studies  {el / 60:.1f} min  ({i / el:.2f} studies/s)  errors={len(errors)}", flush=True)

slots = pd.DataFrame(rows)
slots.to_csv(OUT / f"slots_shard{SHARD}.csv", index=False)
(OUT / f"errors_shard{SHARD}.json").write_text(json.dumps(errors, indent=1))
size_gb = sum(f.stat().st_size for f in (OUT / "stacks").glob("*.npz")) / 1e9
print(json.dumps({"studies": len(studies), "errors": len(errors), "minutes": round((time.time() - t0) / 60, 1),
                  "stacks_gb": round(size_gb, 2),
                  "slot_coverage": slots.groupby("slot").size().to_dict() if len(slots) else {}}, indent=1))
