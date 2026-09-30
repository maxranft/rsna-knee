# ---- submission entry point (appended to preprocess.py + features.py + head.py by scripts/build_kernels.py) ----
# Offline. Test study -> 4 slot stacks (CPU pool) -> frozen encoder (GPU) -> 5 fold heads -> mean probability.
# The same code paths as training: process_study, encode, Head.
import json
import os
import time
from multiprocessing import get_context

import pandas as pd

t0 = time.time()
INPUT = Path(os.environ.get("INPUT_ROOT", "/kaggle/input"))
OUT = Path(os.environ.get("OUT", "/kaggle/working"))
COMP = next(p for p in [Path(os.environ.get("DATA_ROOT", "/nonexistent")),
                        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                        Path("/kaggle/input/rsna-knee-abnormality-detection")] if (p / "test_series.csv").exists())


def find(pattern):
    for depth in ["*", "*/*", "*/*/*", "*/*/*/*"]:
        hits = sorted(INPUT.glob(f"{depth}/{pattern}"))
        if hits:
            return hits
    raise FileNotFoundError(pattern)


series = pd.read_csv(COMP / "test_series.csv")
sample = pd.read_csv(COMP / "sample_submission.csv")
by_study = {u: g[["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]].to_dict("records")
            for u, g in series.groupby("StudyInstanceUID")}
studies = list(pd.read_csv(COMP / "test.csv").StudyInstanceUID)
LABEL_COLS = [c for c in sample.columns if c != "StudyInstanceUID"]


def work(u):
    try:
        return u, process_study(u, by_study.get(u, []), COMP / "test_series")[0], None
    except Exception as e:  # never fail the submission on one study; it gets the prior (0.5)
        return u, {}, f"{type(e).__name__}: {e}"


# start the CPU pool before CUDA initialises (fork after CUDA init is unsafe)
pool = get_context("fork").Pool(os.cpu_count())
dev = "cuda" if torch.cuda.is_available() else "cpu"
encoder = load_encoder(dev, weights=find("dinov2s_224.pt")[0])
cfg = json.loads(find("config.json")[0].read_text())
assert cfg["labels"] == LABEL_COLS, (cfg["labels"], LABEL_COLS)
heads = []
for f in find("head_fold*.pt"):
    h = Head(cfg["tokens"], cfg["dim"], cfg["d"], len(LABEL_COLS), cfg["layers"])
    h.load_state_dict(torch.load(f, map_location="cpu"))
    heads.append(h.to(dev).eval())
pca = None
if cfg["dim"] != encoder.num_features:  # heads trained on a PCA-reduced cache (scripts/pca_cache.py)
    z = np.load(find("pca.npz")[0])
    pca = (torch.as_tensor(z["mean"], device=dev), torch.as_tensor(z["components"], device=dev))
    assert tuple(pca[1].shape) == (cfg["dim"], encoder.num_features), pca[1].shape
print(f"{len(studies)} studies, {len(heads)} heads, pca {pca is not None}, device {dev}, setup {time.time() - t0:.0f}s",
      flush=True)

rows, errors = [], []
with torch.no_grad():
    for i, (u, arrays, err) in enumerate(pool.imap(work, studies, chunksize=2), 1):
        if err:
            errors.append(f"{u}: {err}")
        if not arrays:
            rows.append([u] + [0.5] * len(LABEL_COLS))
            continue
        names = [s for s in SLOT_NAMES if s in arrays]
        x = encode(encoder, np.concatenate([arrays[s] for s in names]), dev).to(dev)
        if pca is not None:  # same projection as the training cache: (x - mean) @ components.T
            x = ((x.float() - pca[0]) @ pca[1].T).half()
        x = x.unsqueeze(0)
        slot = torch.as_tensor(np.concatenate([np.full(len(arrays[s]), SLOT_NAMES.index(s)) for s in names]), device=dev).unsqueeze(0)
        depth = torch.as_tensor(np.concatenate([np.linspace(0, 1, len(arrays[s])) for s in names]), dtype=torch.float32, device=dev).unsqueeze(0)
        pad = torch.zeros_like(slot, dtype=torch.bool)
        p = torch.stack([torch.sigmoid(h(x, slot, depth, pad)) for h in heads]).mean(0)[0].cpu().numpy()
        rows.append([u] + p.tolist())
        if i % 100 == 0:
            print(f"{i}/{len(studies)}  {(time.time() - t0) / 60:.1f} min", flush=True)
pool.close()

sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABEL_COLS)
sub.to_csv(OUT / "submission.csv", index=False)
print(json.dumps({"studies": len(sub), "errors": len(errors), "minutes": round((time.time() - t0) / 60, 2),
                  "first_errors": errors[:5]}, indent=1))
