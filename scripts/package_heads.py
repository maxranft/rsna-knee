"""Assemble the private Kaggle dataset the submission kernel reads (maxranft/rsna-knee-heads) from finished runs.
Nothing is uploaded; the upload command is printed.

    python scripts/package_heads.py runs/<id>                      # one run: 5 fold heads
    python scripts/package_heads.py runs/<a> runs/<b> runs/<c>      # ensemble: 5 heads per run, suffixed _m0, _m1, ...
    python scripts/package_heads.py runs/ens-<id>                   # the members of an ensemble run
    -> build/rsna-knee-heads/{head_fold<f>[_m<i>].pt, config.json, pca.npz (if PCA), dataset-metadata.json}
Files stay flat: the Kaggle CLI skips subfolders when uploading a dataset.

The kernel (kaggle/submit/main.py) averages the probabilities of every head_fold*.pt it finds (for the 3-seed ensemble
this matches the rank-average scored in experiments.md: report 0.7927 vs 0.7922), and applies pca.npz when the heads'
input width differs from the encoder's (384 for DINOv2-S). All members must share one architecture and feature cache.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ("tokens", "dim", "d", "layers", "labels", "feats")
runs = []
for a in sys.argv[1:]:
    d = Path(a)
    if d.name.startswith("ens-"):
        runs += [ROOT / "runs" / m for m in json.loads((d / "args.json").read_text())["members"]]
    else:
        runs.append(d)
cfgs = [json.loads((r / "config.json").read_text()) for r in runs]
cfg = cfgs[0]
for r, c in zip(runs, cfgs):
    assert all(c.get(k) == cfg.get(k) for k in ARCH), f"{r.name} differs from {runs[0].name} in {[k for k in ARCH if c.get(k) != cfg.get(k)]}"
    if c.get("canon"):
        sys.exit(f"{r.name} mirrors left knees (--canon); the submission kernel does not implement laterality yet")

out = ROOT / "build/rsna-knee-heads"
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)
n_heads = 0
for i, r in enumerate(runs):
    heads = sorted(r.glob("head_fold*.pt"))
    assert len(heads) == 5, f"expected 5 fold heads in {r}, found {len(heads)}"
    for h in heads:
        shutil.copy(h, out / (h.name if len(runs) == 1 else f"{h.stem}_m{i}.pt"))
        n_heads += 1
shutil.copy(runs[0] / "config.json", out / "config.json")
pca = Path(cfg["feats"]) / "pca.npz"
if not pca.is_absolute():
    pca = ROOT / pca
if pca.exists():
    shutil.copy(pca, out / "pca.npz")
(out / "SOURCE_RUN.txt").write_text("\n".join(r.name for r in runs) + "\n")
(out / "dataset-metadata.json").write_text(json.dumps({
    "title": "rsna-knee-heads", "id": "maxranft/rsna-knee-heads", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
print(f"packaged {', '.join(r.name for r in runs)}: {n_heads} heads, pca {'yes' if pca.exists() else 'no'} -> {out.relative_to(ROOT)}")
print("upload (first time):  .venv/bin/kaggle datasets create -p build/rsna-knee-heads")
print("upload (new version): .venv/bin/kaggle datasets version -p build/rsna-knee-heads -m '<run id>'")
