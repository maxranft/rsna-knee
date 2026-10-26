"""Assemble the private Kaggle dataset the submission kernel reads (maxranft/rsna-knee-heads) from finished runs.
Nothing is uploaded; the upload command is printed.

    python scripts/package_heads.py runs/<id>                      # one run: 5 fold heads
    python scripts/package_heads.py runs/<a> runs/<b> runs/<c>      # ensemble: 5 heads per run, suffixed _m0, _m1, ...
    python scripts/package_heads.py runs/ens-<id>                   # the members of an ensemble run
    python scripts/package_heads.py runs/mix-<id>                   # an ensemble + its rank_mix.json (scripts/rank_mix.py)
    python scripts/package_heads.py runs/mix-<id> --tta             # + tta.json: leave-one-slot-out TTA in the kernel
    python scripts/package_heads.py runs/mix-<id> --name=rsna-knee-heads-eff   # a second dataset (build/<name>)
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
ARCH = ("tokens", "dim", "d", "layers", "labels", "feats", "aux", "slot_proj", "local")
OPTIONAL = {"aux": [], "slot_proj": False, "local": False}  # absent from older configs
runs, mix = [], None
tta = "--tta" in sys.argv  # the kernel then averages each head over leave-one-slot-out variants (scripts/tta_oof.py)
name = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--name=")), "rsna-knee-heads")  # dataset slug
for a in [a for a in sys.argv[1:] if not a.startswith("--")]:
    d = Path(a)
    if d.name.startswith("mix-"):  # scripts/rank_mix.py: the source's heads plus the rank mix the kernel applies
        mix = json.loads((d / "args.json").read_text())["mix"]
        d = d.parent / json.loads((d / "args.json").read_text())["source"]
    if d.name.startswith("ens-"):
        tta = tta or bool(json.loads((d / "args.json").read_text()).get("tta"))  # scored with TTA -> ship with TTA
        runs += [ROOT / "runs" / m for m in json.loads((d / "args.json").read_text())["members"]]
    else:
        runs.append(d)
cfgs = [json.loads((r / "config.json").read_text()) for r in runs]
cfg = cfgs[0]
for r, c in zip(runs, cfgs):
    same = lambda k: c.get(k, OPTIONAL.get(k)) == cfg.get(k, OPTIONAL.get(k))  # noqa: E731
    assert all(same(k) for k in ARCH), f"{r.name} differs from {runs[0].name} in {[k for k in ARCH if not same(k)]}"
    if c.get("canon"):
        sys.exit(f"{r.name} mirrors left knees (--canon); the submission kernel does not implement laterality yet")

out = ROOT / "build" / name
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
if tta:
    (out / "tta.json").write_text(json.dumps({"leave_one_slot_out": True}))
if mix:
    (out / "rank_mix.json").write_text(json.dumps(mix, indent=1))
(out / "SOURCE_RUN.txt").write_text("\n".join(r.name for r in runs) + "\n")
(out / "dataset-metadata.json").write_text(json.dumps({
    "title": name, "id": f"maxranft/{name}", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
print(f"packaged {', '.join(r.name for r in runs)}: {n_heads} heads, pca {'yes' if pca.exists() else 'no'}, rank mix {mix or 'none'}, tta {tta} -> {out.relative_to(ROOT)}")
print(f"upload (first time):  .venv/bin/kaggle datasets create -p build/{name}")
print(f"upload (new version): .venv/bin/kaggle datasets version -p build/{name} -m '<run id>'")
