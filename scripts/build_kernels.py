"""Build Kaggle script kernels into build/<name>/. A kernel = a module from src/knee + an entry point from
kaggle/<x>/main.py, glued into one file (Kaggle script kernels upload a single file).

    python scripts/build_kernels.py
    kaggle kernels push -p build/rsna-knee-preprocess-shard0
    kaggle kernels push -p build/rsna-knee-labeller-gold --accelerator NvidiaTeslaT4   # GPU kernels
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_SHARDS = 3

SHARD_KERNELS = [f"maxranft/rsna-knee-preprocess-shard{k}" for k in range(N_SHARDS)]
KERNELS = [  # (name, module, entry point, {placeholder: value}, gpu, internet, kernel_sources)
    *[(f"rsna-knee-preprocess-shard{k}", "src/knee/preprocess.py", "kaggle/preprocess/main.py",
       {"{shard}": str(k), "{n_shards}": str(N_SHARDS)}, False, False, []) for k in range(N_SHARDS)],
    *[(f"rsna-knee-labeller-{mode}", "src/knee/report_labeller.py", "kaggle/labeller/main.py",
       {"{mode}": mode}, True, True, []) for mode in ("gold", "all")],
    ("rsna-knee-features-dinov2s", "src/knee/features.py", "kaggle/features/main.py", {}, True, True, SHARD_KERNELS),
    ("rsna-knee-features-dinov2s-parts", [], "kaggle/rechunk/main.py", {}, False, False, ["maxranft/rsna-knee-features-dinov2s"]),
    ("rsna-knee-assets", "src/knee/features.py", "kaggle/assets/main.py", {}, False, True, []),
    ("rsna-knee-submit", ["src/knee/preprocess.py", "src/knee/features.py", "src/knee/head.py"], "kaggle/submit/main.py",
     {}, True, False, ["maxranft/rsna-knee-assets"]),  # + dataset maxranft/rsna-knee-heads, added below
]

for name, modules, entry, subs, gpu, internet, sources in KERNELS:
    modules = [modules] if isinstance(modules, str) else modules  # glued in order; names must not clash
    code = "\n\n".join((ROOT / m).read_text() for m in modules) + "\n\n" + (ROOT / entry).read_text()
    for k, v in subs.items():
        code = code.replace(k, v)
    d = ROOT / "build" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.py").write_text(code)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"maxranft/{name}", "title": name, "code_file": "kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": internet, "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": ["maxranft/rsna-knee-heads"] if name == "rsna-knee-submit" else [], "kernel_sources": sources,
    }, indent=2))
    print("built", d.relative_to(ROOT), "(GPU: push with --accelerator NvidiaTeslaT4)" if gpu else "")
