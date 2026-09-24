"""Build one Kaggle script kernel per shard: src/knee/preprocess.py + kaggle/preprocess/main.py -> build/.

    python scripts/build_preprocess.py            # writes build/preprocess-shard{k}/
    kaggle kernels push -p build/preprocess-shard0
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_SHARDS = 3

module = (ROOT / "src/knee/preprocess.py").read_text()
main = (ROOT / "kaggle/preprocess/main.py").read_text()
for k in range(N_SHARDS):
    d = ROOT / "build" / f"preprocess-shard{k}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "preprocess.py").write_text(module + "\n\n" + main.replace("{shard}", str(k)).replace("{n_shards}", str(N_SHARDS)))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"maxranft/rsna-knee-preprocess-shard{k}", "title": f"rsna-knee-preprocess-shard{k}",
        "code_file": "preprocess.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"], "dataset_sources": [], "kernel_sources": [],
    }, indent=2))
    print("built", d)
