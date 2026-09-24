# ---- kernel entry point (appended to src/knee/report_labeller.py by scripts/build_kernels.py) ----
# Prints metrics only -- never report text.
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODE = os.environ.get("MODE", "{mode}")  # "gold": the 58 gold studies only (fast iteration); "all": every study
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
OUT = Path(os.environ.get("OUT", "/kaggle/working"))

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from vllm import LLM, SamplingParams

ROOT = next(p for p in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                        Path("/kaggle/input/rsna-knee-abnormality-detection")] if (p / "train.csv").exists())
train = pd.read_csv(ROOT / "train.csv")
gold_mask = train[LABELS].notna().all(axis=1)
df = train[gold_mask] if MODE == "gold" else train
print(f"mode={MODE} model={MODEL} reports={len(df)} gpus={torch.cuda.device_count()}", flush=True)

t0 = time.time()
llm = LLM(model=MODEL, dtype="half", tensor_parallel_size=torch.cuda.device_count(), max_model_len=4096,
          gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0)
tok = llm.get_tokenizer()
print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

keys, prompts = [], []
for uid, report in zip(df.StudyInstanceUID, df.Report):
    for key, prompt in build_prompts(report, tok):
        keys.append((uid, key))
        prompts.append(prompt)
t1 = time.time()
outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20))
print(f"{len(prompts)} prompts in {time.time() - t1:.0f}s", flush=True)

scores = {}
for (uid, key), o in zip(keys, outs):
    cand = o.outputs[0].logprobs[0]
    scores.setdefault(uid, {})[key] = yes_probability((c.decoded_token, c.logprob) for c in cand.values())
res = pd.DataFrame.from_dict(scores, orient="index").rename_axis("StudyInstanceUID").reset_index()
res.to_csv(OUT / f"report_scores_{MODE}.csv", index=False)

# ---- agreement with gold (AUC of report score vs gold label) ----
g = res.merge(train.loc[gold_mask, ["StudyInstanceUID"] + LABELS], on="StudyInstanceUID", suffixes=("", "_gold"))
metrics = {}
for lab in LABELS:
    y, s = g[f"{lab}_gold"].values, g[lab].values
    metrics[lab] = {"auc": round(float(roc_auc_score(y, s)), 3), "pos": int(y.sum()),
                    "mean_score_pos": round(float(s[y == 1].mean()), 3),
                    "mean_score_neg": round(float(s[y == 0].mean()), 3),
                    "acc@0.5": round(float(((s > 0.5) == (y == 1)).mean()), 3)}
metrics["macro_auc"] = round(float(np.mean([metrics[l]["auc"] for l in LABELS])), 4)
aux = [k for k in QUESTIONS if k.startswith("aux_")]
metrics["aux_mean_score"] = {k: round(float(res[k].mean()), 3) for k in aux}
metrics["aux_right_vs_left_decided"] = round(float(((res.aux_right_knee > 0.5) ^ (res.aux_left_knee > 0.5)).mean()), 3)
metrics["seconds"] = round(time.time() - t0)
(OUT / f"gold_eval_{MODE}.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps(metrics, indent=1))
