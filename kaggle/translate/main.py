# ---- standalone GPU kernel: sentence-aligned English translations of every training report ----
# For the review tool: each report becomes [[original sentence, English sentence], ...]. Reports that are already
# English are split but not translated. Report text stays inside Kaggle; this prints counts only.
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
OUT = Path(os.environ.get("OUT", "/kaggle/working"))
SYSTEM = ("You translate sentences from knee MRI radiology reports into English. Translate faithfully and precisely: "
          "keep anatomy, laterality, negations (no, without, not) and measurements exactly as written. "
          "If the sentence is already English, repeat it unchanged. Output only the English sentence.")
EN_WORDS = re.compile(r"\b(the|and|of|with|is|are|no|without|there)\b", re.I)


def split_sentences(text):
    parts = [p.strip() for p in re.split(r"(?<=[.!?;])\s+|\n+", str(text)) if p.strip()]
    out = []
    for p in parts:  # glue fragments like "2." or "(LL" back onto the previous sentence
        if out and (len(p) < 4 or not re.search(r"[^\W\d_]{2}", p)):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def is_english(text):
    words = re.findall(r"[^\W\d_]+", str(text))
    return bool(words) and len(EN_WORDS.findall(str(text))) / len(words) > 0.08


ROOT = next(p for p in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                        Path("/kaggle/input/rsna-knee-abnormality-detection")] if (p / "train.csv").exists())
train = pd.read_csv(ROOT / "train.csv", usecols=["StudyInstanceUID", "Report"])
t0 = time.time()
llm = LLM(model=MODEL, dtype="half", tensor_parallel_size=torch.cuda.device_count(), max_model_len=2048,
          gpu_memory_utilization=0.90, seed=0)
tok = llm.get_tokenizer()

result, jobs, prompts = {}, [], []
n_english = 0
for uid, report in zip(train.StudyInstanceUID, train.Report):
    sents = split_sentences(report)
    english = is_english(report)
    n_english += english
    result[uid] = [[s, s if english else None] for s in sents]
    if not english:
        for k, s in enumerate(sents):
            jobs.append((uid, k))
            prompts.append(tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": s}],
                                                   tokenize=False, add_generation_prompt=True))
print(f"{len(train)} reports, {n_english} already English, {len(prompts)} sentences to translate", flush=True)
outs = llm.generate(prompts, SamplingParams(max_tokens=256, temperature=0.0))
for (uid, k), o in zip(jobs, outs):
    result[uid][k][1] = o.outputs[0].text.strip()
(OUT / "translations.json").write_text(json.dumps(result, ensure_ascii=False))
print(json.dumps({"reports": len(result), "english": n_english, "sentences": sum(len(v) for v in result.values()),
                  "translated": len(prompts), "minutes": round((time.time() - t0) / 60, 1)}, indent=1))
