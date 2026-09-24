# Report labeller (S1) log

Code: [../src/knee/report_labeller.py](../src/knee/report_labeller.py) plus the kernel entry point [../kaggle/labeller/main.py](../kaggle/labeller/main.py).
One yes/no question per target; score = P(yes) / (P(yes) + P(no)) from the first answer token.
Metric: per-label AUC of the report score against the 58 gold labels (the ceiling on how clean the training labels are).
Outputs go to `data/labeller/<run>/` (git-ignored). `report_scores_*.csv` holds scores only, no report text.

**Overfitting guard:** 58 studies. Only try a few clinically motivated wordings per label, and don't keep a
variant for a gain under about 0.03 unless the mean scores show a clear mechanism (e.g. the negative mean dropping).

| Run | Date | Model | Change | Macro AUC | Weakest (AUC) | GPU time | Notes |
|---|---|---|---|---|---|---|---|
| gold-v1 | 2026-09-23 | Qwen2.5-7B-Instruct fp16, 2×T4, vLLM | first version | **0.852** | Effusion 0.711, Synovitis 0.716, Lateral OA 0.795 | 257 s (load 230 s, 870 prompts 27 s) | Effusion: gold negatives score 0.57, so small effusions probably count as negative in gold. Synovitis / PF OA: positives score only about 0.5, so the question is too narrow. Laterality is decided in only 38% of reports. |
| gold-v2 | 2026-09-23 | same | + 7 variant wordings for weak labels | 0.8525 (base) | Synovitis 0.717, Effusion 0.713, Lateral OA 0.795 | 249 s | **Effusion@moderate 0.820 (+0.107), adopted**: negatives drop 0.57 → 0.18. ACL@abnormal +0.031 (noise, not adopted, column kept). Broader Synovitis −0.03, Lateral OA −0.016, PF OA −0.039, Contusion −0.091: rejected, gold uses strict definitions. The base questions repeat v1 to ±0.003. |

| all-v2 | 2026-09-24 | same | full run, 4,407 reports × 17 questions | 0.852 (base); **0.861 with Effusion@moderate** | Synovitis 0.714, Lateral OA 0.795 | 2,301 s (0.64 GPU h) | `data/labeller/all-v2/report_scores_all.csv` is the training-label source. Laterality decided in 36% of reports; prior surgery mean 0.005 (rare). |

Per-label v1: ACL 0.862, MCL 0.927, Med Men 0.901, Lat Men 0.865, Med OA 0.949, Lat OA 0.795, PF OA 0.870,
Effusion 0.711, Synovitis 0.716, Baker's 0.889, Contusion 0.835, Fracture 0.904.

Full run estimate (4,407 reports × 15 questions): about 35 min of prompts + 4 min load, so about 0.7 GPU h.

**Current training labels:** v1 questions, with Effusion taken from `Effusion@moderate` (`ADOPTED` in the code).
Gold agreement with that swap: about 0.861 macro.
**Next labeller levers:** a stronger model (Qwen2.5-14B-AWQ or Qwen3-8B) for Synovitis / Lateral OA / Contusion.
Rewording against 58 studies has run its course for those labels.
