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

Per-label v1: ACL 0.862, MCL 0.927, Med Men 0.901, Lat Men 0.865, Med OA 0.949, Lat OA 0.795, PF OA 0.870,
Effusion 0.711, Synovitis 0.716, Baker's 0.889, Contusion 0.835, Fracture 0.904.

Full run estimate (4,407 reports × 15 questions): about 35 min of prompts + 4 min load, so about 0.7 GPU h.
