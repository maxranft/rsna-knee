"""Compare runs side by side: per-label gold and report AUC, plus a paired bootstrap of the macro-AUC difference.
Same numbers as notebooks/results.ipynb (both use src/knee/results.py).

    python scripts/compare_runs.py runs/<baseline> runs/<candidate> [more runs...]

Every later run is compared with the first. The bootstrap resamples studies (the same resample for both runs), so it
shows how much of a difference could be study-sampling noise; with 58 gold studies the interval is wide.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knee import results as R  # noqa: E402

ids = [Path(r).name for r in sys.argv[1:]]
runs = R.load_runs(include_smoke=True)
t = pd.DataFrame({f"{w} {r[-6:]}": pd.Series(runs.loc[r, f"{w}_by_label"]) for r in ids for w in ("gold", "report")})
t = t.reindex(R.LABELS)
t.loc["MACRO"] = t.mean()
print(t.round(3).to_string())
for r in ids[1:]:
    print(f"\n{r} vs {ids[0]}")
    print(R.paired_bootstrap(runs, ids[0], r).round(4).to_string())
