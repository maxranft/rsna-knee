"""Report AUC per label for right vs left knees (meta/laterality.csv). If a model has not learnt the knee side, the
one-sided findings (menisci, tibiofemoral OA, MCL, Baker's) score lower on one side. Same numbers as notebooks/results.ipynb.

    python scripts/side_split.py runs/<id> [runs/<id> ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knee import results as R  # noqa: E402

runs = R.load_runs(include_smoke=True)
for r in (Path(a).name for a in sys.argv[1:]):
    t = R.side_split(runs, r)
    print(f"== {r} · {runs.loc[r, 'change']}")
    print(t.round(3).to_string())
    print("mean L − R:", t.groupby("group")["L − R"].mean().round(3).to_dict(), "\n")
