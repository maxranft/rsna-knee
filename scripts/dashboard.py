"""Build the results dashboard -> runs/dashboard.html (one self-contained page; open it in any browser).

    python scripts/dashboard.py                 # build once
    python scripts/dashboard.py --open          # build and open in the default browser
    python scripts/dashboard.py --watch 60      # rebuild every 60 s (Ctrl+C to stop)
    python scripts/dashboard.py --ref runs/<id> # compare against another reference run

train_head.py also rebuilds it (in the background) after every fold and at the end of every run, and the open page
reloads itself every minute while a run is in progress. The reference run is the one marked **Current reference**
in experiments.md. Metrics and aggregates only: no report text, no images.
"""
import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOCK = ROOT / "runs/.dashboard.lock"


def build_locked(ref, quiet):
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 180:
        if not quiet:
            print("another build is running; skipped")
        return None
    LOCK.write_text(str(os.getpid()))
    try:
        from knee import dashboard
        out, secs = dashboard.build(ref)
        if not quiet:
            print(f"built {out.relative_to(ROOT)} in {secs:.1f} s")
        return out
    finally:
        LOCK.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", help="runs/<id> to use as the reference (default: the one experiments.md marks)")
    p.add_argument("--open", action="store_true")
    p.add_argument("--watch", type=int, default=0, help="rebuild every N seconds")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    ref = Path(args.ref).name if args.ref else None
    out = build_locked(ref, args.quiet)
    if args.open and out:
        webbrowser.open(out.as_uri())
    while args.watch:
        time.sleep(args.watch)
        build_locked(ref, True)


if __name__ == "__main__":
    main()
