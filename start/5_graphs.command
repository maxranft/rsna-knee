#!/bin/bash
# GRAPHS: every dashboard chart on its own. Saves them as PNGs in graphs/ (open that folder in VS Code; graphs/README.md
# shows them all in the Markdown preview, Shift+Cmd+V) and opens them in a window of their own, which reloads itself
# every minute while a run is in progress. Metrics only.
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/graphs.py --open
