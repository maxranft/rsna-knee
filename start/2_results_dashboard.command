#!/bin/bash
# RESULTS VISUALS: rebuild the dashboard (runs/dashboard.html) from every run and open it in your browser.
# Overview, the run in progress, every run with its verdict, per-label AUC and ROC curves, the data, efficiency.
# Metrics only. Training also rebuilds it after every fold, and the open page reloads itself every minute
# while a run is in progress.
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/dashboard.py --open
