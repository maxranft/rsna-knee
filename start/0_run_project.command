#!/bin/bash
# RUN PROJECT: everything at once.
#   1. background training (start/3_background_training.command: 3 new seeds of the best config, or shows the run
#      already in progress), in its own window with the live log
#   2. the results dashboard (start/2_results_dashboard.command), rebuilt and opened in the browser
#   3. the grading window (start/1_review_tool.command) at http://127.0.0.1:8765
# Closing the windows stops watching only; training keeps going. To also keep training with the lid closed, run
# start/4_allow_lid_closed.command afterwards.
cd "$(dirname "$0")/.." || exit 1
open -a Terminal start/3_background_training.command
sleep 2
open -a Terminal start/1_review_tool.command
.venv/bin/python scripts/dashboard.py --open
echo
echo "Running: training (its own window), grading tool (its own window, http://127.0.0.1:8765), dashboard (browser)."
