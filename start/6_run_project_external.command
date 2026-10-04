#!/bin/bash
# RUN PROJECT EXTERNAL ("run project external" to Claude): everything for the real-world scan in "external scan/".
#   1. the results dashboard (start/2_results_dashboard.command), rebuilt and opened in the browser
#   2. the model on the scan (scripts/external_study.py): reads "external scan/dicom/", opens three graphs in Preview
#   3. the grading window for the scan (scripts/review.py --external) at http://127.0.0.1:8766, run from this window
# Training and the competition grading window (port 8765) are left alone. It shows MRI images: for you only, never
# paste what it shows into a chat. Ctrl+C in this window stops the grading window.
#
# Lid closed: macOS sleep is switched off first (your admin password), so all of this keeps running with the lid shut.
# Sleep comes back on when the grading window stops, or after SLEEP_OFF_HOURS (default 3) so a forgotten window can't
# keep the Mac awake in a bag. A running training queue that needs sleep off keeps it off (scripts/keep_awake.sh).
# Desk only, ideally plugged in. By hand: sudo pmset -a disablesleep 0
#
#   start/6_run_project_external.command                      # "external scan"
#   SCAN="other data" start/6_run_project_external.command    # another folder with the same layout
cd "$(dirname "$0")/.." || exit 1
SCAN="${SCAN:-external scan}"
pmset -g batt | grep -q "AC Power" || echo "On battery: with the lid closed the Mac stays awake and uses power. Plugging in is better."
echo "Switching sleep off so this keeps running with the lid closed. macOS will ask for your password."
if sudo -v; then
  # this shell's pid becomes the grading window's at the exec below, so sleep stays off exactly as long as it runs
  sudo nohup "$PWD/scripts/keep_awake.sh" $$ "${SLEEP_OFF_HOURS:-3}" > /dev/null 2>&1 &
else
  echo "No password given: sleep stays on, so closing the lid pauses this until you open it."
fi
echo
.venv/bin/python scripts/dashboard.py --open
echo
if ! .venv/bin/python scripts/external_study.py "$SCAN"; then
  echo
  echo "The model could not run on \"$SCAN\" (see above), so its grading window was not opened."
  exit 1
fi
PID=$(lsof -tiTCP:8766 -sTCP:LISTEN 2>/dev/null | head -1)
if [ -n "$PID" ]; then
  if ps -o command= -p "$PID" | grep -q "scripts/review.py"; then
    kill "$PID"; sleep 1  # restart it on the results just written
  else
    echo "Port 8766 is used by another program ($(ps -o comm= -p "$PID")). Close it and try again."
    exit 1
  fi
fi
echo
echo "Starting the grading window for \"$SCAN\" (Ctrl+C stops it)..."
exec .venv/bin/python scripts/review.py --external "$SCAN"
