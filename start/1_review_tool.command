#!/bin/bash
# FRONT-END VISUALS: the review (grading) tool, scripts/review.py, at http://127.0.0.1:8765 in your browser.
# Grade the model's predictions one study and finding at a time; decisions go to meta/reviews.csv.
# It shows MRI images and report text: for you only, never paste what it shows into a chat.
#
#   start/1_review_tool.command                  # the latest finished training run
#   RUN=runs/<id> start/1_review_tool.command    # a specific run
#
# If the tool is already running with the current code, this just opens it; if the code changed since it started,
# it restarts it (graded items stay graded). Ctrl+C in this window stops the tool.
cd "$(dirname "$0")/.." || exit 1
PID=$(lsof -tiTCP:8765 -sTCP:LISTEN 2>/dev/null | head -1)
if [ -n "$PID" ]; then
  if ps -o command= -p "$PID" | grep -q "scripts/review.py"; then
    started=$(date -j -f "%a %b %d %T %Y" "$(ps -o lstart= -p "$PID" | sed 's/  */ /g; s/^ //; s/ *$//')" +%s 2>/dev/null || echo 0)
    newest=$(stat -f %m scripts/review.py scripts/review_ui/* src/knee/head.py src/knee/features.py | sort -n | tail -1)
    if [ -z "$RUN" ] && [ "$started" -ge "$newest" ]; then
      echo "The review tool is already running with the current code; opening http://127.0.0.1:8765"
      open http://127.0.0.1:8765
      exit 0
    fi
    echo "Restarting the review tool (newer code, or a different run was asked for)..."
    kill "$PID"; sleep 1
  else
    echo "Port 8765 is used by another program ($(ps -o comm= -p "$PID")). Close it and try again."
    exit 1
  fi
fi
echo "Starting the review tool (Ctrl+C stops it)..."
exec .venv/bin/python scripts/review.py ${RUN:+--run "$RUN"}
