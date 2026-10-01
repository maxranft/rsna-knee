#!/bin/bash
# FRONT-END VISUALS: the review tool (scripts/review.py) at http://127.0.0.1:8765 in your browser.
# Grade the model's predictions one study and finding at a time; decisions go to meta/reviews.csv.
# It shows MRI images and report text: for you only, never paste what it shows into a chat.
#
#   start/1_review_tool.command                  # the latest finished training run
#   RUN=runs/<id> start/1_review_tool.command    # a specific run
#
# Ctrl+C in this window stops the tool.
cd "$(dirname "$0")/.." || exit 1
if lsof -tiTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "The review tool is already running; opening http://127.0.0.1:8765"
  open http://127.0.0.1:8765
  exit 0
fi
echo "Starting the review tool (Ctrl+C stops it)..."
exec .venv/bin/python scripts/review.py ${RUN:+--run "$RUN"}
