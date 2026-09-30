#!/bin/bash
# Follow the training queue live in a terminal: every run log written by the newest runs/queue*.sh, as epochs finish.
# Ctrl+C stops watching; training carries on in the background.
#   scripts/watch_training.sh              # the newest queue
#   scripts/watch_training.sh runs/queue3.sh
cd "$(dirname "$0")/.." || exit 1
Q=${1:-$(ls -t runs/queue*.sh 2>/dev/null | head -1)}
[ -z "$Q" ] && { echo "no runs/queue*.sh found"; exit 1; }
LOGS=$(grep -oE 'runs/[A-Za-z0-9_$]+\.log' "$Q" | sort -u)
SEEDS=$(grep -oE 'for SEED in [0-9 ]+' "$Q" | head -1 | sed 's/for SEED in //')
FILES=()
for L in $LOGS; do
  if [[ "$L" == *'$SEED'* ]]; then for S in $SEEDS; do FILES+=("${L//\$SEED/$S}"); done; else FILES+=("$L"); fi
done
for F in "${FILES[@]}"; do [ -f "$F" ] || touch "$F"; done
clear
echo "Watching $Q  ($(date '+%H:%M'))  —  Ctrl+C stops watching, not training."
echo "Dashboard: runs/dashboard.html (rebuilt after every fold)."
echo
tail -n 20 -F "${FILES[@]}" 2>/dev/null
