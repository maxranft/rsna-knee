#!/bin/bash
# BACKGROUND TRAINING: train more models in the background, one at a time, keep the Mac awake while they run,
# and show the live log here.
#
#   start/3_background_training.command                     # 3 new seeds of the best single-model config
#   SEEDS="60 61" start/3_background_training.command        # choose the seeds
#   start/3_background_training.command runs/queue11.sh      # run an existing queue file instead
#
# Ctrl+C (or closing this window) stops watching, not training. To stop training too:
#   pkill -f 'runs/queue'; pkill -f scripts/train_head.py
# Each finished run is logged in experiments.md and appears on the dashboard (start/2_results_dashboard.command).
cd "$(dirname "$0")/.." || exit 1

if pgrep -f scripts/train_head.py >/dev/null; then
  echo "A training run is already going. Only one at a time: two fill the Mac's memory and both stall."
  echo "Showing its live log instead."
  exec scripts/watch_training.sh
fi
pmset -g batt | grep -q "AC Power" || echo "WARNING: on battery. Plug in, or the Mac may sleep with the lid closed and pause training."

if [ -n "$1" ]; then
  Q="$1"
else
  # Best single-model config so far (experiments.md, 2026-09-30): PCA-128 cache, 16 epochs, dropout 0.3, slot-drop 0.3.
  SEEDS=${SEEDS:-$(.venv/bin/python -c "
import glob, json
s = [json.load(open(f)).get('seed') for f in glob.glob('runs/*/args.json')]
m = max([x for x in s if isinstance(x, int)] or [42])
print(' '.join(str(m + i) for i in (1, 2, 3)))")}
  Q=runs/queue_$(date +%Y%m%d-%H%M%S).sh
  cat > "$Q" <<EOF
#!/bin/bash
# Written by start/3_background_training.command: best config, seeds $SEEDS, one training at a time.
cd "$PWD"
PY=.venv/bin/python
COMMON=(--feats data/features/dinov2s-pca128 --labels data/labeller/all-v2/report_scores_all.csv --label-map '{"Effusion": "Effusion@moderate"}' --workers 0 --no-open --drop 0.3 --epochs 16 --slot-drop 0.3)
for SEED in $SEEDS; do
  \$PY scripts/train_head.py "\${COMMON[@]}" --seed \$SEED --log "ensemble member: best config (16 ep, dropout 0.3, slot-drop 0.3), seed \$SEED" > runs/best_s\$SEED.log 2>&1
done
EOF
  echo "Queue: $Q (best config, seeds $SEEDS; about 20 min per seed)"
fi

chmod +x "$Q"
nohup "$Q" > "${Q%.sh}.out" 2>&1 &
QPID=$!
nohup caffeinate -is -w $QPID >/dev/null 2>&1 &
echo "Started (pid $QPID). The Mac stays awake until the queue ends."
sleep 2
exec scripts/watch_training.sh "$Q"
