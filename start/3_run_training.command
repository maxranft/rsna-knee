#!/bin/bash
# RUN TRAINING ("run training" to Claude): train more models in the background, one at a time, keep the Mac awake
# while they run, and show the live log here. It trains the current best config: the auto-improve base
# (runs/auto/state.json), so a change that scripts/auto_improve.py adopts is picked up here too.
#
#   start/3_run_training.command                     # 3 new seeds (about 45 min per seed plugged in)
#   SEEDS="90 91" start/3_run_training.command        # choose the seeds
#   start/3_run_training.command runs/queue11.sh      # run an existing queue file instead
#   LID_CLOSED=1 start/3_run_training.command         # also switch macOS sleep off until the queue ends, so the lid
#                                                     # can be closed (your admin password; desk only, ideally
#                                                     # plugged in). Works on a queue that is already running too.
#
# Ctrl+C (or closing this window) stops watching, not training. To stop training too:
#   pkill -f 'runs/queue'; pkill -f scripts/train_head.py
# Each finished run is logged in experiments.md and appears on the dashboard (start/2_results_dashboard.command).
cd "$(dirname "$0")/.." || exit 1

lid_closed() {  # $1 = pid of the queue; scripts/keep_awake.sh switches sleep back on when it ends
  [ -n "$LID_CLOSED" ] || return 0
  if [ -z "$1" ]; then echo "No training queue found to keep the Mac awake for."; return 0; fi
  echo "Switching sleep off until the queue ends, so you can close the lid. macOS will ask for your password."
  if ! sudo -v; then echo "No password given: a closed lid will pause training."; return 0; fi
  sudo nohup "$PWD/scripts/keep_awake.sh" "$1" > /dev/null 2>&1 &
  sleep 2
  if pmset -g | grep -q "SleepDisabled[[:space:]]*1"; then
    echo "Sleep is off. It switches back on when the queue ends (or now with: sudo pmset -a disablesleep 0)."
  else
    echo "Could not confirm sleep is off; check with: pmset -g | grep SleepDisabled"
  fi
}

if pgrep -f scripts/train_head.py >/dev/null; then
  echo "A training run is already going. Only one at a time: two fill the Mac's memory and both stall."
  echo "Showing its live log instead."
  # the queue process itself ("/bin/bash runs/queue....sh"), not a log watcher that mentions its name
  lid_closed "$(pgrep -f '^/bin/bash [^ ]*runs/queue[^ ]*\.sh$' | head -1)"
  exec scripts/watch_training.sh
fi
pmset -g batt | grep -q "AC Power" || echo "WARNING: on battery. Low Power Mode makes training about 3x slower, and a closed lid pauses it. Plug in."

if [ -n "$1" ]; then
  Q="$1"
else
  CONFIG=$(.venv/bin/python -c "
import shlex, sys
sys.path.insert(0, 'scripts')
import auto_improve
base = auto_improve.load_state()[1]['base']
print(' '.join(k if v is True else k + ' ' + shlex.quote(str(v)) for k, v in base.items()))") || exit 1
  SEEDS=${SEEDS:-$(.venv/bin/python -c "
import glob, json
s = [json.load(open(f)).get('seed') for f in glob.glob('runs/*/args.json')]
m = max([x for x in s if isinstance(x, int)] or [42])
print(' '.join(str(m + i) for i in (1, 2, 3)))")}
  Q=runs/queue_$(date +%Y%m%d-%H%M%S).sh
  cat > "$Q" <<EOF
#!/bin/bash
# Written by start/3_run_training.command: current best config, seeds $SEEDS, one training at a time.
cd "$PWD"
PY=.venv/bin/python
COMMON=($CONFIG --workers 0 --no-open)
for SEED in $SEEDS; do
  \$PY scripts/train_head.py "\${COMMON[@]}" --seed \$SEED --log "ensemble member: current best config, seed \$SEED" > runs/best_s\$SEED.log 2>&1
done
EOF
  echo "Queue: $Q (current best config, seeds $SEEDS)"
fi

chmod +x "$Q"
nohup "$Q" > "${Q%.sh}.out" 2>&1 &
QPID=$!
nohup caffeinate -is -w $QPID >/dev/null 2>&1 &
echo "Started (pid $QPID). The Mac stays awake until the queue ends, as long as the lid is open."
lid_closed "$QPID"
sleep 2
exec scripts/watch_training.sh "$Q"
