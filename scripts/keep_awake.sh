#!/bin/sh
# Keep the Mac awake with the lid closed while process <pid> runs (at most <hours> if given), then let it sleep again,
# unless another holder still needs it awake. Run with sudo, in the background:
#     sudo nohup scripts/keep_awake.sh <pid> [hours] >/dev/null 2>&1 &
# Holders are files named after their pid in /tmp/rsna-knee-awake, so the training queue (start/3_run_training with
# LID_CLOSED=1) and the real-scan grading window (start/4_run_external) can't switch sleep back on under each other.
# By hand: sudo pmset -a disablesleep 0
PID=$1
HOURS=${2:-0}
DIR=${AWAKE_DIR:-/tmp/rsna-knee-awake}
PMSET=${PMSET:-pmset}  # PMSET=echo for a dry run
mkdir -p "$DIR"
touch "$DIR/$PID"
$PMSET -a disablesleep 1
END=$(( $(date +%s) + HOURS * 3600 ))
while kill -0 "$PID" 2>/dev/null && { [ "$HOURS" = 0 ] || [ "$(date +%s)" -lt "$END" ]; }; do
  sleep "${AWAKE_POLL:-30}"
done
rm -f "$DIR/$PID"
for f in "$DIR"/*; do
  [ -e "$f" ] || continue
  if kill -0 "$(basename "$f")" 2>/dev/null; then exit 0; fi  # another holder is still running
  rm -f "$f"
done
$PMSET -a disablesleep 0
