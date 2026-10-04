#!/bin/bash
# LET TRAINING RUN WITH THE LID CLOSED: switch macOS sleep off until the running training queue ends, then switch it
# back on automatically. Normally a closed lid always sleeps the Mac (caffeinate cannot stop that), which pauses
# training. Needs your admin password (sudo).
#
# Safety: close the lid only on a desk with air around it, ideally plugged in. Do not put the Mac in a bag while this
# is on: it stays awake and warm. To switch sleep back on early by hand:  sudo pmset -a disablesleep 0
cd "$(dirname "$0")/.." || exit 1

# the queue process itself ("/bin/bash runs/queue....sh"), not the log watcher that mentions its name
QPID=$(pgrep -f '^/bin/bash [^ ]*runs/queue[^ ]*\.sh$' | head -1)
if [ -z "$QPID" ]; then
  echo "No training queue is running. Start one first: start/3_background_training.command"
  read -r -p "Press Return to close." _; exit 1
fi
echo "Training queue: $(ps -o command= -p "$QPID")  (pid $QPID)"

if ! pmset -g batt | grep -q "AC Power"; then
  echo
  echo "You are on battery. With sleep off and the lid closed, training keeps drawing power (roughly a third of"
  echo "the battery per hour of training) and the Mac stays warm. Plugging in is better."
  read -r -p "Continue on battery anyway? [y/N] " a
  [ "$a" = "y" ] || [ "$a" = "Y" ] || exit 1
fi

echo
echo "Switching sleep off until that queue ends. macOS will ask for your password."
sudo -v || exit 1
sudo nohup "$PWD/scripts/keep_awake.sh" "$QPID" > /dev/null 2>&1 &  # shared with start/6_run_project_external.command
sleep 2
if pmset -g | grep -q "SleepDisabled[[:space:]]*1"; then
  echo "Sleep is off. You can close the lid now; training continues."
  echo "Sleep switches back on by itself when the queue ends and nothing else needs it off (or now with: sudo pmset -a disablesleep 0)."
else
  echo "Could not confirm sleep is off; check with: pmset -g | grep SleepDisabled"
fi
read -r -p "Press Return to close this window (training is not affected)." _
