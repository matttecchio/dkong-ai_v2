#!/usr/bin/env bash
# THE canonical launch command for the current run line (run 27ak dials).
# Edit dials HERE and only here — auto_resume.sh and manual deploys both call this,
# so a crash-resume can never fire with stale settings.
#
# Usage: current_launch.sh <init-from-model> <log-file>
# Prints the trainer PID on stdout.
set -u
cd /home/claw3/dkong-ai || exit 1
model="$1"
log="$2"

nohup .venv/bin/python -m dkong_ai.train --rom-dir ./roms \
  --timesteps 100000000 --n-envs 16 \
  --save artifacts/ppo_dkong_run27 --logdir logs \
  --gamma 0.999 --ent-coef 0.01 --lr 5e-5 --n-epochs 3 \
  --stack 2 --p-no-barrels 0.0 --p-curric 0.8 \
  --lstm --lstm-hidden 256 \
  --backward-dir artifacts/backward_dense11 --bw-threshold 0.3 \
  --init-from "$model" \
  > "$log" 2>&1 &
echo $!
