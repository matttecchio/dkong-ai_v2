#!/usr/bin/env bash
# THE canonical launch command for the current run line (run 28 dials).
# Edit dials HERE and only here — auto_resume.sh and manual deploys both call this,
# so a crash-resume can never fire with stale settings.
#
# Run 28 (2026-07-10): capacity bundle — LSTM 512, RAM MLP 128, difficulty in
# obs (RAM 75), spawn burn-in, rehearsal cap K=8, dense12 (WC dedupe 5->2),
# levels reset. First launch used --transfer-features-from run27_last (CNN
# only); resumes warm-start the full run28 model via --init-from.
# lr 5e-5 -> 1e-4 (run 28b, user-approved 2026-07-10 eve): the solo hot-lr
# test from the 27y post-mortem — all 12 frontiers flat 0-1% for 5h at 5e-5
# on 12h-old heads. GUARDRAIL: clip_fraction >0.25 sustained -> back to 5e-5
# (run-19 collapse was 2.5e-4).
#
# Usage: current_launch.sh <init-from-model> <log-file>
# Prints the trainer PID on stdout.
set -u
cd /home/claw3/dkong-ai || exit 1
model="$1"
log="$2"

nohup .venv/bin/python -m dkong_ai.train --rom-dir ./roms \
  --timesteps 100000000 --n-envs 16 \
  --save artifacts/ppo_dkong_run28 --logdir logs \
  --gamma 0.999 --ent-coef 0.01 --lr 1e-4 --n-epochs 3 \
  --stack 2 --p-no-barrels 0.0 --p-curric 0.8 \
  --lstm --lstm-hidden 512 \
  --backward-dir artifacts/backward_dense13 --bw-threshold 0.3 \
  --sil-coef 0.05 \
  --init-from "$model" \
  > "$log" 2>&1 &
echo $!
