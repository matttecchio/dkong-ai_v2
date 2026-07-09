#!/usr/bin/env bash
# Auto-resume DK training after a WSL reboot or trainer crash.
# Installed in claw3's crontab: @reboot (delayed) + every-10-min watchdog.
#
# Behavior:
#   - exits silently if the trainer is running or .maintenance exists
#   - otherwise relaunches from the NEWEST saved model (mtime, not step number —
#     per-run step counters reset so filenames lie across runs)
#   - crashloop guard: max 3 auto-restarts in 2h, then writes logs/AUTO_RESUME_GAVE_UP
#
# To stop training ON PURPOSE (so the watchdog doesn't undo you):
#   touch /home/claw3/dkong-ai/.maintenance
# then kill the trainer. Remove .maintenance when done.
set -u
cd /home/claw3/dkong-ai || exit 1

exec 9>/tmp/dk_auto_resume.lock
flock -n 9 || exit 0                       # another instance mid-flight

[ -e .maintenance ] && exit 0              # manual intervention in progress

# trainer already up? ([.] avoids ever matching a shell quoting this pattern)
pgrep -f "python -m dkong_ai[.]train" >/dev/null 2>&1 && exit 0

ts=$(date +%Y%m%d_%H%M%S)

# crashloop guard: give up after 3 auto-restarts inside 2 hours
journal=logs/auto_resume_restarts.log
if [ -f "$journal" ]; then
    recent=$(awk -v cutoff=$(( $(date +%s) - 7200 )) '$1 >= cutoff' "$journal" | wc -l)
    if [ "$recent" -ge 3 ]; then
        [ -e logs/AUTO_RESUME_GAVE_UP ] || echo "$ts: 3 auto-restarts in 2h — giving up, needs a human/Claude look" | tee logs/AUTO_RESUME_GAVE_UP
        exit 1
    fi
fi

echo "$ts: trainer down — auto-resuming"
pkill -x mame 2>/dev/null && sleep 2       # clear orphaned emulators (never pkill -f)

model=$(ls -t artifacts/ppo_dkong_run28_last.zip artifacts/checkpoints/ppo_dkong_run28/*.zip 2>/dev/null | head -1)
if [ -z "$model" ]; then
    echo "$ts: NO MODEL FOUND — aborting"
    exit 1
fi

log="logs/run28_auto_${ts}.log"
pid=$(scripts/current_launch.sh "$model" "$log")
echo "$pid" > logs/run_current.pid
ln -sfn "run28_auto_${ts}.log" logs/run_current.log
date +%s >> "$journal"
echo "$ts: relaunched PID $pid from $model (log $log)"
