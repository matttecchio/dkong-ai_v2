# Run 31 obs bundle — deploy checklist (branch: run-31)

**Trigger: either (a) first honest clear consolidation phase, or (b) the
waterfall wall-verdict (extremes empty ~24h after frontier re-arrival
with full volume).**

## What this branch changes (RAM 84 -> 102)
- +6 per-barrel WIND-UP flags (status==2: DK lifting the barrel — the
  release metronome pro rhythm-reading keys on)
- +10 fireball velocities (vx, vy per slot — drift direction for
  deliberate dodging)
- +1 x131 climb margin (generalized `_ladder_margin`; the next contested
  ladder after the waterfall)
- +1 hammer time remaining (201-exchange floor, measured 2026-07-15;
  cannot jump while wielding — expiry transition is a death trap)

## Deploy day (mirrors Stage B / docs/STAGE_B.md)
1. maintenance flag; zombie-proof stop (verify exit; SIGKILL trainer AND
   workers on timeout; then pkill mame).
2. Merge run-31 into main; pytest (dim tests updated).
3. Launcher: --save artifacts/ppo_dkong_run31,
   --transfer-features-from <newest run30 artifact> (CNN carries; heads
   fresh); LEVEL RESET [0]*16.
4. Rewrite launcher/auto_resume to run31 family (run30 checkpoints
   become shape-incompatible).
5. Expect 1-2 day rebuild; judge by slope; battery series continues
   (KEY_CELLS manifest-stable).

NOTE: the projected-occupancy channel is NOT here — it may ship into
run 30 live (channel-1 painting, no shape change); prototype in
scratchpad, decision pending.
