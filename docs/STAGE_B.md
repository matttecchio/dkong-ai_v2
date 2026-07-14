# Stage B obs bundle — deploy checklist (branch: stage-b)

**Gate: deploy ONLY after the first verified honest bottom-up clear**
(episode CSV row with bottomup start, cleared=1, glitch_kill=0).

## What this branch changes
- WATCH list 60 → 62 (append-only): `bonus_timer` 0x62B1, `mario_facing` 0x6207
  — in BOTH `dkong_ai/memory_map.py` and `scripts/bridge.lua`, tests updated.
- RAM features 75 → 83: per-barrel `lad203` (distance to the real floor ladder,
  mirrors lad53; +6), bonus timer /48 (+1), facing bit 7 (+1).
- edge_dist for INACTIVE barrels now defaults 1.0 (safe), was 0.0 (= at-edge,
  the most dangerous reading; review r7 #4).
- Safe-climb margin feature CODED (83 -> 84): the reward gate's
  time-race exposed as a continuous input (difficulty already in base 75).
- Obs shape change ⇒ the running run28/29 model CANNOT load these envs.

## Deploy day (run 30)
1. `touch logs/.maintenance` — freeze auto_resume.
2. Stop trainer (SIGTERM PID from logs/run_current.pid, wait, SIGKILL if env-close
   hang; `pkill -x mame`).
3. `git checkout go-explore && git merge stage-b`
4. Edit `scripts/current_launch.sh`: save name `ppo_dkong_run30`, add
   `--transfer-features-from artifacts/ppo_dkong_run2X_last.zip` (CNN transfers;
   RAM-MLP/LSTM/heads fresh), and RESET LEVELS: `rm artifacts/backward_dense13/levels.json`
   (fresh heads must re-earn the chains — do not resume levels).
5. Launch via current_launch.sh, update logs/run_current.pid + symlink,
   verify 16 mames + "[transfer] ..." line, rm .maintenance.
6. Watch first battery: bottomup should recover within ~2 days; if not, the
   transfer is suspect — film before touching dials.
