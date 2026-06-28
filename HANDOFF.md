# Donkey Kong RL — Handoff / Complete Project State

**This is the single source of truth.** The original assistant's chat history and
private memory are gone (project moved to a new Claude plan). Everything needed to
understand, run, and continue this project is in this file + the code. Read this
fully before changing anything — several mechanisms are non-obvious and easy to
regress (see Gotchas).

Pairs with `README.md` (quick reference). Last updated after Run 8 launch (2026-06-26).

---

## 0. TL;DR — where it landed (honest)

- The **full pipeline works and is robust**: MAME `dkong` driven from Python, a
  Gymnasium env over a socket bridge, PPO (Stable-Baselines3) on pixels, reward
  from RAM. 16 parallel envs, ~800–1000 fps, runs 30M steps overnight with **0
  crashes**.
- The agent **learns to climb ~half the barrel board (height ~52 of ~192),
  jump barrels, and score** (5–7k/game). It can **occasionally clear the board
  (~1% of episodes) when *started near the top*** (curriculum), but **cannot
  reliably climb from the bottom to Pauline**. **Goal not yet achieved.**
- The persistent blocker is one specific spot: the **left-traverse to the
  2nd-girder ladder (~height 47–53)**. The agent camps on the right of that
  girder farming barrels instead of going left to the ladder. Five training
  approaches each got stuck at/near this wall.
- **Best models to look at:** `artifacts/ppo_dkong_curric.zip` (climbs ~53, rare
  top clears) or `artifacts/ppo_dkong_explore_last.zip`. (Details in §9.)

## 1. Goal

Train an RL agent to play arcade **Donkey Kong** (`dkong`) through MAME from
**pixels** (CNN), reward from **game RAM**. First milestone: **clear the barrel /
girders board** (reach Pauline at the top). Stretch: all 4 stages.

## 2. Machine / environment (already set up)

- WSL2 Linux, **RTX 4080 SUPER 16GB**, 22 cores, 30GB RAM.
- **MAME 0.264** (`apt`). Python venv at `dkong-ai/.venv` (torch+CUDA, SB3 2.x,
  gymnasium, opencv, numpy).
- ROM: **`dkong.zip` in `dkong-ai/roms/`** (verified `romset dkong is good`). It
  is copyrighted — not redistributable; it's already in place on this machine.
- Project root: `/home/claw3/dkong-ai/`. All commands below run from there.

## 3. Quick start

```bash
cd /home/claw3/dkong-ai

# Train (defaults: 16 envs, 30M steps; warm-start + curriculum optional)
.venv/bin/python -m dkong_ai.train --rom-dir ./roms --save artifacts/ppo_dkong_NEW

# Watch a trained model play (records a clean .inp, then plays it windowed)
.venv/bin/python -m dkong_ai.eval --rom-dir ./roms --model artifacts/ppo_dkong_curric --port 5100
./scripts/playback.sh artifacts/recordings/<file>.inp

# Diagnose where a policy peaks/dies
.venv/bin/python -m dkong_ai.diag --rom-dir ./roms --model artifacts/ppo_dkong_curric --port 5100

# Monitor a running train (logs go to /tmp/dk_*.log; pick the right one)
grep -E "total_timesteps|ep_rew_mean|height_mean|height_best|clear_rate" /tmp/dk_*.log | tail
.venv/bin/tensorboard --logdir logs
```
⚠️ `eval`/`diag` use `--port 5100` to avoid colliding with a training run (5000+).

## 4. Architecture

```
MAME (dkong) --autoboot_script--> scripts/bridge.lua    (socket SERVER, lock-step)
                                          | TCP 127.0.0.1:(5000+env_index)
                          dkong_ai/mame_env.py   (Gymnasium env, socket CLIENT)
                                          |
                          dkong_ai/train.py   (SB3 PPO CnnPolicy, 16 parallel envs)
```
- Lock-step: per step the env sends 1 action byte; the bridge applies inputs, runs
  `frameskip`(=4) frames, ships `[ram_bytes][pixels]`. Env makes the 84×84
  grayscale obs (frame-stacked ×4) + reward.
- Obs = pixels (CNN). Reward = from RAM. (Hybrid, decided up front.)
- Bridge **control bytes** (not agent actions): `0xF1` coin, `0xF2` start,
  `0xFE` soft-reset, `0xFD` clean-quit (flush .inp), `0xFC` save-state,
  `0xFB` load-state, `0xE0+i` load curriculum state i. Agent actions are
  `0x00–0x1F` = bitmask over [Left,Right,Up,Down,Jump].
- Action set (`ACTIONS` in mame_env): noop, L, R, U, D, jump, jump+L, jump+R.

## 5. Confirmed RAM map (`dkong_ai/memory_map.py`)

| name | addr | notes |
|---|---|---|
| lives | 0x6228 | death = decrement (RELIABLE — use this for death) |
| screen_id | 0x6227 | 1=barrels 2=pie 3=elevator 4=rivet; clear = increment (verified) |
| level | 0x6229 | level counter |
| game_start | 0x622C | 1 once a game is underway |
| mario_x | 0x6203 | +right |
| mario_y | 0x6205 | smaller = higher; bottom/start ≈ 240, top ≈ 48 |
| score | 0x7721/41/61/81 | tile RAM digits hundreds→100k; digit = byte low nibble |

⚠️ **`0x6200` ("is_dead") is UNRELIABLE** — reads 1 while Mario is alive and
walking. Never use it for death/control detection; use `lives`.
⚠️ **Pre-game score artifact:** DK shows ~3700 before a real game then resets to
0; the score-reward guard ignores it (`0 < gain ≤ 2000`).

## 6. Reward (`dkong_ai/mame_env.py:_reward`) — current ("explore"/curriculum)

Per non-death/non-clear step:
- **Height milestone:** +0.5 per *new* pixel of max height this episode
  (telescopes to best_height·0.5; top ≈ +96). Only progress pays; camping pays
  nothing. Descending to dodge is free.
- **Novelty + corridor:** first visit to an `(x//16, height//16)` cell this
  episode pays +0.2, plus up to +0.3 if the cell is on the **expert route**
  (`artifacts/expert_corridor.json`, a height→x map from the demo). Once per cell
  → not farmable; rewards exploring left/up toward the ladders.
- **Score:** +0.003/point (barrel jump = +0.3), artifact-guarded.
- **Death −10, clear +100.**

Reward evolution (what each run taught us — see §8): started as score+climb (led
to point-farming), → height-milestone (still camped), → +novelty+corridor (broke
the plateau slightly), with curriculum + BC as separate levers.

## 7. Reset / intro / save-state / curriculum mechanics (subtle — read before editing)

- **One persistent MAME per env**; socket lives for the whole run. We do NOT
  relaunch per episode (relaunch rebinds the port; MAME's listen socket has no
  SO_REUSEADDR → "Address already in use").
- **Intro:** after coin+start there's a **~19s intro** (Kong climbing) + a "ready"
  freeze where input is ignored. RAM flags and the attract-mode demo all look
  "live" during this, so they can't be trusted. `_start_game` detects real control
  by **input response**: hold RIGHT until `mario_x` actually increases.
- **Fast resets:** when `record=False` (training), the first reset plays the intro
  once and **saves a state**; later resets **load** it (~0.03s vs ~1.5s).
  Disabled when `record=True` (a state-load isn't an input event → breaks .inp
  playback).
- **RNG diversity:** after each load, advance a random 0–15 NOOP frames so the
  barrel pattern differs per episode (generalization across DK's RNG).
- **Self-curriculum:** with prob `_p_curric` (0.5 if curriculum states exist),
  reset loads a random expert upper-board state (`curric_0..5`, heights ~35→182,
  in `artifacts/states/dkong/`) so the agent practices the upper board + finish.
  Frozen snapshots fall back to the bottom start (`_is_responsive` probe).
  Create the states with `scripts/make_curriculum.lua` (replays the demo).
  **NOTE:** curriculum makes `height_mean`/`height_best` **confounded by start
  position** — use **`clear_rate`** as the success metric, and eval **bottom-up**
  runs (`record=True`, no curriculum) to measure true climbing.

## 8. Complete run history (what was tried, what happened)

| run | model file | reward / method | steps | bottom-up height | clear_rate | takeaway |
|---|---|---|---|---|---|---|
| 1 "overnight" | `ppo_dkong_overnight_last` | score + per-step climb | 16.7M | ~47 | 0 | **point-farming local optimum** (reward rose to +35 by camping & jumping barrels, never climbed) |
| 2 "climb" | `ppo_dkong_climb` | height-milestone dominant | 30M | ~47.8 | 0 | milestone alone didn't break the wall |
| 3 "explore" | `ppo_dkong_explore_last` | + novelty + expert corridor | 7.9M | **~52** | 0 | first to nudge the mean past 47, but slow |
| 4 "curric" | `ppo_dkong_curric` | explore reward + curriculum (start near top) | 30M | ~53 | **0.01** | **first clears ever** — but only from near-top starts; bottom-up still ~53 |
| 5 "bcrl" | `ppo_dkong_bcrl` | BC init + curriculum | 30M | ~43 | 0.01 | **BC init hurt bottom-up**; brittle BC policy + RL didn't recover |
| 6 "waypoint" | `ppo_dkong_waypoint` | wall curriculum (heights 40/45/50) + waypoint milestones (WP0 x<120), warm-start from curric | 30M | ~53 | ~0 | **WP0 never fired**: agent arrives at h36 from x≈143; x<120 threshold too tight. Anti-camping absent. Wall unbroken. |
| 7 "run7" | `ppo_dkong_run7_last` | same + wider WP0 (x<140) + anti-camping penalty + fresh start | ~2M (stopped) | — | — | **Stopped early**: `gamma=0.99` discovered to make +100 clear reward worth ~1e-8 at episode start (invisible to value fn). See §13 for gamma math. |
| 8 "run8" | `ppo_dkong_run8` | all run-7 reward + **`gamma=0.999`** + **static ladder-map channel** (obs now 84×84×2) | 60M | — | — | **RUNNING** — see §13 for design |

Diagnosis throughout (confirmed by the user, a DK expert, watching replays + by
`diag.py`): the agent reaches the 2nd/3rd girder (~height 47–53) then **camps on
the RIGHT farming barrels; the up-ladder is to the LEFT and it won't traverse
left to it.** This is a hard *exploration/navigation* wall, not a reward-scale
problem. The forced-rightward start (control-probe + the "dodge the fireball"
opening) likely reinforced the right-bias.

## 9. Models inventory (`artifacts/*.zip`)

- `ppo_dkong_curric.zip` — **best overall**; climbs ~53, jumps barrels, rare
  top-start clears. Good for demos / a continuation base.
- `ppo_dkong_explore_last.zip` — similar climber (~52), no curriculum confound.
- `ppo_dkong_bcrl.zip` — BC→RL; bottom-up ~43 (worse). 
- `ppo_dkong_bc.zip` — behavioral-cloning-only; **brittle**, dies at height ~20
  alone (kept for reference / re-fine-tuning experiments).
- `ppo_dkong_climb.zip`, `ppo_dkong_overnight_last.zip` — earlier reward designs
  (point-farmers / wall-stuck).
- `ppo_dkong.zip`, `ppo_dkong_last.zip` — the initial 100k sanity run; ignore.
- `*_last.zip` = saved on interrupt/finish; the bare name = final `model.save`.

## 10. Behavioral cloning pipeline (built in Run 5)

- **Extract:** `dkong_ai/extract_bc.py` replays the expert demo (`demos/dkong.inp`)
  through the bridge in **EXTRACT mode** (`DK_EXTRACT=1` → bridge doesn't apply
  actions, appends the playback's current input bitmask to each obs). Keeps the
  first barrel board, maps bitmask→nearest action, saves `artifacts/bc_data.npz`
  (got 2032 pairs; expert uses LEFT > RIGHT — the route knowledge).
- **Train:** `dkong_ai/train_bc.py` builds 4-frame stacks, constructs an SB3
  CnnPolicy via a spaces-only stub env, supervised-trains (loss = −log_prob) →
  `artifacts/ppo_dkong_bc` (88% train acc at 40 epochs).
- **Fine-tune:** `train.py --init-from artifacts/ppo_dkong_bc` warm-starts PPO.
- **Result:** BC-alone is brittle (no recovery data — expert never died), and
  BC→RL did not beat the from-scratch curriculum run. See §12 for why and what to
  try instead.

## 11. The expert demo (`demos/dkong.inp`)

- Recorded on **MAME 0.241**; **plays back faithfully on our 0.264** (dkong
  emulation is deterministic across these versions — verified: stage progression
  matched DK's real structure, lives rose to 4 = a genuine bonus life). This also
  confirmed `screen_id` increments on a real clear.
- Used to build the **route corridor** (`artifacts/expert_corridor.json`) and the
  **curriculum states** (`artifacts/states/dkong/curric_*.sta`) and the **BC
  dataset**. Drop more `.inp` files in `demos/` to extend any of these.
- **Videos (mp4/mkv) were considered and rejected:** no action labels (would need
  a VPT/inverse-dynamics pipeline) + most YouTube DK footage is the NES/other
  ports with a *different board layout* than arcade `dkong`.

## 12. Run 6 "waypoint" — design and post-mortem

**Changes made:**
1. Wall-zone curriculum: heights now {35,40,45,50,65,95,126,155,182} (9 states, curric_0..8).
2. Waypoint milestones (WP0 x<120, WP1 x<75 +15, WP2-4 for later zig-zags).
3. Warm-start from `ppo_dkong_curric`.

**Failure mode (confirmed by post-run analysis):**
- WP0 (x<120) **never fired.** The agent arrives at height 36 from x≈143 (just got off
  the first ladder). x<120 is 23px too tight — the agent is already to the right of it.
- The curriculum states captured the *first-ladder climb* (x≈143), not positions on the
  2nd girder. So curriculum starts were also not near the traverse.
- Run 6 died without breaking x≈47–53 wall.

## 13. Run 8 "gamma + ladder" — design and what to watch (CURRENT RUN)

**Root causes addressed:**

### A. `gamma=0.99` makes the clear reward invisible
With ~1840 steps/episode and γ=0.99: present value of +100 clear at episode start
= 0.99^1840 ≈ **1e-8**. The value function treats it as zero; clearing the board is
essentially not in the policy objective.

With γ=0.999: 0.999^1840 ≈ **0.16** → the clear is worth ~16 units at episode start.
The value function can now represent "being near the top is better" and TD errors
propagate useful signal back to the bottom of the board.

Source: Wiering et al. 2018 DK paper uses γ=0.999 explicitly for this reason.

### B. The CNN had no ladder visibility
The Wiering paper passes explicit 7×7 ladder-vision grids as input features. Our CNN
had to discover ladders from raw pixels, which apparently it didn't. Run 8 adds a
**second observation channel** — a static 84×84 binary map showing exactly where the
complete ladders are. The barrel board never changes so this map is pre-computed once
in `_build_ladder_map()`.

**Only complete ladders are included** (broken ladders omitted). Positions derived from
expert-corridor trajectory (mario_x while ascending = ladder x):
| channel pixel | game x | game y range | ladder |
|---|---|---|---|
| x84≈47 | 143 | 175–224 | 1st: right side, floor→2nd girder |
| **x84≈17** | **53** | **155–196** | **2nd: FAR LEFT, 2nd→3rd girder ← critical** |
| x84≈43 | 131 | 118–158 | 3rd: right-ish, 3rd→4th girder |
| x84≈22 | 67 | 85–125 | 4th: left, 4th→5th girder |
| x84≈48 | 147 | 48–100 | top section to Pauline |

**Obs space is now (84,84,2).** After VecFrameStack(4) → (84,84,8). Old models
trained with (84,84,1) obs are INCOMPATIBLE with this env (don't load them here).

**Run 8 params:**
- Fresh start (no `--init-from`), 60M steps
- `gamma=0.999`, `ent_coef=0.02`
- All run-7 reward: height milestone + waypoints (WP0 x<140) + anti-camping + novelty+corridor + score*0.003 + death-10 + clear+100
- Trainer PID: see `/tmp/dk_run8.pid`. Log: `/tmp/dk_run8.log`
- Save: `artifacts/ppo_dkong_run8`

**How to judge run 8:**
```bash
grep -E "height_mean|height_best|clear_rate|ep_rew_mean" /tmp/dk_run8.log | tail -30
```
- `height_mean > 53` AND `clear_rate > 0` = wall broken. This is the milestone.
- `ep_rew_mean` is not comparable to prior runs (reward scale changed + gamma changed).
- Run bottom-up eval at ~20M and ~40M steps to check real climbing.

## 14. Recommended next steps (if run 8 also stalls)

1. **Exact ladder positions from tilemap RAM**: tile RAM at 0x7400–0x77FF holds the
   actual barrel-board tilemap. A short Lua probe can extract exact ladder x columns
   and y extents to replace the approximate hardcoded values in `_build_ladder_map`.
2. **Stronger exploration** (RND intrinsic curiosity via `stable-baselines3-contrib`).
3. **More BC data** + DAgger-style correction from varied start positions.
4. **Reduce forced-rightward start** (right-camp bias reinforced from the opening).
5. **Anneal `_p_curric`** down over training once clears appear.

## 15. Gotchas (already bit us — don't rediscover)

- **Headless = no X.** Even with `-video none`, SDL opens an X connection to `:0`;
  a WSLg display hiccup then kills MAME mid-run. FIX (in `_launch_mame`):
  `SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy`, drop `DISPLAY`/`WAYLAND_DISPLAY`
  for headless procs. (Windowed eval/playback are separate and keep the display.)
- **Self-healing:** if a MAME dies (`ConnectionError: bridge closed` / timeout),
  `step`/`reset` catch it and `_recover()` relaunches a fresh instance + new
  save-state, ending just that episode. One crash ≠ whole run dies. Keep this.
- **Killing MAME:** `pkill -x mame` (process name). **NEVER** `pkill -f 'mame
  dkong'` — `-f` matches your own shell's command line and kills the shell.
- **Orphan prevention:** `train.py` closes the vec env in `finally` + a SIGTERM
  handler; MAME launches with `PR_SET_PDEATHSIG=SIGKILL`. Verified 0 orphans after
  normal exit, SIGTERM, SIGKILL. Check with `pgrep -c -x mame`.
- **Stream framing:** the env keeps `self._rxbuf`; handshake bytes can over-read
  into the first obs frame and must not be dropped (caused intermittent
  `IndexError` at 16-env launch).
- **Throughput is CPU-bound** (MAME emulation), GPU ~idle. 16 envs saturates 22
  cores (~800–1000 fps). More envs oversubscribe and don't help. More speed isn't
  the bottleneck anyway — the wall is a learning problem.
- **Warm-start logging:** `PPO.load` restores the saved model's `verbose`; the BC
  model had `verbose=0` → silent training (looked stuck). `train.py` now forces
  `model.verbose=1` on warm-start and uses `progress_bar=False` (rich bar garbles
  file logs).
- **Curriculum metric confound:** see §7 — judge by `clear_rate` + bottom-up eval.
- **MAME runs ~20× real-time unthrottled** (`-nothrottle`); a "1.5s" reset is
  ~30s emulated. Use `-nothrottle` for fast headless playback/extraction.
- **Inline `python -c` with quotes/`%`/`$?` gets mangled** by this shell's wrapper
  — write a module/script file and run `-m`.
- **nohup wrapper PID ≠ Python trainer PID.** `nohup ... &` prints the *shell
  wrapper* PID; the Python trainer is one more level down. Save the trainer PID
  immediately after launch: `sleep 2; pgrep -n -x python > /tmp/dk_run8.pid`.
  `kill $(cat /tmp/dk_run8.pid)` kills the trainer; MAME processes die ~5s later
  (PR_SET_PDEATHSIG). Killing only the wrapper leaves the trainer + 16 MAMEs running.
- **Obs space break:** models trained with obs (84,84,1) are INCOMPATIBLE with the
  current env (84,84,2 since run 8). Load old models only with a patched env or
  the old code version.

## 16. File map

`dkong_ai/`:
- `mame_env.py` — Gymnasium env: MAME launch, socket protocol, reset/intro/
  save-state/curriculum, reward (`_reward`), obs preprocessing, self-recovery.
- `memory_map.py` — RAM addresses + score decode.
- `train.py` — PPO training, `ClimbMetricsCallback` (logs `climb/*`),
  checkpointing (per-run dir, every 500k steps), clean shutdown, `--init-from`,
  `--ent-coef`, `--n-envs`, `--timesteps`, `--save`.
- `eval.py` — run a model, print reward/height/score/cleared, record a .inp.
- `diag.py` — death/peak-position diagnostic.
- `extract_bc.py` / `train_bc.py` — behavioral cloning pipeline.
- `analyze_ladders.py` — build the route corridor from a trajectory log.
- `probe.py` / `smoke.py` — discovery (ports/screen) / end-to-end env sanity.

`scripts/`:
- `bridge.lua` — the MAME-side lock-step bridge (autoboot script). Has EXTRACT
  mode for BC.
- `playback.sh <file.inp>` — watch a .inp windowed. `human_record.sh <name>` —
  record human play to a .inp.
- `make_curriculum.lua` — snapshot curriculum states from a demo replay.
- `ladder_extract.lua`, `playback_log.lua` — read-only loggers for playback.

Root: `run_mame.sh` (windowed MAME + cheatfind plugin for RAM work).
`artifacts/` — models (`*.zip`), `checkpoints/<run>/`, `recordings/` (.inp),
`states/dkong/` (save-states), `expert_corridor.json`, `bc_data.npz`.
`demos/` — drop expert `.inp` files here. `logs/` — tensorboard + per-port logs.

## 17. Gameplay facts (from the user, a DK expert)

- The **first barrel becomes the fireball** — don't camp at spawn; move right a
  little to start.
- **Holding a direction while jumping** enlarges the barrel-jump scan, so you
  reliably score the +100 (a plain jump often scores nothing). The action set
  includes jump+L / jump+R for this reason.
- The barrel board's ladders **alternate sides** (the route zig-zags); the
  2nd-girder up-ladder is on the **left** — the spot the agent won't go to.

## 18. Note on "memory"

This file replaces the original assistant's chat history + private memory store,
which do not transfer to the new Claude plan. If you (a future model/human)
continue this: read §0, §8, §12, §13 first. The pipeline is solid and the problem
is well-characterized — what's left is cracking the one left-traverse, most
likely via wall-placed curriculum (§12 #1) and/or better demonstration data.
