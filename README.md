# Donkey Kong RL (MAME + Stable-Baselines3)

Train an agent to play arcade **Donkey Kong** (`dkong`) through MAME. First goal:
**clear the barrel/girders board** (reach Pauline at the top).

> **New here / picking this up? Read [`HANDOFF.md`](HANDOFF.md) first** — it has the
> full project state, what's been tried and why, the current training run, and
> next steps. This README is the reference; HANDOFF is the story + status.

- **Approach (current):** **Go-Explore** — phase 1 (`dkong_ai/go_explore.py`) is a
  policy-free exploration archive over MAME save-states that found the project's
  first-ever bottom-up clears (465 verified winners across two archives); phase 2
  (`train.py --backward-dir`) robustifies them with the backward algorithm:
  RecurrentPPO starting near Pauline, walking the start back down the proven
  routes as the clear rate rises.
- **Observation:** Dict — `image`: 84×84×4 (2-frame stack, grayscale + threat/ladder map);
  `ram`: 62 normalised features (barrel positions, velocities, edge proximity, fireball, hammer).
- **Reward (from RAM):** height-milestone + exploration novelty + expert-route
  corridor + waypoint milestones + climb bonus + de-weighted score + death/clear.
  All height rewards gated on `is_jumping==0`. Episodes are **single-life** (any
  death terminates). See `dkong_ai/mame_env.py:_reward` and HANDOFF.
- **Algorithm:** RecurrentPPO / LSTM (`sb3_contrib.RecurrentPPO`, `MultiInputLstmPolicy`),
  GPU, 16 parallel envs.

## Architecture

```
MAME (dkong) --autoboot_script--> scripts/bridge.lua   (socket SERVER, lock-step)
                                          | TCP 127.0.0.1:5000+i
                          dkong_ai/mame_env.py   (Gymnasium env, socket CLIENT)
                                          |
                          dkong_ai/train.py   (SB3 PPO, CnnPolicy, N parallel envs)
```

Per step: env sends a 1-byte action → bridge applies inputs, runs `frameskip`
frames, ships `[ram][pixels]` → env builds obs + reward. The bridge also handles
control bytes: coin/start, soft-reset, **save/load state**, and clean quit.

## Setup (already done on this machine)

- MAME 0.264 (`apt`); Python venv at `.venv` with torch (CUDA), SB3, gymnasium,
  opencv. ROM: legally-supplied `dkong.zip` in `./roms/` (verified).

## Confirmed RAM map (`dkong_ai/memory_map.py`)

Source: Don Hodges 2008 Z80 disassembly, cross-verified empirically.

**Mario / game state**

| name | addr | notes |
|---|---|---|
| lives | 0x6228 | death = lives decrement (reliable) |
| screen_id | 0x6227 | 1=barrels 2=pie 3=elevator 4=rivet; clear = increment |
| level | 0x6229 | current level number |
| mario_x | 0x6203 | +right |
| mario_y | 0x6205 | smaller = higher; start row ~240 |
| is_jumping | 0x6216 | 1 while Mario is mid-jump |
| jump_dir | 0x6211 | jump direction |
| has_hammer | 0x6217 | 1 while Mario holds a hammer |
| game_start | 0x622C | 1 once a real game is underway |
| bonus | 0x62B1 | bonus timer |
| eol_counter | 0x6388 | end-of-level counter |
| bonus_item | 0x6343 | bonus item on board |

⚠️ **`0x6200` ("is_dead") is unreliable** — reads 1 while Mario is alive and
controllable. Use `lives` for death, not this.

**Score** (tile RAM, stride 0x20 — DK monitor is rotated)

| name | addr | notes |
|---|---|---|
| score_100 | 0x7721 | hundreds digit; low nibble = digit value |
| score_1k | 0x7741 | thousands digit |
| score_10k | 0x7761 | ten-thousands digit |
| score_100k | 0x7781 | hundred-thousands digit |

Tens digit (0x7701) always 0 (scores are multiples of 100) and sits in a volatile
timer region — excluded. `decode_score()` in `memory_map.py` handles the two valid
tile encodings (`'0'=0x00` in live play, `'0'=0x10` in some HUD states).

**Barrels** (6 slots at 0x6700, stride 0x20 per slot)

| offset | meaning |
|---|---|
| +0x00 | status: 0=inactive, 1=rolling, 2=deploying |
| +0x03 | x position (same coord system as mario_x) |
| +0x05 | y position (same coord system as mario_y) |

Slots: `barrel0` @ 0x6700, `barrel1` @ 0x6720, … `barrel5` @ 0x67A0.

**Fireball** (flame enemy that chases Mario)

| name | addr | notes |
|---|---|---|
| fireball_st | 0x6400 | 0=inactive, 1=active |
| fireball_x | 0x6403 | x position |
| fireball_y | 0x6405 | y position |

**Hammer pickup**

| name | addr | notes |
|---|---|---|
| hammer_x | 0x6A1C | x of the hammer sprite on the board |
| hammer_y | 0x6A1F | y of the hammer sprite |
| has_hammer | 0x6217 | 1 while Mario is wielding it |

## Run it

```bash
# Train (16 parallel envs, ~800–1000 fps, ~3.5h per 10M steps)
.venv/bin/python -m dkong_ai.train --rom-dir ./roms --timesteps 30000000 --n-envs 16

# Watch what a checkpoint learned (records a clean .inp, then plays it windowed)
.venv/bin/python -m dkong_ai.eval --rom-dir ./roms --model artifacts/checkpoints/ppo_dkong_<N>_steps
./scripts/playback.sh artifacts/recordings/<file>.inp

# Live metrics / graphs (current run logs to /tmp/dk_explore.log — see HANDOFF)
grep -E "ep_rew_mean|height_mean|height_best|clear_rate" /tmp/dk_explore.log | tail
.venv/bin/tensorboard --logdir logs

# Diagnose where the policy dies / peaks (death + peak positions)
.venv/bin/python -m dkong_ai.diag --rom-dir ./roms --model artifacts/ppo_dkong_explore --port 5100
```

Other entry points: `dkong_ai.probe` (dump screen geometry + input fields),
`dkong_ai.smoke` (end-to-end env sanity + `check_env`), `run_mame.sh` (windowed
MAME with the cheatfind plugin for RAM work).

## Progress metrics (logged each PPO update, via ClimbMetricsCallback)

- `climb/height_mean`, `climb/height_best` — pixels climbed above the start row
  (`BASE_Y=240`); higher = better. Captured as the per-episode *peak*, so the
  level-advance height reset doesn't corrupt it.
- `climb/clear_rate` — fraction of recent episodes that cleared the board
  (`screen_id` 1→2). **The definitive "reached the top" signal.**
- `climb/clear_rate_bottomup` — clears among **live-barrel bottom-start**
  episodes only: excludes curriculum starts and `no_barrels` episodes (the
  latter faked this metric in run 27g — see HANDOFF §12, spawn bug). This is
  the milestone metric.
- Per-episode ground truth: `logs/episodes/dk_<port>.monitor.csv` (reward, len,
  max_height, cleared, start_type, start_y, start_screen, end_screen, bw_pos,
  no_barrels). **Audit any surprising aggregate against these rows first.**

## How resets work (and the recording tradeoff)

- One **persistent MAME per env**; the socket lives for the whole run.
- **Training (`record=False`):** first reset plays the ~19s intro once, then
  **saves a state**; every later reset **loads** it (~0.03s vs ~1.5s). Fast, but
  a save-load isn't an input event so it breaks `.inp` playback.
- **Eval (`record=True`):** uses intro/soft-reset (no state loads) and records a
  clean, playable `.inp` per session for watching the trained policy.

## Key engineering notes / gotchas

- **Process safety:** `train.py` closes the vec env in a `finally` + SIGTERM
  handler; MAME launches with `PR_SET_PDEATHSIG` so it can never orphan. Verified
  0 strays after normal exit, SIGTERM, and SIGKILL. Kill strays with
  `pkill -x mame` — **never** `pkill -f 'mame dkong'` (matches your own shell).
- **Intro/control:** after coin+start there's a ~19s intro + a "ready" freeze
  where input is ignored. `_start_game` detects real control by input *response*
  (hold right until Mario actually walks), not RAM flags.
- **Stream framing:** the env keeps a `_rxbuf` so handshake bytes that over-read
  into the first obs frame aren't dropped (that desync caused intermittent
  `IndexError` at 16-env launch).
- **Throughput** is GPU/inference-bound, not reset-bound: scale via `--n-envs`
  (8≈250 fps, 16≈800 fps).

## Status

**Full project state, run history, models, and next steps: [`HANDOFF.md`](HANDOFF.md).**

In short (run 26 active): the LSTM broke the long-standing height~54 wall — run 22
first reached height_best=146 with RecurrentPPO. Run 25 at 42M steps has `height_mean≈38`
(reliably into the first ladder) and `height_best=162` (4th girder). Still 0 clears
bottom-up with live barrels. The remaining blocker is the left traverse from the 2nd
girder to the x=53 ladder — the agent needs to learn to time that crossing around
barrel gaps.
