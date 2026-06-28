# Donkey Kong RL (MAME + Stable-Baselines3)

Train an agent to play arcade **Donkey Kong** (`dkong`) through MAME. First goal:
**clear the barrel/girders board** (reach Pauline at the top).

> **New here / picking this up? Read [`HANDOFF.md`](HANDOFF.md) first** — it has the
> full project state, what's been tried and why, the current training run, and
> next steps. This README is the reference; HANDOFF is the story + status.

- **Observation:** raw pixels → 84×84 grayscale, frame-stacked ×4, CNN policy.
- **Reward (from RAM):** height-milestone (progress to the top) + exploration
  novelty + expert-route corridor guidance + de-weighted score + death/clear.
  See `dkong_ai/mame_env.py:_reward` and HANDOFF for *why* it's shaped this way.
- **Algorithm:** PPO (Stable-Baselines3), GPU.

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

| name | addr | notes |
|---|---|---|
| lives | 0x6228 | death = lives decrement (reliable) |
| screen_id | 0x6227 | 1=barrels 2=pie 3=elevator 4=rivet; clear = increment |
| mario_x | 0x6203 | +right |
| mario_y | 0x6205 | smaller = higher; start row ~240 |
| game_start | 0x622C | 1 once a game is underway |
| score | 0x7721/41/61/81 | tile RAM digits (hundreds→100k); digit = low nibble |

⚠️ **`0x6200` ("is_dead") is unreliable** — reads 1 while Mario is alive and
controllable. Use `lives` for death, not this.

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

## Status (final, after 5 training runs)

**Full project state, run history, models, and next steps: [`HANDOFF.md`](HANDOFF.md).**

In short: the pipeline works and is robust; the agent **climbs ~half the board
(height ~52/192), jumps barrels, scores ~5–7k**, and can **occasionally clear when
started near the top (~1%)** — but **cannot yet reliably climb bottom-to-top to
Pauline.** The unsolved blocker is the **left-traverse to the 2nd-girder ladder**
(it camps right and farms barrels instead). Five approaches (reward shaping,
exploration+route-corridor, curriculum, behavioral cloning) each stalled at/near
that wall. Best models: `artifacts/ppo_dkong_curric.zip` /
`ppo_dkong_explore_last.zip`.

Recommended next step (see HANDOFF §12): **curriculum states placed *at* the wall**
(2nd girder, right side, height ~47) so the agent drills the exact left-traverse,
plus stronger exploration and/or better demonstration data.
