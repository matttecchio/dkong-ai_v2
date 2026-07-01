# Donkey Kong RL — Handoff / Complete Project State

**Single source of truth.** Read this before changing anything — several mechanisms
are non-obvious and easy to regress. Pairs with `README.md` (quick reference).

Last updated: 2026-07-01, Run 23 restarted (PID 477461, RecurrentPPO_7) after six-bug fix pass.

---

## 0. TL;DR

- **Full pipeline works and is robust**: MAME `dkong` driven from Python, a
  Gymnasium env over a socket bridge, PPO/RecurrentPPO (SB3 / SB3-Contrib) on
  pixels+RAM, reward from RAM. 16 parallel envs, ~600–900 fps, runs overnight
  with 0 crashes.
- The agent learns to jump barrels, score, and climb roughly half the board.
  With barrel-free episodes it can clear the stage (~18% of training episodes
  at run 19 peak). **Bottom-up with live barrels: still 0 clears after 21 runs.**
- The persistent blocker was height ~53–54. **RecurrentPPO (LSTM) broke it**: run 22
  reached height_best=146 (5th girder, one below Pauline) — first run ever past 54.
  **Run 23 is active**, warm-starting from run 22's LSTM weights with lower lr and new
  upper-board rewards to push past the new stall at 146.
- **lr_schedule bug fixed (2026-06-29)**: `PPO.load()` restores `lr_schedule`
  from the checkpoint; setting `model.learning_rate` alone has no effect. Fix:
  also set `model.lr_schedule = get_schedule_fn(args.lr)` after warm-start.

---

## 1. Goal

Train an RL agent to play arcade **Donkey Kong** (`dkong`) through MAME from
pixels (CNN) + RAM features (MLP). First milestone: **clear the barrel/girder
stage bottom-up with live barrels** (reach Pauline at the top). Stretch: all 4
stages.

---

## 2. Machine / environment

- WSL2 Linux, **RTX 4080 SUPER 16GB**, 22 cores, 30GB RAM.
- **MAME 0.264** (`apt`). Python venv at `.venv` (torch+CUDA, SB3 2.x,
  gymnasium, opencv, numpy).
- ROM: `dkong.zip` in `./roms/` (copyrighted — not redistributable).
- Project root: `/home/claw3/dkong-ai/`.

---

## 3. Quick start

```bash
# Train LSTM (run 23) — full LSTM warm-start from run22, lr=2e-5, n_epochs=3
nohup .venv/bin/python -m dkong_ai.train --rom-dir ./roms \
    --save artifacts/ppo_dkong_run23 --stack 2 --gamma 0.999 \
    --lstm --lstm-hidden 256 --n-envs 16 --timesteps 100000000 \
    --p-no-barrels 0.0 --p-curric 0.0 --lr 2e-5 --n-epochs 3 \
    --init-from artifacts/ppo_dkong_run22_last \
    > logs/run23.out 2>&1 &

# Watch a trained model (records .inp, then plays windowed)
.venv/bin/python -m dkong_ai.eval --rom-dir ./roms \
    --model artifacts/checkpoints/ppo_dkong_run23/ppo_dkong_run23_Xsteps \
    --port 5100 --stack 2
./scripts/playback.sh artifacts/recordings/<file>.inp

# Bottom-up live-barrel eval (no curriculum, no barrel-free episodes)
.venv/bin/python -m dkong_ai.eval --rom-dir ./roms \
    --model <ckpt> --port 5100 --stack 2 \
    --p-no-barrels 0 --p-curric 0 --episodes 10

# Monitor running train
tail -f logs/run23.out

# TensorBoard (WSL2: bind to 0.0.0.0 so Windows browser can reach it)
nohup .venv/bin/tensorboard --logdir logs --port 6006 --host 0.0.0.0 \
    > /tmp/tensorboard.log 2>&1 &
# Then open http://localhost:6006 in Windows browser (run 23 = RecurrentPPO_7)
```

⚠️ Eval/diag always use `--port 5100` to avoid colliding with training (5000+).

---

## 4. Architecture

```
MAME (dkong) --autoboot_script--> scripts/bridge.lua  (socket SERVER, lock-step)
                                        | TCP 127.0.0.1:(5000+env_index)
                        dkong_ai/mame_env.py  (Gymnasium env, socket CLIENT)
                                        |
                        dkong_ai/train.py  (SB3 PPO MultiInputPolicy, 16 envs)
```

**Observation** (`Dict`):
- `"image"`: `(84, 84, 2)` uint8 — channel 0: grayscale pixels; channel 1:
  static threat/ladder/fall-zone map (see §6). Stacked ×n_stack by
  `DkFrameStackWrapper` → `(84, 84, 2×n_stack)` at policy input.
  Run 21+: `n_stack=2` → `(84, 84, 4)`.
- `"ram"`: `(62,)` float32 — normalised RAM features (see §5).

**Policy** (`dkong_ai/dk_policy.py`):
- `DkFeaturesExtractor`: NatureCNN on image → 256 features; Linear MLP on RAM
  → 64 features; concat → 320 features → LSTM → RecurrentPPO policy/value heads.
- `DkFrameStackWrapper`: stacks `image` across N frames (run 21: **stack=2** —
  optical flow only; LSTM handles long-range temporal memory), passes `ram` from
  latest frame only.
- **Run 21+**: `RecurrentPPO` (`sb3_contrib`) with `MultiInputLstmPolicy`.
  LSTM hidden size 256, 1 layer, shared actor/critic. Stack reduced from 8→2.

**Actions** (8): noop, L, R, U, D, jump, jump+L, jump+R.

**Bridge control bytes** (not agent actions): `0xF1` coin, `0xF2` start,
`0xFE` soft-reset, `0xFD` clean-quit, `0xFC` save, `0xFB` load, `0xE0+i`
load curriculum state i, `0xF8` freeze barrels, `0xF7` unfreeze barrels.

---

## 5. RAM features (`dkong_ai/memory_map.py` + `mame_env.py:_build_ram_features`)

**62 features** (layout: `[mario_x/255, mario_y/240]` + 6 barrels × 7 + 5
fireballs × 3 + hammer × 3):

Per barrel: `[Δx/128, Δy/120, vx/8, vy/20, lad53/64, edge_dist, active]`
- `vx/vy`: per-step velocity (frameskip=4); horiz norm ÷8, vertical ÷20.
- `lad53`: barrel x-distance to the critical left ladder at x=53 (norm ÷64).
  Tells agent whether a barrel is heading for that ladder column.
- `edge_dist`: normalised distance to the girder edge the barrel is heading
  toward (0 = at edge / about to fall, 1 = far away).

Per fireball: `[Δx/128, Δy/120, active]` — all 5 slots tracked.

**Full RAM address map:**

| name | addr | notes |
|---|---|---|
| lives | 0x6228 | death = decrement (RELIABLE) |
| screen_id | 0x6227 | 1=barrels 2=pie 3=elevator 4=rivet |
| level | 0x6229 | level counter |
| game_start | 0x622C | 1 once game is underway |
| mario_x | 0x6203 | +right |
| mario_y | 0x6205 | smaller=higher; start≈240, top≈58 |
| is_jumping | 0x6216 | 1 mid-jump |
| has_hammer | 0x6217 | 1 while wielding hammer |
| hammer_x/y | 0x6A1C/1F | hammer pickup position |
| score_100..100k | 0x7721/41/61/81 | tile RAM digits; digit = byte low nibble |
| barrel0..5_st/x/y | 0x6700+ stride 0x20 | status (0=inactive,1=rolling,2=deploying) |
| fireball0..4_st/x/y | 0x6400+ stride 0x20 | all 5 slots tracked (status + x + y) |

⚠️ `0x6200` ("is_dead") is **unreliable** — use `lives` for death detection.

---

## 6. Observation image channel (channel 1 — threat/ladder/fall-zone map)

Pixel intensities in channel 1:
| value | meaning |
|---|---|
| 255 | complete ladder (static, pre-computed) |
| 200 | fall-zone: predicted barrel landing spot when barrel is within 40px of a girder edge |
| 180 | live barrel current position |
| 128 | broken ladder stub (barrel can fall through; Mario cannot climb) |
| 120 | fireball position |
| 80 | hammer pickup position |

Fall-zone prediction uses `GIRDER_EDGES` (5 entries) to map barrel position +
direction → landing zone on the next girder below, drawn when the barrel is
within `EDGE_PROX=40` game pixels of the relevant edge.

---

## 7. Reward (`dkong_ai/mame_env.py:_reward`)

Per non-death/non-clear step (when `mario_y` is valid):

| term | value | notes |
|---|---|---|
| Height milestone | +0.5 × new pixels of max height | fires only on NEW max; top ≈ +96 |
| Per-step height bonus | +0.003 × height/100 | continuous gradient so value fn knows "up = better" |
| Girder milestone | +10/30/40/55/70 | one-shot per episode on first reaching each girder |
| Waypoints (7) | +5/10/30/8/8/8/20 | zig-zag route milestones; fire once per episode |
| Traverse progress | +0.05/pixel moved left | while height 36-65, x 53-143 (2nd girder) |
| Upper traverse progress | +0.05/pixel moved right | height 140-158, x 67-147 (5th girder → top ladder) |
| Climb bonus | +0.30/step | while ascending at x=43-68, height 40-100 (2nd→3rd ladder) |
| Upper climb bonus | +0.30/step | while ascending at x=137-160, height 138-192 (final ladder) |
| Ladder idle cost | −0.05/step | in either climb zone but y unchanged |
| Anti-camping | −0.01/step | height 36-65, x>130, no hammer |
| Corner penalty | −0.03/step | height<15, x<30 or x>190 (bottom floor dead-ends) |
| Score | +0.003/pt | guarded: 0<gain≤2000; gated in camp zone |
| Death | −10 | on lives decrement |
| Farming death | −5 extra | if episode max height < 40 when dying |
| Clear | +100 | screen_id increment |

**Girder milestones** (new in run 18): heights 44/78/112/144/182 → bonuses
10/30/40/55/70. Full climb path worth ~+300 total vs ~+65 for a farming episode.

---

## 8. Reset / curriculum mechanics

- **One persistent MAME per env.** Socket lives the whole run. No per-episode
  relaunch (port rebind = "Address already in use").
- **Fast resets** (`record=False`): first reset plays ~19s intro, saves state;
  all later resets load it (~0.03s). Disabled when `record=True` (.inp playback
  requires real input events).
- **RNG diversity**: after each load, advance 0–15 random NOOP frames so barrel
  patterns differ per episode.
- **Barrel-freeze training wheels** (`P_NO_BARRELS`, default 0.15, run 18 = 0.70):
  the bridge `0xF8` command zeroes all barrel/fireball status bytes each frame.
  70% barrel-free lets the agent drill the route; 30% live episodes keep it honest.
- **Wall-zone curriculum** (`_p_curric=0.15`): 15% of episodes load one of the
  5 lowest curriculum states (heights 35–52, mid-traverse with live barrels).
  Upper states excluded to avoid confounding height metrics.
- ⚠️ Curriculum confounds `height_mean`/`height_best` — judge by `clear_rate`
  and bottom-up evals (P_NO_BARRELS=0, _p_curric=0).

---

## 9. Complete run history

| run | key changes | steps | bottom-up h | clear_rate | outcome |
|---|---|---|---|---|---|
| 1 "overnight" | score + per-step climb | 16.7M | ~47 | 0 | point-farming local optimum |
| 2 "climb" | height milestone dominant | 30M | ~48 | 0 | milestone alone didn't break wall |
| 3 "explore" | + novelty + expert corridor | 7.9M | ~52 | 0 | first to nudge past 47, slow |
| 4 "curric" | + curriculum (near top) | 30M | ~53 | 0.01 | first clears — top-start only |
| 5 "bcrl" | BC init + curriculum | 30M | ~43 | 0.01 | BC hurt bottom-up; brittle |
| 6 "waypoint" | wall curriculum + waypoints | 30M | ~53 | ~0 | WP0 threshold too tight |
| 7 | wider WP0 + anti-camp | ~2M | — | — | stopped: gamma=0.99 → clear reward ≈ 1e-8 |
| 8 | gamma=0.999 + ladder map channel | — | — | — | superseded |
| 9 | score gating + camping penalty | — | ~54 | 0 | wall at x≈75 |
| 10 | WP1b + climb bonus + ent=0.03 | 10M | ~54 | 0 | wall unchanged |
| 11 | 50% barrel-free episodes | — | ~54 | 0 | skill didn't transfer to live barrels |
| 12 | dense traverse reward + p_no_barrels=0.15 | — | ~54 | 0 | marginal |
| 13 | pure bottom-up (no curriculum) | 34.9M | ~54 | 0 | wall confirmed across 200M+ steps |
| 14 | **hybrid CNN+RAM architecture** | ~5.5M | ~54 | 0.03 | first bottom-up clears — barrel-free only |
| 15 | + vx/vy/lad53 features, stack=8 | ~5.5M | ~54 | 0 | **vx/vy bug: always 0 (see §11)** |
| 16 | **vx/vy bug fixed** + edge_dist + fall-zone overlay | ~6.5M | ~54 | 0 | wall unchanged |
| 17 | + per-step height bonus | ~3M | ~54 | 0 | wall unchanged |
| 18 | **70% barrel-free** + farming death penalty + corner penalty + girder milestones | ~10M | ~54 | 0 | wall unchanged; 70% barrel-free didn't transfer |
| 19 | episode timeout + hammer-wall penalty + WP1b=75, RAM 62 (5 fireballs) | 24.5M | ~54 | 0.18 peak | **best ever at 13.9M**; collapsed at 17M (lr=2.5e-4 too high); 18% clears = barrel-free only |
| 20 | lr=5e-5 + P_NO_BARRELS=0.30, warm-start run19@14M | ~3M | ~54 | 0 | **lr_schedule bug**: lr was still 2.5e-4 (fixed in code); restarted; wall unchanged |
| 21 | **LSTM (RecurrentPPO)**, stack=2, no barrel-free, CNN weights from run19@14M | 30.4M | 16–28 (eval broken) | 0 | stopped: `clip_fraction=0.34` (lr too high); restart at lr=5e-5 recommended |
| 22 | LSTM, lr=5e-5, **no curriculum**, CNN/RAM from run21_last (fresh LSTM weights) | 36.5M | 18–26 (eval) | 0 | **height_best=146** (5th girder — first run past 54); stalled at 37M, height_mean=27 |
| 23 | LSTM, lr=2e-5, n_epochs=3, **full LSTM warm-start** from run22_last, upper-board rewards | **active** | TBD | TBD | **current run** |

---

## 10. Run 23 — current run (LSTM, lr=2e-5, n_epochs=3, full warm-start from run 22)

**PID**: 477461. **Log**: `logs/run23.out`.
**Save**: `artifacts/ppo_dkong_run23`. **Stack**: 2.
**TensorBoard**: `RecurrentPPO_7` (`--host 0.0.0.0` for WSL2 access).
*(Restarted after six-bug fix pass — see §12. Previous short run at RecurrentPPO_6 discarded.)*

**Key changes from run 22:**
- `--lr 2e-5` (was 5e-5 — clip_fraction stuck at 0.20–0.23, policy still thrashing)
- `--n-epochs 3` (was 4 — fewer passes over each rollout batch reduces clip_fraction)
- `--init-from artifacts/ppo_dkong_run22_last` — **full warm-start including LSTM weights**.
  Run 22 built LSTM temporal context that reached height 146; run 23 inherits it.
  (Run 22 only transferred CNN/RAM via `--transfer-features-from`; run 23 keeps everything.)
- **New upper-board rewards** (all in `mame_env.py`):
  - `UPPER_TRAVERSE_PROGRESS`: +0.05/pixel rightward on 5th girder (x=67→147, h=140-158)
    — mirrors the 2nd-girder TRAVERSE_PROGRESS that broke the 54 wall; targets the run 22
    stall zone (height 146 = 5th girder, arrive at x≈67 from 4th-girder ladder)
  - `UPPER_CLIMB_BONUS`: +0.30/step ascending the final ladder (x=137-160, h=138-192)
  - `UPPER_LADDER_IDLE_COST`: −0.05/step idle on the final ladder
  - `WP5`: +20 waypoint near Pauline (height≥170, x>100)

**SUCCESS** = `height_mean` rising past 40 then 60; `height_best` breaking past 146.

---

## 11. Run 22 — stopped at 36.5M steps (2026-07-01)

**Recovery checkpoint**: `artifacts/ppo_dkong_run22_last.zip`.
**Checkpoints**: `artifacts/checkpoints/ppo_dkong_run22/` (every 500k, up to 36.5M).
**TensorBoard**: `RecurrentPPO_5`.

**Result**: `height_best=146` (5th girder — **first run in 22 attempts to break past 54**).
`height_mean=27–29` flat, `clip_fraction=0.20–0.23` not declining.
Stalled at 37M steps with no upward movement for 5M+ steps → stopped early.
Eval (10 eps, live barrels): max_height 18–26 — confirms 146 was a rare outlier, not reliable.

**Key changes from run 21:**
- `--lr 5e-5` (was 2.5e-4 — too high, clip_fraction=0.34 and thrashing)
- `--p-curric 0.0` — no curriculum. Every episode starts from the bottom.
  LSTM temporal context built during the episode is the core insight; curriculum
  starts deprive it of barrel history at the traverse zone.
- `--transfer-features-from artifacts/ppo_dkong_run21_last` — 12 CNN+RAM MLP layers
  transferred; LSTM + policy/value heads freshly initialised.

---

## 12a. Run 21 — stopped at 30.4M steps

**Log**: `/tmp/dk_run21.log`. **Recovery checkpoint**: `artifacts/ppo_dkong_run21_last.zip`.
**Checkpoints**: `artifacts/checkpoints/ppo_dkong_run21/` (every 500k steps, up to 30M).

**Why LSTM**: after 20 runs and ~300M+ steps, the wall at height 53–54 is a
temporal memory problem. The left traverse requires knowing whether a barrel
passed x=53 in the last 2–3 seconds. An 8-frame stack gives ~0.5s of visual
history — not enough. `RecurrentPPO` + LSTM hidden state provides proper
long-range memory.

**What ran:**
- `RecurrentPPO` + `MultiInputLstmPolicy` (sb3-contrib).
- `--stack 2`, `--lstm-hidden 256`, `--p-no-barrels 0.0`, `--lr 2.5e-4`.
- CNN + RAM MLP weights transferred from run19@14M (`features_extractor.*`, 11 layers).
  Note: the first CNN conv layer was **not** transferred (shape mismatch: run19 had
  8 input channels [stack=4 × 2ch/frame], run21 has 4 [stack=2 × 2ch/frame]).
  Only deeper conv layers + RAM MLP were copied.

**Why stopped**: `clip_fraction=0.27–0.34` throughout training (healthy: 0.05–0.15).
34% of gradient steps were hitting the PPO clip ceiling, meaning the policy was
lurching around rather than converging. `height_mean` declined from 33→23 over
12M steps — signature of thrashing. Root cause: `--lr 2.5e-4` is too high for
this LSTM model.

**Training metrics at stop (30.4M steps):**
- height_best: 192 (near top — reached at least once during training)
- height_mean: ~23
- clip_fraction: 0.27
- explained_variance: 0.957
- fps: 511

**SUCCESS** = live-barrel bottom-up `height_mean > 54` or first live-barrel clear.

---

## 12. Critical bugs already fixed (don't reintroduce)

### --p-curric flag silently ignored (fixed 2026-07-01)
`mame_env.py:__init__` set `self._p_curric = 0.15` as an **instance** attribute,
which shadowed the class attribute set by `DonkeyKongEnv._p_curric = args.p_curric`
in `train.py`. Every env construction overwrote the CLI value — so `--p-curric 0.0`
ran at 15% curriculum anyway. Runs 22 and early 23 were affected.
**Fix**: removed the `self._p_curric` assignment from `__init__`. `_p_curric` is
now a class attribute only (same pattern as `P_NO_BARRELS`).

### CORNER_H_MAX too low — corner penalty never fired (fixed 2026-07-01)
`CORNER_H_MAX = 15` was supposed to catch Mario in ground-floor dead-end corners.
But the ground floor is at mario_y ≈ 220–224, giving `height = BASE_Y - mario_y ≈ 16–20`.
`height < 15` is below the floor and never true in normal gameplay — the penalty
was completely dead. `CORNER_X_RIGHT = 190` also left a large unpenalised gap
between the first ladder (x≈143) and the right wall (x≈224).
**Fix**: `CORNER_H_MAX` 15→25, `CORNER_X_RIGHT` 190→160.

### smoke.py broken with Dict observation space (fixed 2026-07-01)
`smoke.py` called `obs.shape` on the reset output, which is a `Dict` since run 14.
This raised `AttributeError` before any validation ran.
**Fix**: changed to `obs["image"].shape`, `obs["image"].dtype`, `obs["ram"].shape`.

### diag.py incompatible with current architecture (fixed 2026-07-01)
Used `VecFrameStack` (wrong — should be `DkFrameStackWrapper`) and `PPO.load()`
only (crashes on RecurrentPPO checkpoints). Effectively broken since run 14.
**Fix**: full rewrite to match `eval.py` — `DkFrameStackWrapper`, auto-detect
RecurrentPPO with LSTM state threading, `--stack` flag.

### eval.py default --stack 8 mismatched training (fixed 2026-07-01)
`eval.py` defaulted to `--stack 8` while `train.py` defaulted to `--stack 4`
and current runs use `--stack 2`. Easy to silently get wrong observation shape.
**Fix**: eval default changed to `--stack 2` (matches all runs from 21+).

### probe.py --headless flag could not be disabled (fixed 2026-07-01)
`action="store_true", default=True` means the flag is always True and cannot
be unset from the CLI. **Fix**: replaced with `--no-headless` flag.

---

### lr_schedule not overridden on warm-start (fixed 2026-06-29)
`PPO.load()` restores `lr_schedule` (a callable) from the checkpoint.
Setting `model.learning_rate = args.lr` updates only a dead attribute; the
optimizer and logged lr remain at the checkpoint value.
**Fix**: after warm-start, also set `model.lr_schedule = get_schedule_fn(args.lr)`.
Applied in `train.py`.

### vx/vy always zero (fixed run 16)
In `step()`, `self._prev = state` was assigned **before** `_preprocess(pix, state)`.
Inside `_preprocess` → `_build_ram_features`, `prev = self._prev` equalled
**current** state → `pbx == bx` → `vx = 0`, `vy = 0` for all barrels, every step.
All velocity features in run 15 were dead weight.
**Fix**: swap order in `step()` — call `_preprocess` first, then update `_prev`.

### gamma=0.99 kills clear reward (fixed run 7)
At ~1840 steps/episode, `0.99^1840 ≈ 1e-8`. The +100 clear reward was effectively
invisible to the value function. **Fix**: `gamma=0.999` → `0.999^1840 ≈ 0.16`.

### 0x6200 ("is_dead") unreliable
Reads 1 while Mario is alive and walking. **Never use for death detection.**
Use `lives` decrement instead.

### eval.py loaded RecurrentPPO models as PPO (fixed 2026-06-30)
`eval.py` used `PPO.load()` on all models. When the model is a `RecurrentPPO`
checkpoint, this loads the wrong algorithm class. Additionally, the eval loop
did not pass `state=` or `episode_start=` to `model.predict()`, so the LSTM
hidden state reset every step — making the LSTM a feedforward network during
eval. **Fix**: `_load_model()` now tries `RecurrentPPO.load()` first and falls
back to `PPO.load()`; LSTM state is threaded through the step loop with proper
`episode_start` flags.

### Recording path never sent barrel freeze/unfreeze (fixed 2026-06-30)
In `mame_env.py:reset()`, the barrel mode command (`0xF8`/`0xF7`) was only sent
in the fast save-state path (training). The recording path (eval) never sent it,
leaving the Lua bridge in whatever `freeze_barrels` state it had from a previous
episode. **Fix**: barrel mode command is now sent in both paths.

### MAME X connection crash (fixed run 3)
Even with `-video none`, SDL opened an X connection to WSLg display. Display
hiccup → MAME killed → `ConnectionError` crashed the whole `SubprocVecEnv`.
**Fix**: `SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy`, drop
`DISPLAY`/`WAYLAND_DISPLAY` for headless processes.

---

## 12. Reward summary (run 21)

| term | value | notes |
|---|---|---|
| Height milestone | +0.5 × new pixels | fires on NEW max height only |
| Per-step height bonus | +0.003 × height/100 | continuous gradient |
| Girder milestone | +10/30/40/55/70 | one-shot per episode per girder |
| Waypoints (6) | +5/10/30/8/8/8 | zig-zag route; fire once per episode |
| Traverse progress | +0.05/pixel left | height 36-65, x 53-143 |
| Climb bonus | +0.30/step | ascending at x=43-68, height 40-100 |
| Ladder idle cost | −0.05/step | climb zone but y unchanged |
| Anti-camping | −0.01/step | height 36-65, x>130 |
| Corner penalty | **−0.20/step** | height<25, x<30 or x>160 (threshold fixed 2026-07-01; was 15/190 = never fired) |
| Score | +0.003/pt | gated in camp zone; **unconditionally gated in right corner** |
| Death | −10 | on lives decrement |
| Farming death | −5 extra | episode max height < 40 at death |
| Episode timeout | −15 | height<60 after 800 steps |
| Hammer-wall penalty | −0.05/step | has_hammer, x<45, height>25 |
| Clear | +100 | screen_id increment |

---

## 13. Why the wall persists (diagnosis) + LSTM rationale

The agent has **never traversed from height 54 to the top with live barrels.**

Confirmed behaviorally (watching .inp):
1. Mario grabs hammer, runs LEFT past x=53 (the climb ladder), farms hammer
   kills at the left wall, then dies when hammer expires.
2. Even without the hammer, he camps the right side of the 2nd girder jumping
   barrels, never attempting the left traverse.
3. He physically passes through the ladder location every episode but doesn't stop.

**Root cause confirmed**: the left traverse (x=143→53) requires knowing whether
a barrel passed the x=53 region in the last 2–3 seconds ("is it safe to go
now?"). Our 8-frame stack gave ~0.5s of visual history — not enough. This is a
**temporal memory problem**, not a reward-shaping problem.

All reward approaches attempted and failed: height milestones, corridor bonuses,
waypoints, score gating, camping penalties, barrel-free training, curriculum.
After 20 runs and ~300M+ steps, only the LSTM architecture change remains untried.

**DAgger (expert demo injection) was considered and rejected**: barrel positions
differ between the demo and live game, so expert actions at "jump now" moments
would be conditioned on barrels that may not exist → teaches nonsense.

---

## 14. Gotchas (already bit us)

- **Kill MAME**: `pkill -x mame`. **Never** `pkill -f 'mame dkong'` — matches
  your shell and kills it.
- **nohup PID**: `nohup ... &` prints the shell wrapper PID. The Python trainer
  is `pgrep -n -f dkong_ai.train`. `kill <trainer_pid>` → MAME processes die
  ~5s later (PR_SET_PDEATHSIG). Killing only the wrapper orphans 16 MAMEs.
- **Throughput is GPU-bound** during PPO (not MAME emulation). 16 envs ~600–900
  fps on RTX 4080 SUPER.
- **Obs space breaks warm-start**: models from a different RAM dim or image shape
  cannot be loaded. RAM dim: 62 (as of run 19). Stack: run 21+ uses `--stack 2`
  (`--stack 8` for runs 15–20). Always match `--stack` to the run being loaded.
- **eval.py --stack default** is now `2` (fixed 2026-07-01; was 8). Always verify
  it matches the training run — run 21+ use `--stack 2`.
- **clip_fraction warning**: for LSTM (RecurrentPPO), healthy clip_fraction is
  0.05–0.15. Above 0.20 means `--lr` is too high and the policy is thrashing.
  Run 21 hit 0.34 (lr=2.5e-4); run 22 stuck at 0.20-0.23 (lr=5e-5); run 23 at lr=2e-5.
- **Curriculum metric confound**: with `_p_curric>0`, `height_mean`/`height_best`
  include curriculum-start episodes. Only `clear_rate` + bottom-up eval are clean.
- **Bottom-up eval**: set `DonkeyKongEnv.P_NO_BARRELS = 0.0` and `base._p_curric = 0.0`
  in the eval script. The training class default is 0.15.
- **Recording + state loads don't mix**: `record=True` uses intro/soft-reset.
  `record=False` uses save-state load (fast). A load isn't an input event → breaks
  `.inp` playback.
- **Stream framing**: `_rxbuf` keeps bytes that over-read from the handshake into
  the first obs frame. Don't remove it — causes intermittent `IndexError` at
  16-env launch.

---

## 15. File map

`dkong_ai/`:
- `mame_env.py` — env: MAME launch, socket bridge, obs build, reward, curriculum.
- `memory_map.py` — all confirmed RAM addresses + score decode.
- `dk_policy.py` — `DkFeaturesExtractor` (CNN+RAM MLP) + `DkFrameStackWrapper`.
- `train.py` — PPO/RecurrentPPO training. Flags: `--n-envs`, `--stack`,
  `--gamma`, `--ent-coef`, `--init-from`, `--p-no-barrels`, `--p-curric`,
  `--save`, `--timesteps`, `--lr`, `--n-epochs`, `--lstm`, `--lstm-hidden`,
  `--transfer-features-from`.
- `eval.py` — eval + record .inp. Flags: `--model`, `--stack`, `--port`,
  `--episodes`, `--p-no-barrels`, `--p-curric`.
- `diag.py` — death/peak position diagnostic.
- `extract_bc.py` / `train_bc.py` — behavioral cloning pipeline (built run 5,
  did not improve over pure RL).
- `find_broken_ladders.py` — tilemap RAM analysis (one-shot tool, keep for reference).

`scripts/`:
- `bridge.lua` — MAME lock-step bridge. Supports EXTRACT mode (BC), barrel
  freeze (0xF8/0xF7), curriculum loads (0xE0+i).
- `playback.sh` — watch a .inp windowed.
- `human_record.sh` — record human play.
- `make_curriculum.lua` — snapshot curriculum states from a demo replay.

`artifacts/`:
- `checkpoints/<run>/` — PPO checkpoints every 500k steps.
- `expert_corridor.json` — height→x route corridor from expert demo.
- `states/dkong/curric_*.sta` — MAME save-states for curriculum.

`demos/dkong.inp` — expert demo (MAME 0.241, plays back faithfully on 0.264).
