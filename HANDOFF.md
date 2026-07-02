# Donkey Kong RL — Handoff / Complete Project State

**Single source of truth.** Read this before changing anything — several mechanisms
are non-obvious and easy to regress. Pairs with `README.md` (quick reference).

Last updated: 2026-07-02, Run 25 ended at ~42M steps; **Run 26 starting**.

---

## 0. TL;DR

- **Full pipeline works and is robust**: MAME `dkong` driven from Python, a
  Gymnasium env over a socket bridge, RecurrentPPO (LSTM) on pixels+RAM, reward
  from RAM. 16 parallel envs, ~500–600 fps, runs overnight with 0 crashes.
- **Run 25 ended at 42M steps**: `height_best=162` (4th girder), `height_mean≈38`
  (reliably on the first ladder, trending up toward 44). `ep_rew_mean=-5.78`.
  `explained_variance=0.962` (value function very well calibrated at this point).
- **Run 26**: warm-starts from run25_last. Re-enables curriculum (`p_curric=0.15`)
  to give the agent practice on the upper board. New `climb/height_mean_bottomup`
  and `climb/height_mean_curric` TensorBoard metrics let you see both signal lines.
- All height metrics honest: gated on `is_jumping==0` so jump arcs don't inflate
  `height_best`, `height_mean`, or the height milestone reward.
- **Bottom-up with live barrels: still 0 clears** after 25 runs and ~300M+ steps,
  but run 25 is the best sustained progress yet (LSTM broke the old height~54 wall).

---

## 1. Goal

Train an RL agent to play arcade **Donkey Kong** (`dkong`) through MAME from
pixels (CNN) + RAM features (LSTM). First milestone: **clear the barrel/girder
stage bottom-up with live barrels** (reach Pauline at the top). Stretch: all 4 stages.

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
# Run 26 start command:
nohup .venv/bin/python -m dkong_ai.train --rom-dir ./roms \
    --timesteps 100000000 --n-envs 16 \
    --save artifacts/ppo_dkong_run26 --logdir logs \
    --gamma 0.999 --ent-coef 0.01 --lr 2e-5 --n-epochs 3 \
    --stack 2 --p-no-barrels 0.0 --p-curric 0.15 \
    --lstm --lstm-hidden 256 \
    --init-from artifacts/ppo_dkong_run25_last \
    > logs/run26.log 2>&1 &

# Check training is alive
ps aux | grep dkong_ai.train | grep -v grep
tail -f logs/run26.log

# Kill gracefully (saves final checkpoint to artifacts/ppo_dkong_run26_last.zip):
kill -SIGTERM <pid>

# Watch a trained model (records .inp, then plays windowed)
.venv/bin/python -m dkong_ai.eval --rom-dir ./roms \
    --model artifacts/checkpoints/ppo_dkong_run26/ppo_dkong_run26_Xsteps \
    --port 5100 --stack 2
./scripts/playback.sh artifacts/recordings/<file>.inp

# TensorBoard (WSL2: bind to 0.0.0.0 so Windows browser can reach it)
# Open http://localhost:6006 in Windows browser. Run 26 = RecurrentPPO_11.
```

⚠️ Eval/diag always use `--port 5100` to avoid colliding with training (5000+).

---

## 4. Architecture

```
MAME (dkong) --autoboot_script--> scripts/bridge.lua  (socket SERVER, lock-step)
                                        | TCP 127.0.0.1:(5000+env_index)
                        dkong_ai/mame_env.py  (Gymnasium env, socket CLIENT)
                                        |
                        dkong_ai/train.py  (SB3 RecurrentPPO MultiInputLstmPolicy, 16 envs)
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
- `DkFrameStackWrapper`: stacks `image` across N frames (run 21+: **stack=2** —
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

**⚠️ Missing feature (known gap):** There is no `lad143` — barrel distance to
the x=143 first ladder. `lad53` helps time the 2nd→3rd girder climb; an
equivalent for x=143 would help time the first-ladder climb between barrels.
Consider adding as a future improvement (changes `RAM_FEATURE_DIM` 62→68,
breaks warm-start from run 25).

**Full RAM address map** (`memory_map.py` + `bridge.lua` WATCH_ADDRS, 47 entries,
ORDER MUST MATCH between both files):

| name | addr | notes |
|---|---|---|
| lives | 0x6228 | death = decrement (RELIABLE) |
| screen_id | 0x6227 | 1=barrels 2=pie 3=elevator 4=rivet |
| mario_y | 0x6205 | smaller=higher; start≈240, top≈58 |
| mario_x | 0x6203 | +right |
| is_dead | 0x6200 | **UNRELIABLE** — use lives for death |
| game_start | 0x622C | 1 once game is underway |
| score_100..100k | 0x7721/41/61/81 | tile RAM digits; digit = byte low nibble |
| barrel0..5_st/x/y | 0x6700+ stride 0x20 | status (0=inactive,1=rolling,2=deploying) |
| fireball0..4_st/x/y | 0x6400+ stride 0x20 | all 5 slots tracked |
| hammer_x/y | 0x6A1C/1F | hammer pickup position |
| has_hammer | 0x6217 | 1 while wielding hammer |
| **is_jumping** | **0x6216** | **non-zero during jump arc; used to gate rewards** |

⚠️ `is_jumping` (0x6216) is the **last entry** in both `WATCH_ORDER` and bridge.lua
`WATCH_ADDRS`. Both lists have exactly 47 entries and must remain in sync.
`tests/test_bridge_sync.py` enforces this mechanically — run it after any WATCH change.

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

## 7. Reward (`dkong_ai/mame_env.py:_reward`) — current as of run 26

### Key design principle: is_jumping gate
**All height-based rewards are gated on `not s.get("is_jumping", 0)`.**
Jump arcs temporarily reduce `mario_y` (Mario goes higher) which would otherwise
give milestone credit for heights never actually stood on. The `is_jumping` flag
(0x6216) is non-zero during any jump arc. `_min_y` (which drives `height_best`
in TensorBoard) is also gated in `step()`.

### One-shot rewards (per episode, not farmable):

| term | value | trigger |
|---|---|---|
| Height milestone | +0.5 × new pixels | NEW max height AND **not jumping** |
| WP0 | +5 | height ≥ 36 AND x < 140 (2nd girder heading left) |
| WP1a | +10 | height ≥ 45 AND x < 75 (approaching 2nd→3rd ladder) |
| WP1b | +75 | height ≥ 45 AND x < 58 (AT the ladder entrance) |
| WP2 | +8 | height ≥ 65 AND x > 100 (3rd girder) |
| WP3 | +8 | height ≥ 100 AND x < 85 (3rd girder left traverse) |
| WP4 | +8 | height ≥ 150 AND x > 130 (near top ladder) |
| WP5 | +20 | height ≥ 170 AND x > 100 (near Pauline) |
| 2nd girder | +10 | height ≥ 44 |
| 3rd girder | +30 | height ≥ 78 |
| 4th girder | +40 | height ≥ 112 |
| 5th girder | +55 | height ≥ 144 |
| Top/Pauline | +70 | height ≥ 182 |
| Stage clear | +100 | screen_id increments |

### Per-step rewards:

| term | value | trigger |
|---|---|---|
| Per-step height bonus | +0.003 × height/100 | continuous gradient |
| Novelty | +0.2 (+0.3 bonus) | first visit to 16×16 (x,height) cell; bonus if on expert corridor |
| Score | +0.003/pt | 0 < gain ≤ 2000; **gated out** when height<65 AND x>115 AND not moving left |
| First-ladder climb | +0.30/step | **not jumping**, x=133-155, height=10-44, mario_y decreasing |
| 2nd→3rd ladder climb | +0.30/step | **not jumping**, x=43-68, height=40-100, mario_y decreasing |
| Top-ladder climb | +0.30/step | **not jumping**, x=137-160, height=138-192, mario_y decreasing |
| 2nd-girder traverse | +0.05/pixel | moving left, height=36-65, x=53-143 |
| 5th-girder traverse | +0.05/pixel | moving right, height=140-158, x=67-147 |

### Penalties:

| term | value | trigger |
|---|---|---|
| Death | −10 | life lost |
| Low-progress death | −5 extra | died without reaching height 40 this episode |
| Episode timeout | −15 | 800 steps elapsed without reaching height 60 |
| Anti-camping | −0.01/step | height=36-65, x>130, no hammer |
| Corner penalty | −0.20/step | height<25 AND (x<30 OR x>160) |
| First-ladder idle | −0.05/step | **not jumping**, x=133-155, height=10-44, mario_y unchanged |
| 2nd→3rd ladder idle | −0.05/step | **not jumping**, x=43-68, height=40-100, mario_y unchanged |
| Top-ladder idle | −0.05/step | **not jumping**, x=137-160, height=138-192, mario_y unchanged |
| Hammer-at-wall | −0.05/step | has_hammer AND x<45 AND height>25 |

---

## 8. Height coordinate system and diagnostic thresholds

`BASE_Y = 240`. `height = BASE_Y - mario_y`. Higher = better.

From the expert corridor (`artifacts/expert_corridor.json`):

| height band | x_med | what's happening |
|---|---|---|
| 0–12 | 91 | ground floor starting zone |
| 12–24 | 115 | ground floor, walking right toward first ladder |
| **24–36** | **143** | **first ladder — Mario is actively climbing** |
| 36–48 | 89 | 2nd girder, traversing left |
| 48–60 | 53 | 2nd→3rd girder ladder |
| 60–84 | 96–107 | 3rd girder |
| 84–96 | 131 | 3rd→4th ladder |
| 96–120 | 67–91 | 4th girder |
| 120–144 | 81–123 | 4th→5th traverse |
| 144–158 | 147–203 | 5th girder rightward traverse |
| 158–204 | 147 | final ladder (5th → Pauline) |

**Diagnostic thresholds for `height_mean`:**
- **< 24**: Mario on ground floor only (not reaching the ladder)
- **24–36**: Mario engaging the first ladder but not completing it
- **> 36**: Mario reliably 2/3+ up the first ladder (unambiguous — no ground-floor jump reaches this)
- **> 44**: Mario completing the first ladder and reaching the 2nd girder

**What height_best tells you:**
- `height_best` uses `_min_y` (minimum mario_y seen this episode).
- Since run 25: gated on `not is_jumping` — jump arcs no longer inflate this.
- `height_best` of 162 means Mario genuinely stood at 4th girder level.

**The warm-start regression pattern:**
Every warm-start from a model trained on a different reward function shows:
1. First few episodes: inherited policy plays at its trained level (~50)
2. PPO updates: gradient from new reward disrupts old strategy → drops to ~25
3. Slow recovery: relearns under the new objective

This is expected and not a bug. height_mean crossing 36 consistently is the signal
that the new objective has been learned.

---

## 9. Reset / curriculum mechanics

- **One persistent MAME per env.** Socket lives the whole run. No per-episode
  relaunch (port rebind = "Address already in use").
- **Fast resets** (`record=False`): first reset plays ~19s intro, saves state;
  all later resets load it (~0.03s). Disabled when `record=True` (.inp playback
  requires real input events).
- **RNG diversity**: after each load, advance 0–15 random NOOP frames so barrel
  patterns differ per episode.
- **Barrel-freeze training wheels** (`P_NO_BARRELS`, run 26: **0.0**):
  bridge `0xF8` command zeroes all barrel/fireball status bytes each frame.
  Currently OFF — all episodes have live barrels.
- **Curriculum** (`_p_curric`, run 26: **0.15**): With 15% probability, reset to
  one of the 5 lowest curriculum states (heights 35-52) instead of the ground floor.
  The `_info()` dict includes `start_type = "curriculum" | "bottomup"` so TensorBoard
  can show `climb/height_mean_curric` and `climb/height_mean_bottomup` separately.
  Wall-zone curriculum: only the lowest 5 states (heights 35-52) are used (`n_wall=min(5,n_curric)`).
  Upper states confound height metrics.
- **Start-type tracking**: `mame_env.py:reset()` sets `self._start_type = "bottomup"` as
  default; overrides to `"curriculum"` only when a curriculum state loads successfully
  (falls back to "bottomup" if `_is_responsive()` fails).

---

## 10. Complete run history

| run | key changes | steps | height_best | height_mean | clear_rate | outcome |
|---|---|---|---|---|---|---|
| 1 "overnight" | score + per-step climb | 16.7M | ~47 | ~47 | 0 | farming local optimum |
| 2 "climb" | height milestone dominant | 30M | ~88 | ~48 | 0 | milestone alone didn't break wall |
| 3 "explore" | + novelty + expert corridor | 7.9M | ~78 | ~52 | 0 | first to nudge past 47, slow |
| 4 "curric" | + curriculum (near top) | 30M | 184 | ~53 | 0.01 | first clears — top-start only |
| 5 "bcrl" | BC init + curriculum | 30M | — | ~43 | 0.01 | BC hurt bottom-up; brittle |
| 6 "waypoint" | wall curriculum + waypoints | ~2M | — | ~53 | 0 | WP0 threshold too tight |
| 7 | wider WP0 + anti-camp | ~2M | — | — | — | stopped: gamma=0.99 |
| 8 | gamma=0.999 + ladder map channel | — | — | — | — | superseded |
| 9 | score gating + camping penalty | — | — | ~54 | 0 | wall at x≈75 |
| 10 | WP1b + climb bonus + ent=0.03 | 10M | — | ~54 | 0 | wall unchanged |
| 11 | 50% barrel-free episodes | — | — | ~54 | 0 | skill didn't transfer |
| 12 | dense traverse reward | — | — | ~54 | 0 | marginal |
| 13 | pure bottom-up | 34.9M | — | ~54 | 0 | wall confirmed 200M+ steps |
| 14 | **hybrid CNN+RAM architecture** | ~5.5M | 193 | ~54 | 0.03 | first bottom-up clears — barrel-free only |
| 15 | + vx/vy/lad53 features, stack=8 | ~5.5M | 192 | ~54 | 0 | vx/vy bug: always 0 |
| 16 | **vx/vy bug fixed** + edge_dist + fall-zone | ~6.5M | — | ~54 | 0 | wall unchanged |
| 17 | + per-step height bonus | ~3M | — | ~54 | 0 | wall unchanged |
| 18 | **70% barrel-free** + girder milestones | ~10M | — | ~54 | 0 | wall unchanged |
| 19 | timeout + hammer-wall penalty + WP1b=75 | 24.5M | 193 | ~54 | 0.18 peak | best at 13.9M; collapsed at 17M (lr too high) |
| 20 | lr=5e-5, warm-start run19@14M | ~3M | — | ~54 | 0 | lr_schedule bug; wall unchanged |
| 21 | **LSTM (RecurrentPPO)**, stack=2 | 30.4M | 192 | ~23 | 0 | clip_fraction=0.34 (lr=2.5e-4 too high) |
| 22 | LSTM, lr=5e-5, no curriculum | 36.5M | **146** | 27–29 | 0 | **first run past 54**; stalled 5th girder |
| 23 | lr=2e-5, n_epochs=3, full LSTM warm-start, upper-board rewards | ~1M clean | 58 | ~27 | 0 | stopped: jump-farming bug found |
| 24 | + is_jumping gate on climb bonuses | ~500K | 58 | ~21 | 0 | stopped: height milestone also unfixed |
| 25 | + is_jumping gate on height milestone + _min_y | **42M** | **162** | **38** | **0** | **ended cleanly; best sustained progress** |
| **26** | **warm-start run25, p_curric=0.15, new curriculum metric segmentation** | **—** | **—** | **—** | **—** | **STARTING** |

---

## 11. Current run — Run 26

**Warm-start**: `artifacts/ppo_dkong_run25_last.zip`.
**Save**: `artifacts/ppo_dkong_run26`. **Stack**: 2.
**TensorBoard**: `RecurrentPPO_11` (next run label after run 25's `RecurrentPPO_10`).

**Key changes from run 25:**
- **`p_curric=0.15`** (was 0.0): re-enables wall-zone curriculum. 15% of episodes
  start at heights 35-52 (the 2nd girder approach zone) so the agent drills the
  left-traverse more frequently.
- **Curriculum metric segmentation**: `mame_env.py:_info()` now includes
  `"start_type": "curriculum" | "bottomup"`. `ClimbMetricsCallback` logs
  `climb/height_mean_bottomup` and `climb/height_mean_curric` as separate
  TensorBoard series. This lets you verify the curriculum isn't contaminating
  the bottom-up signal.
- **tests/test_bridge_sync.py**: 4 tests enforce the 47-entry WATCH_ORDER/WATCH_ADDRS
  invariant mechanically. Run after any RAM map change.
- **tests/test_reward.py**: 9 unit tests for `_reward()` — is_jumping gate on
  milestone, climb bonuses, and idle cost; termination conditions.

**Run 25 final state** (what we're warm-starting from):
- 42M steps, height_mean≈38, height_best=162, ep_rew_mean=-5.78
- explained_variance=0.962 (value function well calibrated)
- clip_fraction≈0.109 (healthy for LSTM RecurrentPPO)

**SUCCESS** = `height_mean` rising past 44 (first ladder complete), then 78 (3rd girder).
Watch `climb/height_mean_bottomup` — this is the honest bottom-up signal.
`climb/height_mean_curric` will be higher (starts partway up) and that's expected.

---

## 12. Critical bugs fixed (do not reintroduce)

### Jump-farming of climb bonuses (fixed runs 23–25, 2026-07-01/02)

**The bug**: All three climb bonuses (`FIRST_CLIMB_BONUS`, `CLIMB_BONUS`,
`UPPER_CLIMB_BONUS`) fired whenever `s["mario_y"] < p["mario_y"]` (upward movement)
at the ladder x-position. During a jump arc, `mario_y` also decreases on the upward
half. Mario could stand at x=133-155 and jump repeatedly, getting +0.30 per upward
frame (~+3.6 per jump) without ever pressing UP to climb.

**Evidence**: `height_mean` stuck at 27 with `height_best` spiking — consistent with
jump apexes from the ground floor (~height 27-35) rather than genuine climbing.

**Fix**: All three climb bonuses, the height milestone, and `_min_y` (the height
metric) are gated on `not s.get("is_jumping", 0)`. The `is_jumping` flag (0x6216)
was added to `memory_map.WATCH_ORDER` and `bridge.lua WATCH_ADDRS` (47th entry in both).

**How to verify**: `height_mean` dropped from ~25-27 to ~21 immediately on the first
honest run (run 25 at 368K steps), confirming 4-6 pixels were jump-apex inflation.

---

### Height milestone triggered by jump arcs (fixed run 25, 2026-07-02)

**The bug**: `_reward_max_h` (the milestone tracker) updated during jump arcs.
A ground-floor jump to apex height 33 paid `(33-20)*0.5 = +6.5` milestone credit
for heights Mario only briefly passed through.

**Fix**: `if height > self._reward_max_h and not_jumping:` — milestone only pays
when Mario is not in a jump arc. `_min_y` in `step()` also gated:
`if state["mario_y"] and not state.get("is_jumping", 0)`.

---

### --p-curric flag silently ignored (fixed 2026-07-01)

`mame_env.py:__init__` set `self._p_curric = 0.15` as an **instance** attribute,
shadowing the class attribute set by `DonkeyKongEnv._p_curric = args.p_curric` in
`train.py`. `--p-curric 0.0` ran at 15% curriculum. Runs 22 and early 23 affected.
**Fix**: removed `self._p_curric` from `__init__`. Class attribute only.

---

### CORNER_H_MAX too low — corner penalty never fired (fixed 2026-07-01)

`CORNER_H_MAX = 15` but ground floor is height ≈ 16-20. `height < 15` never true.
`CORNER_X_RIGHT = 190` left large unpenalised zone past the first ladder.
**Fix**: `CORNER_H_MAX` 15→25, `CORNER_X_RIGHT` 190→160.

---

### lr_schedule not overridden on warm-start (fixed 2026-06-29)

`PPO.load()` restores `lr_schedule` (a callable) from the checkpoint.
Setting `model.learning_rate = args.lr` has no effect.
**Fix**: also set `model.lr_schedule = get_schedule_fn(args.lr)` after warm-start.

---

### vx/vy always zero (fixed run 16)

In `step()`, `self._prev = state` was assigned before `_preprocess(pix, state)`.
Inside `_preprocess` → `_build_ram_features`, `prev = self._prev` equalled current
state → `vx = 0`, `vy = 0` for all barrels, every step.
**Fix**: call `_preprocess` first, then update `_prev`.

---

### gamma=0.99 kills clear reward (fixed run 7)

At ~1840 steps/episode, `0.99^1840 ≈ 1e-8`. Clear reward invisible to value function.
**Fix**: `gamma=0.999` → `0.999^1840 ≈ 0.16`.

---

### 0x6200 ("is_dead") unreliable (known since early runs)

Reads 1 while Mario is alive and walking. **Never use for death detection.**
Use `lives` decrement instead.

---

### eval.py loaded RecurrentPPO models as PPO (fixed 2026-06-30)

Used `PPO.load()` on all models; no LSTM state threading in the step loop.
LSTM reset to zero every step — equivalent to a feedforward network during eval.
**Fix**: `_load_model()` tries `RecurrentPPO.load()` first; LSTM state threaded
through step loop with proper `episode_start` flags.

---

### eval.py default --stack 8 mismatched training (fixed 2026-07-01)

**Fix**: eval default changed to `--stack 2` (matches all runs from 21+).

---

### diag.py incompatible with current architecture (fixed 2026-07-01)

Used `VecFrameStack` and `PPO.load()` only.
**Fix**: full rewrite to match `eval.py` — `DkFrameStackWrapper`, RecurrentPPO
auto-detect with LSTM state threading, `--stack` flag.

---

### smoke.py broken with Dict observation space (fixed 2026-07-01)

Called `obs.shape` on Dict obs.
**Fix**: changed to `obs["image"].shape`, `obs["ram"].shape`.

---

### probe.py --headless flag stuck at True (fixed 2026-07-01)

`action="store_true", default=True` — flag couldn't be unset.
**Fix**: replaced with `--no-headless` flag.

---

### MAME X connection crash (fixed run 3)

SDL opened X connection even with `-video none`. Display hiccup → MAME killed →
crashed whole SubprocVecEnv. **Fix**: `SDL_VIDEODRIVER=dummy`,
`SDL_AUDIODRIVER=dummy`, drop `DISPLAY`/`WAYLAND_DISPLAY` for headless processes.

---

## 13. Temporal awareness / barrel timing

**Current state**: Mario has no explicit "is it safe to climb now?" reward signal.
The climb bonus fires whenever Mario is on the ladder moving up — regardless of
whether a barrel is 2 pixels or 80 pixels away.

**What the agent CAN see**: all 6 barrel positions (x, y, vx, vy) in RAM features.
The LSTM can learn to correlate "barrel at x=143, falling" → "wait." But this is
an emergent behavior that requires many death-from-bad-timing experiences to credit-assign.

**Known gap**: `lad53` (barrel x-distance to x=53 ladder) exists as an explicit feature.
No equivalent `lad143` for the first ladder (x=143). Adding it would give an explicit
"barrel approaching my ladder" signal to the LSTM. Deferred because it changes
`RAM_FEATURE_DIM` 62→68, which breaks warm-start.

---

## 14. Recommended next steps (for Fable / new session)

In priority order:

1. **Watch run 26 stabilize**: After warm-start regression (~2-5M steps), watch
   `climb/height_mean_bottomup`. If it stalls at ~38 as run 25 did, the curriculum
   isn't helping and it's not a reward problem — it's an exploration problem.

2. **Go-Explore** (highest-leverage, ~1 week of work): maintain an archive of
   (x, height) cells reached across all episodes; on episode start, reset to the
   frontier cell with lowest known height_mean. This makes hard-to-reach states
   much more frequently practiced. The existing curriculum infrastructure makes
   this tractable — curriculum states are already snapshots, Go-Explore would just
   dynamically update which snapshot to reload.

3. **Potential-based shaping** (complements Go-Explore): define Φ(s) = arc-length
   along the expert corridor, reward `r_shape = γΦ(s') - Φ(s)`. Policy-invariant
   (can't be farmed), provides dense gradient all the way to Pauline without
   changing the optimal policy.

4. **lad143 feature**: add barrel-distance-to-first-ladder (x=143) as a RAM feature
   alongside lad53. Explicit "barrel approaching my ladder" signal for timing.
   Changes RAM_FEATURE_DIM 62→68 — breaks warm-start from any prior run. Do this
   at the start of a fresh run, not as a mid-run change.

5. **Deferred (lower priority)**:
   - VecNormalize (observation normalization — adds complexity to save/load flows)
   - Config dataclass (code quality, doesn't affect agent behavior)
   - Curriculum metric segmentation already done (run 26)

---

## 15. Why the wall persisted (history) + LSTM rationale

The agent never traversed from height 54 to the top with live barrels in 22 runs.

Confirmed behaviorally (watching .inp):
1. Mario grabs hammer, runs LEFT past x=53 (the climb ladder), farms kills at
   the left wall, then dies when hammer expires.
2. Without hammer: camps right side of 2nd girder farming barrel-jumps.
3. Physically passes the ladder location every episode but doesn't climb.

**Root cause**: the left traverse requires knowing whether a barrel passed x=53
in the last 2–3 seconds ("is it safe now?"). An 8-frame stack gave ~0.5s of
visual history — not enough. This was a **temporal memory problem**.

**LSTM broke it**: run 22 (36.5M steps, RecurrentPPO with 256-unit LSTM) reached
height_best=146 — the first run to break 54. The LSTM's temporal context lets it
track barrel state across the ~3s traverse window.

---

## 16. Gotchas (already bit us)

- **Kill MAME**: `pkill -x mame`. **Never** `pkill -f 'mame dkong'` — matches
  your shell and kills it.
- **nohup PID**: `nohup ... &` prints the shell wrapper PID. The Python trainer
  is found via `ps aux | grep dkong_ai.train`. `kill <trainer_pid>` → MAME
  processes die ~5s later (PR_SET_PDEATHSIG). Killing only the wrapper orphans 16 MAMEs.
- **Throughput is GPU-bound** during PPO (not MAME emulation). 16 envs ~500–600
  fps on RTX 4080 SUPER.
- **Obs space breaks warm-start**: models from a different RAM dim or image shape
  cannot be loaded. RAM dim: 62 (as of run 19). Stack: run 21+ uses `--stack 2`.
  Always match `--stack` to the run being loaded.
- **clip_fraction warning**: for LSTM (RecurrentPPO), healthy clip_fraction is
  0.05–0.15. Above 0.20 means `--lr` is too high and the policy is thrashing.
  Run 21 hit 0.34 (lr=2.5e-4); run 22 stuck at 0.20-0.23 (lr=5e-5); runs 23-25
  at lr=2e-5 → 0.10-0.13 (healthy).
- **Warm-start regression**: inheriting weights trained on a different reward
  function causes a dip from ~50 to ~25 in height_mean as PPO corrects toward
  the new objective. Normal; wait for recovery. Run 26 warm-starts from a SAME
  reward function so regression should be minimal — just curriculum re-adaptation.
- **WATCH_ORDER / WATCH_ADDRS must match**: `memory_map.WATCH_ORDER` (Python)
  and `WATCH_ADDRS` in `bridge.lua` (Lua) must have entries in the same order.
  Currently 47 entries each. `is_jumping` is the 47th (last) in both.
  `tests/test_bridge_sync.py` enforces this mechanically.
- **Recording + state loads don't mix**: `record=True` uses intro/soft-reset.
  `record=False` uses save-state load (fast). A load isn't an input event → breaks
  `.inp` playback.
- **Stream framing**: `_rxbuf` keeps bytes that over-read from the handshake into
  the first obs frame. Don't remove it — causes intermittent `IndexError` at
  16-env launch.
- **start_type in info**: `_info()` returns `"start_type"` but it's only meaningful
  at episode end (when the callback reads it). The value from `step()` reflects the
  start type set during the last `reset()`, which is correct.

---

## 17. File map

`dkong_ai/`:
- `mame_env.py` — env: MAME launch, socket bridge, obs build, reward, curriculum.
  `_info()` returns `{"state", "max_height", "cleared", "start_type"}`.
- `memory_map.py` — all confirmed RAM addresses + score decode. 47 WATCH_ORDER entries.
- `dk_policy.py` — `DkFeaturesExtractor` (CNN+RAM MLP) + `DkFrameStackWrapper`.
- `train.py` — RecurrentPPO training. Callback logs `climb/height_mean_bottomup`
  and `climb/height_mean_curric` in addition to the existing metrics. Flags:
  `--n-envs`, `--stack`, `--gamma`, `--ent-coef`, `--init-from`, `--p-no-barrels`,
  `--p-curric`, `--save`, `--timesteps`, `--lr`, `--n-epochs`, `--lstm`,
  `--lstm-hidden`, `--transfer-features-from`.
- `eval.py` — eval + record .inp. Flags: `--model`, `--stack`, `--port`,
  `--episodes`, `--p-no-barrels`, `--p-curric`.
- `diag.py` — death/peak position diagnostic (RecurrentPPO-aware, stack=2).
- `smoke.py` — quick sanity check (Dict obs-aware).
- `probe.py` — MAME field discovery. Use `--no-headless` for windowed mode.
- `extract_bc.py` / `train_bc.py` — behavioral cloning pipeline (built run 5,
  did not improve over pure RL; kept for reference).

`scripts/`:
- `bridge.lua` — MAME lock-step bridge. 47 WATCH_ADDRS. Supports EXTRACT mode
  (BC), barrel freeze (0xF8/0xF7), curriculum loads (0xE0+i).
- `playback.sh` — watch a .inp windowed.
- `human_record.sh` — record human play.
- `make_curriculum.lua` — snapshot curriculum states from a demo replay.

`tests/`:
- `test_bridge_sync.py` — 4 tests: WATCH_ORDER count, WATCH_ADDRS count, address
  match, is_jumping last. Run after any RAM map change.
- `test_reward.py` — 9 unit tests for `_reward()`. Tests is_jumping gate, idle
  cost, milestone, death/clear termination. No MAME required.

`artifacts/`:
- `checkpoints/<run>/` — PPO checkpoints every 500k steps.
- `expert_corridor.json` — height→x route corridor from expert demo.
- `states/dkong/curric_*.sta` — MAME save-states for curriculum.
- `ppo_dkong_run22_last.zip` — run 22 recovery (height_best=146, 36.5M steps).
- `ppo_dkong_run25_last.zip` — run 25 final checkpoint (42M steps, height_mean≈38).

`demos/dkong.inp` — expert demo (MAME 0.241, plays back faithfully on 0.264).
`logs/run25.log` — run 25 training log (complete).
`logs/run26.log` — run 26 training log (current).
`logs/` — TensorBoard event files. RecurrentPPO_10=run25, RecurrentPPO_11=run26.
