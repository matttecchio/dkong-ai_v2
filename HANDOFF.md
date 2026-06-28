# Donkey Kong RL — Handoff / Complete Project State

**Single source of truth.** Read this before changing anything — several mechanisms
are non-obvious and easy to regress. Pairs with `README.md` (quick reference).

Last updated: 2026-06-28, after Run 18 launch.

---

## 0. TL;DR

- **Full pipeline works and is robust**: MAME `dkong` driven from Python, a
  Gymnasium env over a socket bridge, PPO (Stable-Baselines3) on pixels+RAM,
  reward from RAM. 16 parallel envs, ~600–900 fps, runs 30M steps overnight
  with 0 crashes.
- The agent learns to jump barrels, score, and climb roughly half the board.
  With barrel-free curriculum episodes it can clear the stage (~10% of training
  episodes). **Bottom-up with live barrels: still 0 clears as of run 18.**
- The persistent blocker: the agent reaches height ~53–54 (2nd girder) then
  **either farms barrel jumps on the right, or grabs the hammer, runs left past
  the ladder, waits at the left wall, and dies when the hammer expires.**
- Run 18 is the current run: 70% barrel-free episodes, stronger girder-level
  rewards, farming-death penalty, corner penalty, height bonus. The code is
  also ready for run 19 (all 5 fireballs tracked, obs space change).

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
# Train (16 envs, 30M steps, warm-start optional)
.venv/bin/python -m dkong_ai.train --rom-dir ./roms \
    --save artifacts/ppo_dkong_runN --stack 8 --gamma 0.999

# Watch a trained model (records .inp, then plays windowed)
.venv/bin/python -m dkong_ai.eval --rom-dir ./roms \
    --model artifacts/checkpoints/ppo_dkong_runN/ppo_dkong_runN_Xsteps \
    --port 5100 --stack 8
./scripts/playback.sh artifacts/recordings/<file>.inp

# Bottom-up live-barrel eval (no curriculum, no barrel-free episodes)
# Run inline Python: set P_NO_BARRELS=0.0, _p_curric=0.0, load model, 10 eps

# Monitor running train
grep -E "total_timesteps|ep_rew_mean|height_mean|height_best|clear_rate" \
    /tmp/dk_run18.log | tail
.venv/bin/tensorboard --logdir logs
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
  static threat/ladder/fall-zone map (see §6). Stacked ×8 by
  `DkFrameStackWrapper` → `(84, 84, 16)` at policy input.
- `"ram"`: `(50,)` float32 — normalised RAM features (see §5).

**Policy** (`dkong_ai/dk_policy.py`):
- `DkFeaturesExtractor`: NatureCNN on image → 256 features; Linear MLP on RAM
  → 64 features; concat → 320 features → PPO policy/value heads.
- `DkFrameStackWrapper`: stacks `image` across N frames, passes `ram` from the
  latest frame only.

**Actions** (8): noop, L, R, U, D, jump, jump+L, jump+R.

**Bridge control bytes** (not agent actions): `0xF1` coin, `0xF2` start,
`0xFE` soft-reset, `0xFD` clean-quit, `0xFC` save, `0xFB` load, `0xE0+i`
load curriculum state i, `0xF8` freeze barrels, `0xF7` unfreeze barrels.

---

## 5. RAM features (`dkong_ai/memory_map.py` + `mame_env.py:_build_ram_features`)

**50 features** (layout: `[mario_x/255, mario_y/240]` + 6 barrels × 7 + 1
fireball × 3 + hammer × 3):

Per barrel: `[Δx/128, Δy/120, vx/8, vy/20, lad53/64, edge_dist, active]`
- `vx/vy`: per-step velocity (frameskip=4); horiz norm ÷8, vertical ÷20.
- `lad53`: barrel x-distance to the critical left ladder at x=53 (norm ÷64).
  Tells agent whether a barrel is heading for that ladder column.
- `edge_dist`: normalised distance to the girder edge the barrel is heading
  toward (0 = at edge / about to fall, 1 = far away).

⚠️ **Run 19 queued change**: RAM → 62 features (all 5 fireball slots tracked,
currently only slot 0). Requires fresh start.

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
| fireball0_st/x/y | 0x6400/03/05 | only slot 0 currently watched (5 exist) |

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
| Waypoints (6) | +5/10/30/8/8/8 | zig-zag route milestones; fire once per episode |
| Traverse progress | +0.05/pixel moved left | while height 36-65, x 53-143 |
| Climb bonus | +0.30/step | while ascending at x=43-68, height 40-100 |
| Ladder idle cost | −0.05/step | in climb zone but y unchanged |
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
| 18 | **70% barrel-free** + farming death penalty + corner penalty + girder milestones | **active** | TBD | TBD | **current run** |

---

## 10. Run 18 — current run

**PID**: 180275. **Log**: `/tmp/dk_run18.log`.
**Save**: `artifacts/ppo_dkong_run18`. **Stack**: 8.

Key parameters vs run 17:
- `--p-no-barrels 0.70` — 70% of episodes barrel-free (was 0.15)
- `--init-from artifacts/checkpoints/ppo_dkong_run17/ppo_dkong_run17_3000000_steps`
- Farming death penalty: extra −5 if episode max height < 40 at death
- Corner penalty: −0.03/step at bottom floor corners (height<15, x<30 or x>190)
- Girder milestones: +10/30/40/55/70 on first reaching each girder

**Plan**: let run to ~5M steps → bottom-up eval. If clear_rate rising:
keep going. If still walled: reduce `--p-no-barrels` to 0.40 for run 19
(gradually reintroduce barrels) and apply queued run-19 changes (all 5
fireballs, RAM 50→62, fresh start).

---

## 11. Critical bugs already fixed (don't reintroduce)

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

### MAME X connection crash (fixed run 3)
Even with `-video none`, SDL opened an X connection to WSLg display. Display
hiccup → MAME killed → `ConnectionError` crashed the whole `SubprocVecEnv`.
**Fix**: `SDL_VIDEODRIVER=dummy`, `SDL_AUDIODRIVER=dummy`, drop
`DISPLAY`/`WAYLAND_DISPLAY` for headless processes.

---

## 12. Run 19 — queued changes (not yet launched)

Requires fresh start (obs space change: RAM 50→62).

- **All 5 fireball slots tracked** (currently only slot 0). `memory_map.py`
  already updated: `fireball0..4_st/x/y`. `_build_ram_features` loops over 5.
  Adds 12 features → RAM_FEATURE_DIM 50→62.
- Corner penalty (already in code, active from run 18 onwards via reward).
- Girder milestones (already in code).

Launch command (after run 18 completes or is stopped):
```bash
nohup .venv/bin/python -m dkong_ai.train \
  --rom-dir ./roms --timesteps 30000000 --n-envs 16 \
  --save artifacts/ppo_dkong_run19 --logdir logs \
  --gamma 0.999 --ent-coef 0.01 --stack 8 \
  --p-no-barrels 0.40 \
  > /tmp/dk_run19.log 2>&1 &
```
(Reduce `--p-no-barrels` from 0.70 to 0.40 — start reintroducing barrels.)

---

## 13. Why the wall persists (diagnosis)

The agent has **never traversed from height 54 to the top with live barrels.**
Without that experience, the value function at the wall state cannot represent
"there is a +300 reward path above me." It only knows farming ≈ +65/ep.

Confirmed behaviorally by watching .inp recordings:
1. Mario grabs hammer, runs LEFT past x=53 (the climb ladder), and farms hammer
   kills at the left wall.
2. Stands at the left wall until hammer expires, then dies to a barrel.
3. He physically passes through the ladder location every episode but doesn't stop.

The value function needs experience of the path above the wall to assign credit.
Possible paths forward:
- 70% barrel-free (run 18): let agent solidly learn the route without threat,
  then gradually reintroduce barrels.
- Raise girder milestone rewards further so even one crossing attempt on a
  barrel-free episode is worth far more than a farming episode.
- Recurrent policy (LSTM) for longer-horizon barrel-grouping strategy.

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
  cannot be loaded. RAM changed: run 14 (26→44 initially, then 44, then 50).
  Stack changed: runs 15+ use stack=8. Always specify `--stack 8` for runs 15+.
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
- `train.py` — PPO training. Flags: `--n-envs`, `--stack`, `--gamma`, `--ent-coef`,
  `--init-from`, `--p-no-barrels`, `--save`, `--timesteps`.
- `eval.py` — eval + record .inp. Flags: `--model`, `--stack`, `--port`, `--episodes`.
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
