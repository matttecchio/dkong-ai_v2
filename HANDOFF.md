# Donkey Kong RL — Handoff / Complete Project State

**Single source of truth.** Read this before changing anything — several mechanisms
are non-obvious and easy to regress. Pairs with `README.md` (quick reference).

Last updated: 2026-07-09 night, **Run 27aj active** — the video-to-expert
loop is now the project's sharpest tool: film the policy failing → the user
(world-class player) diagnoses in one viewing → fix ships within the hour.
Round 1 found the 000-BONUS-TIMER wall (pro point-max play leaves late-route
states with seconds of clock; nine time-starved WC cells pruned, §11b).
Round 2 found three behavioral bugs at the mid-board wall: untaxed mid-board
stub ritual-climbing, pacing at the top-rung INVISIBLE BARRIER (measured:
hard stop at x=107), and dismounting the final ladder — all three fixed in
27aj (generalized stub tax, top dead-end tax, final-ladder commitment).
Honest floor (CLEAN bottom-up mean, excluding self-punishing glitch-guard
deaths) is at 39+, past the old all-time baseline. Still 0 honest clears.
TARGETS (user, world-class DK player): ≥90% barrel-board clears, eventually
kill screen (L22/screen 117); post-L4 barrels+springs ≈ 100%, pies/rivets
tolerate genuine no-win scenarios.

---

## 0. TL;DR

- **Full pipeline works and is robust**: MAME `dkong` driven from Python, a
  Gymnasium env over a socket bridge, RecurrentPPO (LSTM) on pixels+RAM, reward
  from RAM. 16 parallel envs, ~500–600 fps, runs overnight with 0 crashes.
- **Run 28e ACTIVE** (TB `RecurrentPPO_55`, launched 2026-07-11): the run-28
  line is the **capacity bundle** (LSTM 512, RAM MLP 128, difficulty in obs
  = RAM 75, rehearsal cap K=8, CNN transferred / heads fresh / levels
  reset) plus, in order: **28b** lr 5e-5→1e-4 (solo hot-lr test — all 12
  frontiers flat 0-1% on 12h-old heads; guardrail: clip>0.25 sustained →
  revert); **28c** stochastic 50/50 {0,8} spawn burn-in (film review #3 +
  beeline probe PROVED the fixed 8-step burn-in phase-shifted runs into
  deterministic traffic doom on fixed-RNG cells — the policy already knew
  the routes); **28d** CUMULATIVE glitch guard (film review #4: the
  3-consecutive-streak guard was ratchet-evaded — 100% of ALL bottom-up
  height was x=99 stub glitch; **every prior floor metric is retracted**,
  honest floor = 0); **28e** APPROACH REPLAY (verified per-cell approach
  bytes force-executed at spawn: warm LSTM + correct phase + zero timer
  cost; `augment_approaches.py`) + two PROGRESS-GATED FLOOR CHAINS (WC
  cells h0-51, +40px gate — the first direct floor curriculum).
  Dials: lr 1e-4, ent 0.01, p_curric 0.8, thresh 0.3, `backward_dense13`
  (14 chains), governor 0.40/0.48, levels in `<backward-dir>/levels.json`.
- **Standing diagnostic rules** (proven this week, run by the monitoring
  protocol): DOOM TRIAGE — any frontier cell at 0% over 500+ clean-spawn
  draws gets a scripted beeline probe (clears = policy problem, train on;
  all-die = retire/re-mine the cell); FILM PIPELINE — any frontier flat
  >6h gets a `film_cells.py` reel for human review. Two film reviews found
  two self-built walls in two days (burn-in phase doom; guard ratchet).
- **Scoreboard**: `eval_battery.py` daily (bottom-up det+stoch clean
  means + key-cell probes split by burnin/approach) → logs/battery/.
- **Deployment is scripted**: `scripts/current_launch.sh` is THE canonical
  launch command (edit dials there only); `scripts/auto_resume.sh` relaunches
  from the newest checkpoint after a reboot/crash (guards: `.maintenance`
  file for intentional stops, flock, 3-per-2h crashloop cap). Touch
  `.maintenance` BEFORE any manual kill. Crontab entries (@reboot + 10-min
  watchdog) designed but NOT yet installed — see §3.
- **2026-07-05, the x=99 glitch (§12)**: policy AND go-explore winners climbed
  a broken ladder stub with frame-perfect inputs (user spotted it on film;
  census: 20/20 bottom episodes). Guard now ends off-ladder climbing as a
  death; `climb/glitch_kill_rate` tracks unlearning. Honest bottom baseline
  reset ~40 -> ~26. Winner routes below h~80 are tainted — if the walk-back
  stalls there, re-run phase 1 with the guard (it lives in the env).
- **2026-07-04, the spawn bug (§12)**: `--p-curric`/`--p-no-barrels` NEVER
  reached the workers — every 27-series run before 27i trained at 15%
  curriculum (not 80%) with 15% barrel-free episodes (not 0%). The barrel-free
  bottom climbs faked `clear_rate_bottomup` 0.04→0.14; 425 controlled
  live-barrel evals measured 0 clears. Fixed `da6b2dc`; the metric now
  excludes `no_barrels` episodes.
- All height metrics honest: gated on `is_jumping==0` so jump arcs don't inflate
  `height_best`, `height_mean`, or the height milestone reward. Per-episode
  audit trail: `logs/episodes/dk_<port>.monitor.csv` (start_y, start_screen,
  end_screen, bw_pos, no_barrels) — check any surprising aggregate there first.
- **Bottom-up with live barrels: still 0 honest clears** (~430M+ steps) —
  and as of 2026-07-11 the honest floor is **0**: a trajectory audit
  (film review #4) showed 100% of all bottom-up height gain came through
  the x=99 broken-stub ratchet, so every historical floor number
  (including the old line's 38-48 "honest band") was glitch-inflated.
  Real floor skill is being built for the first time by the progress-gated
  floor chains. Mid-tower competence is real and verified per-cell.
  The establishment-wall signature to know: curriculum spawn → ~9s
  dodge-in-place → death with ZERO height gain — an LSTM cold-drop +
  traffic-phase problem, now addressed by approach replay (verified
  arrival context) with stochastic burn-in as the fallback.

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
# THE canonical launcher (run 28 dials live HERE — edit this file, nothing else):
#   scripts/current_launch.sh <init-from-model> <log-file>
# Resume after a crash/reboot (newest checkpoint by mtime; silent no-op if
# a trainer is already running or .maintenance exists):
./scripts/auto_resume.sh

# Live run identifiers (maintained by auto_resume / deploys):
cat logs/run_current.pid      # trainer PID
tail -f logs/run_current.log  # symlink to the live log

# STOPPING ON PURPOSE (or any manual kill+relaunch):
touch .maintenance            # FIRST — or the watchdog relaunches behind you
kill -SIGTERM $(cat logs/run_current.pid)   # saves ppo_dkong_run28_last.zip
# known shutdown mode: _last saves within seconds, then the process can HANG
# in env-close — after ~60s, SIGKILL the literal PID (the save is already on
# disk). Then: rm .maintenance
# NEVER pkill -f anything (self-match footguns, twice); kill literal PIDs.

# Crash-resume crontab (designed + tested, NOT yet installed — needs user):
# (crontab -l; echo '@reboot sleep 90 && /home/claw3/dkong-ai/scripts/auto_resume.sh >> /home/claw3/dkong-ai/logs/auto_resume.log 2>&1'; echo '*/10 * * * * /home/claw3/dkong-ai/scripts/auto_resume.sh >> /home/claw3/dkong-ai/logs/auto_resume.log 2>&1') | crontab -

# Watch a trained model (records .inp, then plays windowed)
.venv/bin/python -m dkong_ai.eval --rom-dir ./roms \
    --model artifacts/checkpoints/ppo_dkong_run28/ppo_dkong_run28_Xsteps \
    --port 5100 --stack 2
./scripts/playback.sh artifacts/recordings/<file>.inp

# TensorBoard (WSL2: bind to 0.0.0.0 so Windows browser can reach it)
# Open http://localhost:6006 in Windows browser. Run 28e = RecurrentPPO_55.

# Daily scoreboard (honest floor + key cells; ~20 min on port 5100):
.venv/bin/python -m dkong_ai.eval_battery --rom-dir ./roms --model <newest>

# Film a reel of stuck cells for human review:
.venv/bin/python -m dkong_ai.film_cells --rom-dir ./roms --model <newest> \
    --cells 4:40,11:40 --eps 5 --avi reel.avi   # then ffmpeg -> mp4

# Augment a manifest with verified approach bytes (dense12 -> dense13 was):
.venv/bin/python -m dkong_ai.augment_approaches --src <dir> --out <dir>
```

⚠️ Eval/diag always use `--port 5100` to avoid colliding with training (5000+).
⚠️ Run-27-era checkpoints do NOT load at current code (capacity change, §12) —
check out a pre-`7dea1c4` commit to eval them.

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
- `"ram"`: `(75,)` float32 — normalised RAM features (see §5).

**Policy** (`dkong_ai/dk_policy.py`):
- `DkFeaturesExtractor`: NatureCNN on image → 256 features; Linear MLP on RAM
  → 128 features; concat → 384 features → LSTM → RecurrentPPO policy/value heads.
- `DkFrameStackWrapper`: stacks `image` across N frames (run 21+: **stack=2** —
  optical flow only; LSTM handles long-range temporal memory), passes `ram` from
  latest frame only.
- **Run 21+**: `RecurrentPPO` (`sb3_contrib`) with `MultiInputLstmPolicy`.
  Run 28+: LSTM hidden **512**, RAM MLP **128** (was 256/64 through run 27 —
  bumped for interference reduction after the hollow-tower diagnosis, §11b).

**Actions** (8): noop, L, R, U, D, jump, jump+L, jump+R.

**Bridge control bytes** (not agent actions): `0xF1` coin, `0xF2` start,
`0xFE` soft-reset, `0xFD` clean-quit, `0xFC` save, `0xFB` load, `0xE0+i`
load curriculum state i, `0xF8` freeze barrels, `0xF7` unfreeze barrels.

---

## 5. RAM features (`dkong_ai/memory_map.py` + `mame_env.py:_build_ram_features`)

**75 features** (layout: `[mario_x/255, mario_y/240]` + 6 barrels × 9 + 5
fireballs × 3 + hammer × 3 + difficulty/5):

Per barrel: `[Δx/128, Δy/120, vx/8, vy/20, lad53/64, edge_dist, active, crazy, blue]`
- `vx/vy`: per-step velocity (frameskip=4); horiz norm ÷8, vertical ÷20.
- `lad53`: barrel x-distance to the critical left ladder at x=53 (norm ÷64).
  Tells agent whether a barrel is heading for that ladder column.
- `edge_dist`: normalised distance to the girder edge the barrel is heading
  toward (0 = at edge / about to fall, 1 = far away).
- `crazy/blue` (run 27w+): the game's own barrel-type flags — wild (crazy)
  barrels bounce vertically so every rolling-barrel prior is wrong for them;
  blue barrels end in the oil drum → fireball spawn.

Per fireball: `[Δx/128, Δy/120, active]` — all 5 slots tracked.

`difficulty/5` (run 28+): internal difficulty 0x6380 (1-5). Wild-barrel
behaviour is regime-dependent (user lore: 3-4 = worst/most erratic, 5 =
predictable with a static neutral-jump counter) — this makes play
regime-conditional instead of averaged across regimes.

**⚠️ Missing feature (known gap):** There is no `lad143` — barrel distance to
the x=143 first ladder. `lad53` helps time the 2nd→3rd girder climb; an
equivalent for x=143 would help time the first-ladder climb between barrels.
Consider adding as a future improvement (changes `RAM_FEATURE_DIM` 62→68,
breaks warm-start from run 25).

**Full RAM address map** (`memory_map.py` + `bridge.lua` WATCH_ADDRS, 60 entries,
ORDER MUST MATCH between both files):

| name | addr | notes |
|---|---|---|
| lives | 0x6228 | death = decrement (RELIABLE) |
| screen_id | 0x6227 | 1=barrels 2=pie 3=elevator 4=rivet |
| mario_y | 0x6205 | smaller=higher; start≈240, top≈58 |
| mario_x | 0x6203 | +right |
| is_dead | 0x6200 | **INVERTED**: 1=alive, 0=dead — use lives for death |
| game_start | 0x622C | 1 once game is underway |
| score_100..100k | 0x7721/41/61/81 | tile RAM digits; digit = byte low nibble |
| barrel0..5_st/x/y | 0x6700+ stride 0x20 | status (0=inactive,1=rolling,2=deploying) |
| fireball0..4_st/x/y | 0x6400+ stride 0x20 | all 5 slots tracked |
| hammer_x/y | 0x6A1C/1F | hammer pickup position |
| has_hammer | 0x6217 | 1 while wielding hammer |
| **is_jumping** | **0x6216** | **non-zero during jump arc; used to gate rewards** |

⚠️ `WATCH_ORDER` and bridge.lua `WATCH_ADDRS` are **append-only** and must stay
in sync (60 entries: 47 original + 12 barrel type flags (27w) + difficulty
(27x)). `tests/test_bridge_sync.py` enforces the count and the appended-entry
positions mechanically — run it after any WATCH change.

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
- **RNG diversity**: after each load, advance random NOOP exchanges so barrel
  patterns differ per episode — BOTTOM STARTS ONLY (0-20 exchanges; idling at
  a curriculum cell was a death sentence, see §12 jitter-death).
- **Spawn burn-in (run 28+)**: for the first `BURN_IN_STEPS=8` steps of a
  curriculum episode starting BELOW h172, `step()` executes NOOP regardless
  of the chosen action — the LSTM fills with real observations before
  decisions count (a state-load drops the policy mid-traffic with zeroed
  memory; the establishment-wall signature was 9s dodge-in-place, zero
  height gain). Skipped at top-girder cells: they sit in the barrel-spawn
  lane where standing still kills, and they clear 90-100% without help.
  Trade-off: burn-in transitions store the chosen action but executed NOOP
  (~2% of steps, accepted off-policy noise).
- **Rehearsal cap (run 28+)**: `BW_REHEARSAL_CAP=8` — the non-frontier 50%
  of curriculum draws sample the 8 cells just above the frontier, not the
  whole passed window. Uncapped windows starved per-cell maintenance
  (~50%/level of chain draws → tiers gated at 0.31 decayed to 0-13%; the
  hollow tower, §11b).
- **Approach replay (run 28e+)**: cells carrying an `approach` entry (from
  `augment_approaches.py` — replay-verified anchor + action indices) load
  the mid-leg anchor instead and `step()` FORCE-EXECUTES the proven arrival
  actions before handing control over (`HANDOVER_JITTER=6` random suffix
  drop = diversity along a proven-survivable path). Warm LSTM + correct
  traffic phase + zero bonus-timer cost. Ordering trap (hit twice — burn-in
  28c, approach 28e): anything set during a curriculum load must be stashed
  and applied AFTER `_begin_episode`, which zeroes per-episode state.
  `approach_len` CSV column attributes episodes.
- **Progress-gated chains (run 28e+)**: a manifest chain with
  `"gate": "progress"` gates on `max_height - start_height >= 40px`
  instead of clears (BackwardCallback.PROGRESS_PX). Makes LOW cells usable
  as curriculum — a clear-gate from h20 could never advance. Floor chains
  (slots 12/13) are WC cells h0-51 ordered BOTTOM-FIRST so level 0 drills
  the floor traffic-crossing immediately.
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
| 26 | warm-start run25, p_curric=0.15, curriculum metric segmentation | 40M | 193 (once @7M) | 36-38 flat | 0 | converged-flat at lr 2e-5; curric spawns gained ~0px → dodge-skill deficit proven |
| — | **GO-EXPLORE PIVOT** (phase 1: no NN, CPU random search + state banking) | 7.8M+0.9M explore | **192 (top)** | n/a | **418+47 verified winners** | first-ever bottom-up live-barrel clears; 11 min to first winner |
| 27 | **phase 2 backward algorithm**, warm-start run26, lr 5e-5, p_curric 0.8 | ~1M | — | — | curric 0.53@L0 | level 0→1 @336K — first trained-policy live-barrel clears; restarted: slot-clobber bug |
| 27b | + slot backup fix (honest bottomup labels) | ~7M | — | ~35 | curric ~0.3 | stalled level 1: 20% of curric states frozen + tier-1 "blind spots" |
| 27c | + verified manifest (13 frozen dropped), thresh 0.3, frontier metric | 18M | — | ~35 | frontier ~0 | stalled level 2 17M steps → exposed the REAL bug ↓ |
| **27d** | **single-life episodes (`done = died or ...`)** | 3.2M | — | ~43 honest | **level 3 in 2M steps** | multi-life episodes were the phase-2 wall; walk-back moving |
| 27e-27aj | per-chain frontier gating, dense3-11, obs 62→74, level-reset rule, WC chain, 000-timer pruning, film-review fixes | ~2 weeks | 193 | floor 3.4→48.6 clean | curric ≤0.24; bottomup 0 | full story §11b; ended in the hollow-tower stall |
| 27ak | crash recovery (WSL reboot #2), auto-resume infra | 14.1M | 192 | floor 47-48.6 record | 0 | stable but 3 advances/night — tower stalled at the h163-168 shelf |
| 28 | **capacity bundle**: LSTM 512, RAM 128, difficulty obs (75), spawn burn-in, rehearsal cap, dense12, levels reset | 20M | 192 | retracted | 0 | tower rebuilt (63 gates day 1); retention PERFECT (hollow tower cured); frontiers then flat 0-1% at 12h |
| 28b | lr 1e-4 (solo hot-lr test) | ~1h | — | retracted | 0 | approved after 5h frontier flatline; clip warm but outcomes positive throughout |
| 28c | stochastic 50/50 burn-in (film #3 + beeline probe: fixed 8-step burn-in = phase doom on fixed-RNG cells) | ~14M | 192 | retracted | 0 | WC chains unlocked in 2h (0→level 4); "floor breakout" 10→40... |
| 28d | **cumulative glitch guard** (film #4: streak guard ratchet-evaded; 100% of floor height was stub glitch) | ~3M | — | **honest floor = 0** | 0 | ALL prior floor metrics retracted; 80% of bottomups die to guard while unlearning |
| **28e** | **approach replay** (27 verified cells) + **progress-gated floor chains** (first direct floor curriculum) | **active** | — | 0 (rebuilding honestly) | 0 | three-way attribution live: approach vs clean-spawn vs burn-in |

---

## 11. Run 26 — the last pure-RL run (superseded by Go-Explore, §11b)

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

**Run 26 OUTCOME (2026-07-03, 40M steps): FLAT.** `height_mean_bottomup` oscillated
35-42 with no trend; clear_rate 0 throughout; entropy/KL/score all converged-flat at
lr 2e-5. Decisive new fact from the metric split: `height_mean_curric` ≈ 42-45 while
curriculum spawns average height ~44 — since max-height ≥ spawn height, the agent gains
~0-1px from mid-traverse spawns, i.e. it dies almost instantly in barrel traffic at the
wall. The deficit is **dodge-survival skill in traffic**, not route knowledge.
Eval @40.5M (5 eps, live barrels): max heights 4-54, scores 0-100.

---

## 11b. Go-Explore pivot (2026-07-03) — CURRENT DIRECTION

`dkong_ai/go_explore.py` — classic policy-free Go-Explore phase 1 (no NN, no GPU,
CPU-only, ~1100 steps/s with 6 workers on ports 5200+). Archive of cells keyed
`(mario_x//8, height//8, has_hammer)`, each an immutable 2KB MAME save-state
(`artifacts/go_explore/cells/cell_N.sta`) + exact action-byte trajectory from its
parent. Workers loop: select under-visited cell (count/height/chain-length weights) →
restore (copy .sta onto slot `dk_<port>.sta` + fixed prologue 3×LOAD, 2×NOOP,
UNFREEZE — **no bridge changes**) → ~100 sticky-random steps → snapshot every new cell.
Snapshot command bytes are appended to the trajectory so `restore(parent)+bytes` lands
frame-exactly on the child state (generational stitching). Mid-death-animation cells
retire via early-death stats. Success = `screen_id` leaves 1 with lives>0 → winning
byte trajectory saved in `archive.json`, auto-verified by deterministic replay.

Validated (2026-07-03): 150-step restore determinism PASS; cross-port .sta round-trip
PASS. Launch: `python -m dkong_ai.go_explore --rom-dir ./roms --workers 6`
(`--validate` self-test; archive resumes from `archive.json`).

**Phase 1 RESULTS (2026-07-03)**: two archives, both with verified bottom-up
live-barrel clears (screen_id 1→4, all lives intact) — `artifacts/go_explore_run1/`
(6 workers, ~11 min to first winner, 47 winners) and `artifacts/go_explore/`
(18 workers, seed 7, first winner at 6 min, 418 winners, ~2,970 steps/s CPU-only).
What 26 PPO runs / 250M+ steps never did, random search + state banking did in
minutes — the wall was pure exploration, not capability.
Winner videos: `dkong_ai/replay_winner.py` replays a winner's ancestor chain
seamlessly (each restore lands on the state the machine is already in) with MAME
`-aviwrite` → ffmpeg mp4. See `artifacts/recordings/first_clear_run{1,2}.mp4`.
A true .inp is impossible for stitched winners (playback replays inputs only).

**Phase 2 (backward algorithm) — BUILT, RUNNING as run 27**:
- `dkong_ai/export_chains.py`: archives → `artifacts/backward/{manifest.json,*.sta}`;
  dedupes winners by distinct final cell; always overwrites state files; refuses to
  write an empty manifest.
- `mame_env.py`: `backward_manifest` ctor param (requires record=False, empty
  manifest disables with a warning); `load_state_file()` is THE primitive for
  loading an arbitrary .sta through the slot — it restores the `bottom_<port>.sta`
  backup after the load so "slot file == bottom start" always holds (slot-clobber
  bug class); missing files raise RuntimeError (fail-fast, not OSError→recover
  storm); `set_backward_level(k)` widens the start window [n-1-k, n-1].
- `train.py`: `--backward-dir` + `BackwardCallback` (walk back one cell when
  rolling curric clear rate ≥ 0.5 over 64 episodes); logs `climb/backward_level`,
  `climb/backward_clear_rate`, `climb/clear_rate_bottomup` (the honest metric).
- Run 27 history: 27 (slot-clobber found) → 27b (stalled: frozen states +
  multi-life noise) → 27c (verified manifest, thresh 0.3, frontier metric;
  exposed the multi-life bug) → 27d (single-life episodes: walk-back genuinely
  descending for the first time) → 27e (frontier-gated promotion: advance on
  the deepest tier's own clear rate, not the window-diluted mix) → 27f
  (per-chain levels; widened post-load RNG jitter — with a units bug) → 27g
  (jitter units fix `c0cc81a`: 0–20 exchanges, not 0–47 ≈ 3.1s of idling;
  6 chains promoted; "bottom-up clears" 0.04→0.14 appeared — **phantoms**, see
  §12 spawn bug) → 27h (per-episode CSV instrumentation `9d29df3`; caught the
  phantom clears in 3 minutes: all `no_barrels=True`) → **27i ACTIVE**
  (TB `RecurrentPPO_20`, spawn fix `da6b2dc`): first run at the real 80%
  curriculum / 0% barrel-free (measured 76%/0% in worker CSVs); chains
  re-promote within minutes of launch. Watch `climb/backward_level` and — now
  trustworthy — `climb/clear_rate_bottomup` off 0. See §12 for the curriculum
  bugs — do not reintroduce.
- 27j–27q (2026-07-04/05): jitter-death fix, governor + level persistence,
  x=99 glitch guard, honest baseline reset, `backward_dense`/`dense2`
  (choke-band `--densify 130:190:5`, `--prune-descents 15`), frontier share
  0.5→0.7. Ended stuck: 27q/27r walled at the c446 pool (h167-168, ~8% over
  4k draws) with promotions frozen.
- **27r post-mortem (2026-07-05 eve)** — the triple deadlock, found via
  per-cell CSV audit (`bw_pos` isolates tiers): (1) 0.7 frontier share ground
  an ~8% cell with 70% of draws; (2) that gradient DECAYED the adjacent
  mastered tier 43%→5% (states 4 macro-steps apart — interference), even a
  once-passed tier regressed; (3) pooled rehearsal pinned ~0.55 < CONSOL_OFF
  0.68 → promotions frozen the entire run. Freezing stops promotion, not
  decay; only rehearsal share stops decay.
- **27s (dense3)**: new `densify_stuck.py` minted doom-screened rungs INSIDE
  the c446 gap (j=1-3 of each successor leg) + share back to 0.5. Six
  advances in 35 min — chains 5 and 9 passed c446 itself — then the governor
  re-froze on COMPOSITION (each hard promotion enters rehearsal at ~0.3,
  pooled equilibrium sank to ~0.47-0.57; per-cell audit showed every tier
  RISING while frozen). Tier decay reversed (5%→30%) — the share, not the
  freeze, was the protection.
- **27t**: governor recalibrated 0.40/0.48 (thresholds must track the
  tower's difficulty mix). 9 more advances, zero freezes: all five trunk
  chains cleared c446 AND c446_d4, converging at c433 (h174).
- **27u (dense4)**: run-1 (a0) chains 0-3 were a write-off —
  un-densifiable (their archive predates a bridge change; byte-replay
  desyncs) and mislabeled (a0_c469.sta actually loads at h161/x203, not its
  recorded h176 — snapshots can catch Mario below the reach-height). A fresh
  90-min phase-1 run (`artifacts/go_explore_run3`, `--validate` PASSED first,
  4 workers alongside training, 166 winners) replaced them: 4 verified
  chains at level 0, live a1 levels carried over via `levels.json` remap.
  Level-0 frontier cleared at gate rate immediately; 10 advances first hour.
  Then chains 4-8 walled at c433 (h174) ~11h.
- **27v (dense5)**: `densify_stuck` rungs at the c433 choke (c446_d1-3 on the
  chains 4-9 legs, c445_d1-3 for 10/11; level-remap rule: chains whose
  frontier sits BELOW an insertion need +m on their saved level). Chains 5/6
  gated their rungs within hours.
- **27w**: obs 62→74 — barrel TYPE flags (crazy/wild +0x01, blue +0x02) per
  slot; WATCH 47→59; bundled floor tweaks: CORNER_X_RIGHT 160→156 (closed a
  5px no-penalty harbor beside the x=143 ladder), x=99 stub tax 0.08/step,
  novelty bonus excluded in the stub zone (film review had shown ritual stub
  climbs — 8.4%% of bottom-ups ended via the glitch guard — and corner
  camping). CNN transferred (11 layers); RAM MLP/LSTM/heads FRESH.
- **27x**: internal difficulty (0x6380, 1-5) tracked-only: WATCH 59→60,
  difficulty_start/difficulty_end in episode CSVs, NOT in obs. VERIFIED:
  bottom start = 1, curriculum states = 3 (wild-barrel regime lore from the
  user: 3-4 = worst/most erratic wilds, 5 = predictable + static neutral-jump
  counter → the curriculum drills the WORST regime; re-snapshot LATER in game
  time is the lever if a cell's hardness proves difficulty-driven).
  Relearn was slow (clip 0.03-0.08 at lr 5e-5 = timid for fresh heads).
- **27y — FAILED EXPERIMENT (reverted, keep the lesson)**: lr→1e-4 +
  p_curric→0.6 bundled; rehearsal ERODED 0.41→0.35 in 2h, floor unmoved
  (the floor "poverty trap" is reward topology, not gradient volume: fresh
  heads near spawn face penalties in most directions and the only paid
  behavior — x=143 ladder ascent — lies beyond an unrewarded traffic
  crossing). Floor-death gradient at 40%% share interferes with tier skills
  (the 0.7-frontier-share disease, from below). ONE DIAL AT A TIME.
- **27z**: reverted to lr 5e-5 / p_curric 0.8; stabilized but top-band tiers
  stayed FLAT at 31-33% (mature policy: 90-100%).
- **27aa**: single dial ent 0.02→0.01 — entropy fell 1.79→1.55 as designed
  but tiers stayed flat: entropy-trap theory FALSIFIED by its own criterion.
- **27ab — THE ROOT CAUSE + LEVEL RESET (the big one)**: per-height split
  showed the fresh policy at 97-100% on h180-192 while "stuck at 32%" —
  pooling artifact. The real disease: at 27w I preserved the MATURE policy's
  walk-back levels through the obs change; fresh heads inherited an
  IMPOSSIBLE curriculum (50%% of draws grinding 0-5%% mid-tower frontiers,
  rehearsal spread over unclearable inherited tiers). **LEVELS ENCODE THE
  POLICY'S SKILL — after any fresh-heads change, reset levels to 0** (they
  re-earn in minutes at true competence). Reset → 42 advances in the first
  hour, rehearsal records (0.828), zero governor freezes, then honest shelf
  grind + first-ever sustained floor improvement (3.4→7.2).
- **27ac (dense6)**: `densify_stuck` generalized (rung→rung + pruned-
  adjacency legs; desync check vs the successor's ACTUAL loaded height —
  labels lie) → 18 rungs halving six shelf frontiers' gaps; gating within
  30 min. Doom screen found chains 10/11's real blocker: the c445 approach
  is a DEATH ZONE (all candidate rungs 6/6 lethal) — route problem, not
  skill. `bw_chain` CSV column added (per-cell audits were getting
  cross-chain contaminated via height-label lag — cost an hour chasing a
  phantom gate bug; sanity checks: t-origin, volume plausibility).
- **Difficulty-5 re-snapshot probed and REJECTED for L1**: freeze-idle
  (0xF8 + NOOP) does tick 0x6380 (3→4 in ~18s) but the bonus timer kills at
  ~29s from deep cells, still at difficulty 4 (still-hard wilds). Usable
  difficulty 5 does not exist on L1 boards; trivial on L5+ (starts at 5).
- **27ad (dense7)**: `demos/dkong.inp` revealed to be a world-class
  run to 21-6. `scripts/wc_mine.lua` replayed its L1 barrel board, banking
  a state every 45 frames → 162 mined, curated to 70 (running-max descent
  prune — pros milk points, so the raw line wanders; responsiveness + doom
  screens). Slots 0,1,2,10,11 (dead routes) replaced by the WC chain
  (h4→h177, difficulty 1-4 — pro pacing burns the L1 timer, per the user).
  The same .inp holds L5+ barrel boards at genuine difficulty 5, springs,
  pies, rivets — future curriculum source for the whole roadmap.
- **27ae (dense8) — THE 000-TIMER DISCOVERY**: the WC slots' shared frontier
  wc_151 sat at 0.01%% over 14.6k draws (25%% of all curriculum wasted). A
  35s video of the segment (`wc_right_pocket.mp4`) + the user's one viewing
  solved it: the pro's right-corner point-run ends at BONUS 000 — a grace
  window of a few seconds where death lands only while STANDING. Cold starts
  there are speedruns against an expired clock, not skill drills. The whole
  late-route band wc_132..wc_151 carries 0.6-15s of timer (frame proxy:
  000 ≈ frame 7900) — physically unclearable.
- **27af-27ah (dense9-11)**: pruned the time-starved band; probed the
  boundary: 16s (wc_130) 0/4105, 20s (wc_125) 0/3312 → jumped with margin to
  wc_115 (27.6s). Death-time distribution EXONERATED the timer there
  (failures die at 0-13s, only 7/3043 reach the timer window) → at h131 the
  enemy is mid-board barrel traffic. Timer-viability boundary is
  POLICY-SPEED-DEPENDENT — trimmed cells re-splice when the policy speeds
  up. Also: judge the floor by CLEAN bottom-up mean (glitch-guard deaths cap
  at h11 and DRAG the pooled mean down; guard is working, ~25%% attempt rate
  is self-punishing noise). Clean floor crossed 39 (old all-time base: 38).
- **27aj ACTIVE — film-review round 2 fixes**: filmed 16 policy attempts
  from the stuck cells (aviwrite + eval-style venv + `load_state_file`
  between episodes → `stuck_midboard_attempts.mp4`). User diagnosed three
  behavioral bugs in one viewing: (1) ritual-climbing UNTAXED mid-board
  broken stubs (the floor-stub disease, one flight up) → stub tax
  generalized to all 7 BROKEN_STUBS (now a class constant shared by threat
  map + reward); (2) pacing against the top-rung INVISIBLE BARRIER toward
  Kong (measured empirically: hard stop at x=107) → top dead-end tax
  0.08/step at mario_y≤80 & x<133 (5th girder y≥85 stays untaxed);
  (3) mounting the final ladder then dismounting → UPPER_CLIMB_BONUS
  0.30→0.50, IDLE 0.05→0.15, H_HI 192→200 (cover the clear trigger).
  ep_rew not comparable across the change. ALSO: deploy-race lesson — never
  run two kill+launch chains concurrently (two trainers, PIDs 3 apart, 32
  MAMEs; identify duplicates via /proc/PID/fd/1 → log path).
- **27ak (2026-07-09/10)**: WSL hard-reboot #2 killed 27aj at 1.29M (~15 min
  lost — the 500k CheckpointCallback earns its keep). Built + E2E-tested the
  auto-resume infra (`scripts/current_launch.sh` = canonical dials,
  `scripts/auto_resume.sh` = guarded relauncher; §3). Overnight per-cell
  audit produced **THE HOLLOW-TOWER DIAGNOSIS**: all 7 explorer frontiers at
  1.5-3.1% over 800-1150 draws, ALL loading at h163-168 (four chains strung
  on the same c446 rung ladder); the 5 WC slots' shared frontier wc_115 at
  0/1379 — and previously-gated tiers (passed at 0.31) decayed to 0-13%,
  low-and-flat (starved equilibrium: uncapped rehearsal windows give each
  tier ~2% of chain draws at level 20). The cliff is BEHAVIORAL, not
  spatial: two same-height snapshots clear 77% vs 2% — LSTM cold-drop
  establishment, the same mechanism at every wall since run 26.
  Densification of decayed (vs never-passed) cells is the WRONG TOOL; the
  WC chain is un-densifiable past its shelf (the pro's climb from h135
  lives inside the 000-timer endgame — no mintable in-between states with
  viable timers). Also: floor record 48.6 clean; wc_115 timer exonerated.
- **28 (2026-07-10, ACTIVE, TB `RecurrentPPO_51`, commit `7dea1c4`)**: the
  capacity bundle, one restart, five parts: (1) LSTM 512 + RAM MLP 128
  (interference reduction — the actual fix for tier decay); (2) difficulty
  in obs (75); (3) spawn burn-in (§9); (4) rehearsal cap K=8 (§9);
  (5) dense12 = WC slots deduped 5→2 (both copies keep slots 0,1), explorer
  chains 6/7/9 duplicated into slots 2/10/11 (double gate volume). CNN
  transferred (8 layers), heads fresh, LEVELS RESET (the 27ab rule).
  First 35 min: 53 advances at 0.31-1.00 gates, rehearsal 0.844 (record).
  Success criteria: (a) re-descend to and PASS the h163-168 shelf — the
  old policy never did; (b) re-passed tiers HOLD >30%; (c) burn-in cells
  show height-gain >0 on failures; (d) floor recovery slope (fresh heads
  start ~5 in the poverty trap; judge slope, not level).

---

## 12. Critical bugs fixed (do not reintroduce)

### The glitch-guard ratchet: 100% of "honest" floor was stub glitch (fixed run 28d)

The x=99 broken-stub guard killed on 3 CONSECUTIVE off-ladder climb steps —
and reset the streak on any other step. The policy learned climb-pause-climb:
a trajectory audit showed **100% of all bottom-up height gain in 20/20
episodes** coming through the stub (through the broken gap to h62), with 1/20
guard kills. Every floor metric ever celebrated — including the old line's
38-48 "honest band" — was measuring ratchet skill. Found by the user in one
film viewing after the metrics looked historically great.
Fix: the guard accumulates `_glitch_px` per episode (`GLITCH_PX_MAX=6`);
pausing no longer launders the climb. Lessons: (1) a guard the policy can
pace around is not a guard; (2) metric filters that trust a beatable guard
lie confidently; (3) when a long-stuck metric suddenly breaks out, audit the
mechanism before celebrating — `climb_audit.py`-style x-zone attribution of
every gained pixel is cheap.

### Per-episode state set during curriculum loads gets zeroed (hit TWICE: 28c, 28e)

`reset()` calls `_begin_episode(state)` late, and `_begin_episode` zeroes all
per-episode fields (burn-in counters, forced-action queues). Anything decided
during `_load_backward_start` must be STASHED (`_pending_approach` pattern)
and applied after `_begin_episode`, guarded on `start_type=="curriculum"` so
a mid-reset `_recover()` fallback can't inherit it. The burn-in draw (28c)
and the approach queue (28e) both hit this exact trap.

### Old checkpoints don't load after capacity changes (fixed run 28, `7dea1c4`)

`PPO.load()` / `RecurrentPPO.load()` rebuilds the policy skeleton from the
CURRENT code (policy_kwargs reference `DkFeaturesExtractor` by module path, and
its layer widths come from today's constants) then `set_parameters(exact_match=
True)`. After RAM_HIDDEN 64→128, run-27 checkpoints stopped fitting their own
skeleton — `load()` raises `RuntimeError` on state_dict shapes. Consequences:
- `--transfer-features-from` now reads the checkpoint zip's raw state dict via
  `stable_baselines3.common.save_util.load_from_zip_file` (no skeleton built)
  and copies shape-matching `features_extractor.*` tensors only.
- Evaluating a pre-bundle checkpoint requires checking out a pre-`7dea1c4`
  commit (constants must match the weights).
- `auto_resume.sh` must glob ONLY current-line models (`run28*`) — a stale
  glob would feed `--init-from` an unloadable old checkpoint and crashloop.

### The x=99 broken-ladder glitch: superhuman exploit in winners AND policy (guarded run 27o, 2026-07-05)

**The find** (spotted by the user watching footage): Mario climbing a broken
ladder stub with frame-perfect up/down inputs — a TAS-grade exploit no human
performs. Census: 20/20 bottom-start episodes rode the x=99 stub (710
glitch-climb steps). Worse, winner-route arithmetic (chain 0: h=43@x=93 ->
h=80@x=180 in ~20 macro-steps, impossible via the legit x=53/x=143 ladders)
shows the go-explore winners ALSO used it — the curriculum's lower legs
encode the exploit, and height_mean_bottomup's rise (31->49) was partly
glitch-driven.
**Guard** (`35c3399`): in `_reward`, 3+ consecutive climb steps (y falling,
x pinned, alive, no jump arc, screen 1) outside any `COMPLETE_LADDERS`
envelope ends the episode with the death penalty. Verified: kills the
exploit in ~22 steps, zero false positives on legit ladder climbs.
`glitch_kill` is exported per episode (info + CSVs).
**Consequences**: honest bottom-up baselines reset (expect height_mean to
CRASH first, then recover via legit routes); the walk-back below h~80 must
find real connections (x=53 ladder) the winners never demonstrated — if it
stalls flat there, phase 1 (go_explore) must be re-run with the guard
active so winners are legit. The guard lives in env.step()/_reward, so any
phase-1 re-run through env.step inherits it automatically.

### Spawn ate the CLI env params → phantom bottom-up clears (fixed run 27i, 2026-07-04)

**The bug**: `main()` applied `--p-curric`/`--p-no-barrels` by mutating
`DonkeyKongEnv` CLASS attributes, but `SubprocVecEnv(start_method="spawn")`
workers re-import the module — all 16 envs silently reverted to the defaults
(0.15 curriculum, 0.15 barrel-free). Every 27-series run before 27i trained on
the wrong episode mix, and the barrel-free bottom climbs (trivial without
hazards) were counted by `ClimbMetricsCallback` into `clear_rate_bottomup`:
the "honest metric" rose 0.04→0.14 in 27g while 425 controlled live-barrel
bottom starts across three eval modes produced 0 clears. No aggregate log
line could reveal this; the per-episode CSVs exposed it in minutes — every
phantom row read `start_y=240, end_screen=4, no_barrels=True`.
**Fix** (`da6b2dc`): the values ride into workers as `make_env` parameters and
are set as INSTANCE attributes inside the thunk (which executes in the worker);
`clear_rate_bottomup`/`height_mean_bottomup` additionally exclude `no_barrels`
episodes. **Rule**: config that must reach a spawn worker travels in the
pickled thunk (instance state), never via launcher-side class/global mutation.
Verify after any env-config change: the curriculum fraction observed in
`logs/episodes/dk_*.monitor.csv` must match the CLI within a few percent.

### Jitter-death at curriculum cells: the walk-back stall (fixed run 27j, 2026-07-04)

**The bug**: the post-load RNG jitter (`_hold(A_NOOP, 0-20)`) idled Mario up
to 1.3s at the restored cell. Top-girder cells (y=72-80, heights ~160-164)
sit in the barrel-spawn lane; winners passed through IN MOTION, and an idling
Mario dies DURING reset. The game burns a stored life (the .sta carries
lives) and respawns him at the bottom — a ghost episode labeled curriculum:
`max_height` frozen at the start height (the dead-load position sets
`_min_y`), ~98-step median (one bottom life), ~1% frontier clears. Cells
above barrel reach (y<=64) were immune — which is exactly why 6 chains
promoted at 0.81-1.00 while 6 stalled at 0 for the whole of 27g-27i.
Compounding it: `_is_responsive` only checked x/y change, and a death tumble
moves without input, so dying cells passed the probe. And RAM 0x6200
("is_dead") is INVERTED — 1=alive, 0=dead — so naive flag checks read
corpses as alive; death detection must use the lives drop (it does).
**Fix** (`4740dde`): jitter applies to bottom starts only (idling at the
spawn is safe for ~5s; curriculum diversity comes from action sampling + 12
chains); `_is_responsive` fails any probe step with the alive flag down;
0x6200 polarity documented in memory_map.py.
**Diagnosis chain worth remembering**: per-cell clear table from the episode
CSVs (bimodal 0-3% vs 85-100% split at one girder) -> per-step trace (Mario
frozen, inputs ignored) -> -aviwrite frames (sprite absent, game running) ->
input-drive probe (corpse tumble follows nothing; respawn at x=60,y=240).

### Multi-life episodes drowned the backward curriculum (fixed run 27d, 2026-07-04)

**The bug**: `done = (died and s["lives"] == 0)` — episodes packed all 3 lives.
A relic of 19s intro resets (fewer intros per env-step). With 0.03s save-state
resets it was pure harm, and for backward-curriculum starts it was fatal: an
episode starting near Pauline that died once RESPAWNED AT THE BOTTOM and continued
as a mislabeled, unclearable bottom-up run. The frontier tier showed ~0% clears
for 18M steps (runs 27/27b/27c) not because the states were unlearnable but
because one mistake converted the whole episode into noise. Also: every metric in
runs 1-27c was best-over-3-lives, not per-life.
**Fix**: `done = died or cleared or timed_out` (single-life). Telemetry signature
that exposed it: frozen ~50-step death animation → mario_y=0 sentinel → Mario at
height 9 with the episode still running.

### Slot-clobber: curriculum swaps corrupted "bottom" resets (fixed run 27b, 2026-07-03)

**The bug**: loading a curriculum state = copy .sta onto slot `dk_<port>.sta` +
A_LOAD; but a bottom start = "load the slot" — so after one curriculum episode,
every "bottomup" episode silently started near the top (`clear_rate_bottomup`
read 0.74!). **Fix**: `load_state_file()` is THE primitive for loading any .sta
through the slot — it restores the `bottom_<port>.sta` backup right after the
load consumes the swap. Never copy onto `dk_<port>.sta` any other way.

### Frozen curriculum snapshots (fixed 2026-07-04)

**The bug**: 13/65 exported winner-chain states (20%) were snapshotted during
cutscene/transition freezes; they always fail `_is_responsive` and silently fall
back to bottom starts — wasted curriculum draws. **Fix**: `export_chains
--verify-states` loads every state in a scratch MAME and drops unresponsive ones.
Always export with this flag.

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

Phase 2 (backward walk-back) is run 27ad, active. In priority order:

1. **FILM IT FIRST.** The single highest-value diagnostic is now proven
   twice: record the policy failing from the stuck state (aviwrite +
   eval-style venv + `load_state_file` between episodes) and have the user
   watch it — one viewing has repeatedly beaten thousands of rollouts of
   inference (000-timer wall; stub/barrier/final-ladder trio). Before any
   dial or curriculum surgery on a stuck cell, make the 3-minute film.
2. **One dial at a time; one deploy chain at a time (foreground).** Judge by
   outcomes (per-cell CSV rates via `bw_chain`), not proxies — 27y/27aa
   failed on proxy-led reasoning. Rungs via `densify_stuck` remain the
   proven unstick lever for explorer cells; finer re-mining for WC cells.
   Check remaining BONUS TIMER before trusting any pro-play state
   (frame proxy; doom screens cannot see timer doom).
3. **Walk-back to the floor is the plan of record**: the floor policy sits
   in a reward-topology poverty trap (bottom-ups ~h5-7, but drifting up for
   the first time ever) and is EXPECTED to lag until chains descend to floor
   cells. Judge floor progress by chain levels, not height_mean_bottomup.
4. **After any fresh-heads change: RESET LEVELS TO 0** (delete
   `<backward-dir>/levels.json`). Levels encode the POLICY's skill, not the
   project's progress — 27w-27ab lost ~30 hours to inherited levels.
5. **THE NEXT-OBS-CHANGE BUNDLE** (do together, once, at the transition to
   the "master toward 90%" phase after first honest clears): (a) difficulty/5
   into the RAM obs (regime-conditional play is required for the 90% target);
   (b) capacity bump LSTM 256→512 + RAM MLP 64→128 (interference reduction —
   the recurring tier-decay disease is small-net weight sharing); (c) any
   validated floor-reward surgery. Expect ~10-30%% fps cost, a fresh-heads
   relearn, and a LEVEL RESET (rule 3) — plan it, don't stumble into it.
6. **Difficulty-varied curriculum**: the training curriculum is a
   difficulty-3 monoculture (all curric draws inherit diff 3) and L1 cannot
   produce usable difficulty-5 states (bonus timer, probed). When regime
   training matters, mine the WC .inp's L5+ barrel boards
   (`wc_mine.lua`, level gate) — genuine difficulty-5 states from
   world-class play.
7. **Restart hygiene**: capture the trainer PID as a NUMBER first
   (`pgrep -a -f dkong_ai.train | grep -v "bash -c"`), `kill -TERM <pid>`,
   wait on `kill -0`, verify `pgrep -c -x mame` hits 0 before relaunching —
   pgrep/pkill patterns match your own shell's command line (bit us TWICE:
   a stuck wait-gate and a surviving trainer → two trainers writing the same
   checkpoint/levels paths for 4 minutes). Levels resume from `levels.json`;
   delete it to reset the walk-back. When REPLACING chains, rewrite
   `levels.json` slot-for-slot (count mismatch silently resets ALL levels).
   When INSERTING cells, chains whose frontier is BELOW the insertion point
   need their saved level bumped by the insert count.
8. **a0-class archives**: before ever minting rungs from an archive, replay
   one known leg and compare landing height (`densify_stuck` does this
   per-leg and skips desynced legs). Snapshot heights in manifests can be
   ~15px below the recorded reach-height (save lands a few frames later —
   Mario may be falling); trust the loaded state, not the label.

**Targets (from the user — a world-class DK player; treat their DK claims as
ground truth):** ≥90%% barrel-board completion is the bar ("solved" ≠ 50%%);
endgame = kill screen (L22 board 1; screen 117 last completable). Reliability
budget: post-L4 barrel + springboard boards ≈ 100%% (difficulty-5 regimes are
deterministic); pie/rivet boards have genuine no-win scenarios — never gate
or grind toward 100%% there. Note: the L1 barrel board we train (difficulty
1→3, random wilds) may cap BELOW post-L4 barrel boards. Future infra for the
kill-screen goal: multi-board episodes (env currently terminates on clear).

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
- **Spawn workers don't see launcher-side class/global mutations** — full
  writeup in §12. Applies to ANY "set it globally, then build SubprocVecEnv"
  pattern, not just the two params it bit us on.
- **First reset skips the RNG jitter**: a fresh env's first episode starts from
  the exact post-intro state; only episode 2+ resets add the 0–20 exchange
  jitter. One-episode-per-env eval loops therefore hammer ONE fixed barrel
  seed and badly misestimate clear rates (0/60 at a nominal 8% looked like a
  policy bug). Measure with a persistent env across many episodes, or inject
  recorded NOOP steps at episode start in record mode.
- **Training-faithful video exists**: `.inp` recording can't survive state-load
  resets, but MAME `-aviwrite` captures the rendered screen and doesn't care —
  `DonkeyKongEnv(..., record=False, extra_mame_args=["-snapshot_directory",
  dir, "-aviwrite", name])` films snapshot-start episodes (works headless with
  `-video none`; raw AVI ≈ 10 MB/s of game time, convert with ffmpeg).

---

## 17. File map

`dkong_ai/`:
- `mame_env.py` — env: MAME launch, socket bridge, obs build, reward, curricula.
  `_info()` returns `{"state", "max_height", "cleared", "start_type", "bw_start"}`.
  Key primitives: `load_state_file()` (the ONLY sanctioned slot-swap loader),
  `set_backward_level()`, ctor params `backward_manifest`, `extra_mame_args`.
- `memory_map.py` — all confirmed RAM addresses + score decode. 47 WATCH_ORDER entries.
- `dk_policy.py` — `DkFeaturesExtractor` (CNN+RAM MLP) + `DkFrameStackWrapper`.
- `go_explore.py` — **phase 1**: policy-free exploration archive (cells = save-states
  + byte trajectories), workers, `--validate` determinism self-test, winner
  verification. Ports 5200+. TensorBoard `GoExplore_N`.
- `export_chains.py` — archives → `artifacts/backward/` manifest for phase 2.
- `densify_stuck.py` — surgical walk-back rungs: replays a stuck frontier
  cell's successor legs from the archive true-parent (prune-descents makes
  manifest-adjacent ≠ archive parent-child), clean-frame filter, 6-trial
  doom screen, splices a new backward dir with `levels.json` kept valid
  (levels are end-relative; frontier pointers land on the easiest new rung).
- `scripts/wc_mine.lua` — mine route states from .inp playback: on a chosen
  board (screen/level gate), `mac:save` every N frames while alive, logging
  frame/x/y/height/difficulty for curation. Used on the world-class
  `demos/dkong.inp`; re-gate to level==5 for difficulty-5 barrel states.
  ALWAYS use `--verify-states` (drops frozen snapshots).
- `replay_winner.py` — render a winner chain to video (`--avi x.avi`, auto-mp4
  via ffmpeg) or watch live (`--watch`). Port 5300.
- `tb_bridge.py` — one-off: backfill pre-native go-explore logs into TensorBoard.
- `train.py` — RecurrentPPO training. Callbacks log `climb/height_mean_bottomup`,
  `climb/height_mean_curric`, `climb/clear_rate_bottomup`, and (backward mode)
  `climb/backward_level`, `climb/backward_clear_rate`, `climb/backward_clear_frontier`.
  Flags: `--n-envs`, `--stack`, `--gamma`, `--ent-coef`, `--init-from`,
  `--p-no-barrels`, `--p-curric`, `--save`, `--timesteps`, `--lr`, `--n-epochs`,
  `--lstm`, `--lstm-hidden`, `--transfer-features-from`, `--backward-dir`,
  `--bw-window`, `--bw-threshold`.
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
