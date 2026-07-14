"""Unit tests for DonkeyKongEnv._reward().

Creates a minimal stub (no MAME, no socket, no file I/O) by bypassing __init__
and setting only the instance attributes _reward() actually reads. All class-level
constants (BASE_Y, FIRST_CLIMB_X_LO, etc.) are inherited automatically.

Run with: python -m pytest tests/test_reward.py
"""
from dkong_ai.mame_env import DonkeyKongEnv

BASE_Y = DonkeyKongEnv.BASE_Y  # 240


def _make_env(reward_max_h=0):
    """Minimal stub sufficient for _reward()."""
    env = object.__new__(DonkeyKongEnv)
    env._reward_max_h = float(reward_max_h)
    env._wp_hit = set()
    env._visited = set()
    env._episode_steps = 0
    env._corridor = None     # disables novelty corridor bonus (returns None in _target_x)
    env._glitch_px = 0       # glitch-guard accumulators the guard reads/writes
    env._glitch_kill = False
    return env


def _state(mario_y, mario_x, is_jumping=0, lives=3, screen_id=1, score=None,
           has_hammer=0):
    return {
        "mario_y": mario_y,
        "mario_x": mario_x,
        "is_jumping": is_jumping,
        "lives": lives,
        "screen_id": screen_id,
        "score": score,
        "has_hammer": has_hammer,
    }


# ---- height milestone tests --------------------------------------------------

def test_milestone_fires_when_grounded():
    """New max height while not jumping → milestone reward."""
    env = _make_env(reward_max_h=20)
    p = _state(mario_y=220, mario_x=203)          # height = 240-220 = 20
    s = _state(mario_y=210, mario_x=203)           # height = 30 > max_h(20), is_jumping=0
    env._prev = p
    r, done = env._reward(s)
    # milestone: (30 - 20) * 0.5 = 5.0
    assert r > 0, f"expected positive reward (milestone), got {r}"
    assert env._reward_max_h == 30, f"_reward_max_h should update to 30, got {env._reward_max_h}"


def test_milestone_blocked_during_jump():
    """New max height while jumping → NO milestone reward, max_h not updated."""
    env = _make_env(reward_max_h=20)
    p = _state(mario_y=220, mario_x=203)
    s = _state(mario_y=210, mario_x=203, is_jumping=1)  # height 30 > max_h but jumping
    env._prev = p
    r, done = env._reward(s)
    assert env._reward_max_h == 20, f"_reward_max_h must NOT update during jump, got {env._reward_max_h}"


def test_milestone_not_paid_for_existing_max():
    """Same height as current max → no milestone payment."""
    env = _make_env(reward_max_h=30)
    p = _state(mario_y=210, mario_x=100)
    s = _state(mario_y=210, mario_x=100)           # height 30 == max_h
    env._prev = p
    r, done = env._reward(s)
    assert env._reward_max_h == 30


# ---- first-ladder climb bonus tests -----------------------------------------

def test_first_climb_bonus_fires_when_grounded():
    """Ascending first ladder while not jumping → +FIRST_CLIMB_BONUS."""
    env = _make_env(reward_max_h=0)
    # height = 25 in FIRST_CLIMB_H range (10-44); x=143 in FIRST_CLIMB_X range (133-155)
    p = _state(mario_y=220, mario_x=203)           # height 20, y=220
    s = _state(mario_y=215, mario_x=203)           # height 25, y=215 < p.y → ascending
    env._prev = p
    r, done = env._reward(s)
    # Should include FIRST_CLIMB_BONUS (0.30) + milestone for new height + novelty + per-step
    assert r >= DonkeyKongEnv.FIRST_CLIMB_BONUS, (
        f"expected at least FIRST_CLIMB_BONUS={DonkeyKongEnv.FIRST_CLIMB_BONUS}, got {r}"
    )


def test_first_climb_bonus_blocked_during_jump():
    """Ascending first ladder during jump arc → climb bonus suppressed."""
    env = _make_env(reward_max_h=0)
    p = _state(mario_y=220, mario_x=203)
    s = _state(mario_y=215, mario_x=203, is_jumping=1)
    env._prev = p
    r_jump, _ = env._reward(s)

    # Compare against same scenario with is_jumping=0
    env2 = _make_env(reward_max_h=0)
    env2._prev = p
    r_walk, _ = env2._reward(_state(mario_y=215, mario_x=203, is_jumping=0))

    assert r_jump < r_walk, (
        f"jump reward {r_jump:.3f} should be < walk reward {r_walk:.3f} "
        "(climb bonus must be suppressed during jump)"
    )


# ---- ladder idle cost tests --------------------------------------------------

def test_first_ladder_idle_cost():
    """Stationary at first ladder (y unchanged) while not jumping → idle penalty."""
    env = _make_env(reward_max_h=25)    # height already paid, so no milestone
    p = _state(mario_y=215, mario_x=203)           # height 25
    s = _state(mario_y=215, mario_x=203)           # y unchanged → idle
    # Pre-visit the cell so the novelty bonus doesn't offset the idle cost.
    height = BASE_Y - 215  # 25
    env._visited.add((203 // DonkeyKongEnv.CELL, height // DonkeyKongEnv.CELL))
    env._prev = p
    r, done = env._reward(s)
    # Without novelty: per-step height bonus (~0.00075) << idle cost (-0.05).
    assert r < 0, f"expected negative reward for ladder idle (cell pre-visited), got {r}"


def test_idle_cost_blocked_during_jump():
    """Idle cost does NOT fire when is_jumping=1 (can't idle on ladder mid-air)."""
    env = _make_env(reward_max_h=25)
    p = _state(mario_y=215, mario_x=203)
    s = _state(mario_y=215, mario_x=203, is_jumping=1)
    env._prev = p
    r_jump, _ = env._reward(s)

    env2 = _make_env(reward_max_h=25)
    env2._prev = p
    r_idle, _ = env2._reward(_state(mario_y=215, mario_x=203, is_jumping=0))

    assert r_jump > r_idle, (
        f"jump r={r_jump:.3f} should be > idle r={r_idle:.3f} (idle cost blocked during jump)"
    )


# ---- termination tests -------------------------------------------------------

def test_death_terminates():
    """Last life lost (lives → 0) → done=True and negative reward."""
    env = _make_env()
    p = _state(mario_y=215, mario_x=100, lives=1)
    s = _state(mario_y=215, mario_x=100, lives=0)  # game over
    env._prev = p
    r, done = env._reward(s)
    assert done
    assert r <= -10.0


def test_clear_terminates():
    """screen_id increment → done=True and large positive reward."""
    env = _make_env()
    p = _state(mario_y=60, mario_x=100, screen_id=1)
    s = _state(mario_y=60, mario_x=100, screen_id=2)
    env._prev = p
    r, done = env._reward(s)
    assert done
    assert r >= 100.0


def test_hammer_wall_penalty_fires_when_score_is_none():
    """The hammer-at-left-wall penalty must not depend on score decode
    validity: score legitimately reads None on volatile HUD frames, and an
    indentation accident once made the penalty silently skip exactly then
    (external review finding, 2026-07-10). Differential: identical states
    except has_hammer, both with score=None."""
    def reward_with(hammer):
        env = _make_env(reward_max_h=60)          # above height 50: no milestone
        p = _state(mario_y=190, mario_x=40, has_hammer=hammer)  # h=50, left wall
        s = _state(mario_y=190, mario_x=40, has_hammer=hammer)
        assert s["score"] is None
        env._prev = p
        r, _ = env._reward(s)
        return r

    diff = reward_with(1) - reward_with(0)
    assert abs(diff + DonkeyKongEnv.HAMMER_WALL_COST) < 1e-9, (
        f"hammer-wall penalty missing/wrong when score=None: diff={diff}")


def test_glitch_guard_catches_ratchet():
    """Climb-pause-climb on the broken stub must still trigger the guard.
    The old 3-consecutive-streak guard reset on every pause; a trajectory
    audit (2026-07-11) showed 20/20 bottom-up episodes ratcheting 100% of
    their height through the x=99 stub with only one guard kill. The guard
    now accumulates pixels per episode."""
    env = _make_env()
    env._glitch_px = 0
    env._glitch_kill = False
    y = 220
    killed = False
    for pull in range(4):                      # 4x (climb 2px, pause)
        for _ in range(2):                     # 2 climb steps of 1px
            p = _state(mario_y=y, mario_x=99)
            p["is_dead"] = 1
            s = _state(mario_y=y - 1, mario_x=99)
            s["is_dead"] = 1
            env._prev = p
            r, done = env._reward(s)
            y -= 1
            if done and env._glitch_kill:
                killed = True
        # pause step: y unchanged (streak-reset bait for the old guard)
        p = _state(mario_y=y, mario_x=99)
        p["is_dead"] = 1
        s = _state(mario_y=y, mario_x=99)
        s["is_dead"] = 1
        env._prev = p
        env._reward(s)
    assert killed, (
        f"ratchet not caught: {env._glitch_px}px accumulated, no kill")


# ---- potential-based floor shaping (run 29) ----------------------------

def _pbrs_env():
    env = _make_env(reward_max_h=200)   # suppress milestone noise
    env._glitch_px = 0
    env._wp_hit = set(range(16)) | set(range(100, 116))  # waypoints + girders fired
    env._visited = {(x, h) for x in range(16) for h in range(16)}  # no novelty
    return env


def test_pbrs_pays_crossing_progress():
    """Moving toward the x=143 ladder on the floor pays ~coef per pixel."""
    env = _pbrs_env()
    p = _state(mario_y=239, mario_x=100)
    s = _state(mario_y=239, mario_x=110)   # 10px closer to 143
    env._prev = p
    r, _ = env._reward(s)
    assert r > 0.3, f"crossing progress should pay, got {r}"


def test_pbrs_not_farmable_by_looping():
    """There-and-back nets ~zero (telescoping): no oscillation income."""
    env = _pbrs_env()
    total = 0.0
    for px, sx in ((100, 110), (110, 100)):
        p = _state(mario_y=239, mario_x=px)
        s = _state(mario_y=239, mario_x=sx)
        env._prev = p
        env._visited = {(sx // env.CELL, 0), (px // env.CELL, 0)}
        r, _ = env._reward(s)
        total += r
    assert abs(total) < 0.05, f"loop should net ~0, got {total}"


def test_pbrs_saturates_off_floor():
    """Above the floor band, x-position no longer moves the potential."""
    env = _pbrs_env()
    p = _state(mario_y=180, mario_x=60)    # height 60: saturated
    s = _state(mario_y=180, mario_x=100)
    env._prev = p
    r, _ = env._reward(s)
    assert abs(r) < 0.05, f"off-floor x-moves should not pay PBRS, got {r}"


def test_pbrs_stage2_pays_toward_x53():
    """On girder 2 (h25-44), moving toward the x53 ladder pays; the old
    behaviour (saturated = inert) left a reward desert where Mario loitered
    under the girder-3 edge instead of walking to the ladder."""
    env = _pbrs_env()
    p = _state(mario_y=205, mario_x=150)   # height 35, mid girder 2
    s = _state(mario_y=205, mario_x=140)   # 10px closer to x53
    env._prev = p
    r, _ = env._reward(s)
    assert r > 0.3, f"girder-2 progress toward x53 should pay, got {r}"


def test_pbrs_stage2_not_farmable_and_saturates():
    """Stage-2 loop nets ~0; above the ladder base x-moves are inert."""
    env = _pbrs_env()
    total = 0.0
    for px, sx in ((150, 140), (140, 150)):
        p = _state(mario_y=205, mario_x=px)
        s = _state(mario_y=205, mario_x=sx)
        env._prev = p
        env._visited = {(sx // env.CELL, 2), (px // env.CELL, 2)}
        r, _ = env._reward(s)
        total += r
    assert abs(total) < 0.05, f"stage-2 loop should net ~0, got {total}"
    env._visited = {(x, h) for x in range(16) for h in range(16)}  # no novelty
    p = _state(mario_y=190, mario_x=60)    # height 50: above ladder base
    s = _state(mario_y=190, mario_x=100)
    env._prev = p
    r, _ = env._reward(s)
    assert abs(r) < 0.05, f"above h44 x-moves should not pay PBRS, got {r}"


def test_climb_bonus_gap_gated():
    """Climbing the x53 ladder pays only when the column above is clear —
    climbing under a barrel is the low-percentage gamble (user doctrine)."""
    env = _pbrs_env()
    p = _state(mario_y=190, mario_x=53)    # on the ladder, h50
    s = _state(mario_y=186, mario_x=53)    # climbed 4px
    env._prev = p
    r_clear, _ = env._reward(s)
    env._prev = p
    s2 = dict(s, barrel0_st=1, barrel0_x=55, barrel0_y=170)  # barrel ABOVE
    r_blocked, _ = env._reward(s2)
    assert r_clear - r_blocked > 0.2, (
        f"clear-column climb should outpay blocked climb: "
        f"{r_clear} vs {r_blocked}")
    env._prev = p
    s3 = dict(s, barrel0_st=1, barrel0_x=55, barrel0_y=210)  # barrel BELOW
    r_below, _ = env._reward(s3)
    assert abs(r_below - r_clear) < 0.05, (
        f"a barrel BELOW must not gate the climb: {r_below} vs {r_clear}")
