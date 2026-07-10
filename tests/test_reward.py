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
    p = _state(mario_y=220, mario_x=143)          # height = 240-220 = 20
    s = _state(mario_y=210, mario_x=143)           # height = 30 > max_h(20), is_jumping=0
    env._prev = p
    r, done = env._reward(s)
    # milestone: (30 - 20) * 0.5 = 5.0
    assert r > 0, f"expected positive reward (milestone), got {r}"
    assert env._reward_max_h == 30, f"_reward_max_h should update to 30, got {env._reward_max_h}"


def test_milestone_blocked_during_jump():
    """New max height while jumping → NO milestone reward, max_h not updated."""
    env = _make_env(reward_max_h=20)
    p = _state(mario_y=220, mario_x=143)
    s = _state(mario_y=210, mario_x=143, is_jumping=1)  # height 30 > max_h but jumping
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
    p = _state(mario_y=220, mario_x=143)           # height 20, y=220
    s = _state(mario_y=215, mario_x=143)           # height 25, y=215 < p.y → ascending
    env._prev = p
    r, done = env._reward(s)
    # Should include FIRST_CLIMB_BONUS (0.30) + milestone for new height + novelty + per-step
    assert r >= DonkeyKongEnv.FIRST_CLIMB_BONUS, (
        f"expected at least FIRST_CLIMB_BONUS={DonkeyKongEnv.FIRST_CLIMB_BONUS}, got {r}"
    )


def test_first_climb_bonus_blocked_during_jump():
    """Ascending first ladder during jump arc → climb bonus suppressed."""
    env = _make_env(reward_max_h=0)
    p = _state(mario_y=220, mario_x=143)
    s = _state(mario_y=215, mario_x=143, is_jumping=1)
    env._prev = p
    r_jump, _ = env._reward(s)

    # Compare against same scenario with is_jumping=0
    env2 = _make_env(reward_max_h=0)
    env2._prev = p
    r_walk, _ = env2._reward(_state(mario_y=215, mario_x=143, is_jumping=0))

    assert r_jump < r_walk, (
        f"jump reward {r_jump:.3f} should be < walk reward {r_walk:.3f} "
        "(climb bonus must be suppressed during jump)"
    )


# ---- ladder idle cost tests --------------------------------------------------

def test_first_ladder_idle_cost():
    """Stationary at first ladder (y unchanged) while not jumping → idle penalty."""
    env = _make_env(reward_max_h=25)    # height already paid, so no milestone
    p = _state(mario_y=215, mario_x=143)           # height 25
    s = _state(mario_y=215, mario_x=143)           # y unchanged → idle
    # Pre-visit the cell so the novelty bonus doesn't offset the idle cost.
    height = BASE_Y - 215  # 25
    env._visited.add((143 // DonkeyKongEnv.CELL, height // DonkeyKongEnv.CELL))
    env._prev = p
    r, done = env._reward(s)
    # Without novelty: per-step height bonus (~0.00075) << idle cost (-0.05).
    assert r < 0, f"expected negative reward for ladder idle (cell pre-visited), got {r}"


def test_idle_cost_blocked_during_jump():
    """Idle cost does NOT fire when is_jumping=1 (can't idle on ladder mid-air)."""
    env = _make_env(reward_max_h=25)
    p = _state(mario_y=215, mario_x=143)
    s = _state(mario_y=215, mario_x=143, is_jumping=1)
    env._prev = p
    r_jump, _ = env._reward(s)

    env2 = _make_env(reward_max_h=25)
    env2._prev = p
    r_idle, _ = env2._reward(_state(mario_y=215, mario_x=143, is_jumping=0))

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
