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
    env._clean_mount_paid = True   # suppress the one-shots in generic tests
    env._waterfall_paid = True
    env._x131_mount_paid = True
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
    # height = 25 in FIRST_CLIMB_H range (2-30); x=203 in FIRST_CLIMB_X range (196-210)
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
    """Above the ladder base, x-position no longer moves the potential.
    Probes h90 — between the g3 and g5 traverse bands, where no
    directional reward applies either."""
    env = _pbrs_env()
    p = _state(mario_y=150, mario_x=60)    # height 90: saturated
    s = _state(mario_y=150, mario_x=100)
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


def test_time_race_gate_and_mount_bonus():
    """User pro line: a girder-3 barrel matters only if it can reach the
    ladder top before Mario does; tower barrels never block. First clean
    climb step also pays the one-shot mount bonus."""
    env = _pbrs_env()
    p = _state(mario_y=190, mario_x=53)
    base = _state(mario_y=186, mario_x=53)
    env._prev = p
    r_clear, _ = env._reward(base)
    # far-up-tower barrel (y=100): must NOT gate
    env._prev = p
    r_tower, _ = env._reward(dict(base, barrel0_st=1, barrel0_x=55, barrel0_y=100))
    assert abs(r_tower - r_clear) < 0.05
    # girder-3 barrel beyond Mario's remaining reach: safe
    env._prev = p
    r_far, _ = env._reward(dict(base, barrel0_st=1, barrel0_x=140, barrel0_y=150))
    assert abs(r_far - r_clear) < 0.05
    # same barrel close enough to win the race: gated
    env._prev = p
    r_near, _ = env._reward(dict(base, barrel0_st=1, barrel0_x=70, barrel0_y=150))
    assert r_clear - r_near > 0.2
    # one-shot mount bonus fires exactly once
    env2 = _pbrs_env()
    env2._clean_mount_paid = False
    env2._prev = p
    r1, _ = env2._reward(dict(base))
    env2._prev = base
    r2, _ = env2._reward(_state(mario_y=182, mario_x=53))
    assert r1 - r2 > 1.5, f"first climb step should carry the one-shot: {r1} vs {r2}"


def test_waterfall_pass_one_shot():
    """Standing on girder 3 past the sweep (from a low start) pays the
    one-shot exactly once — reachable trigger (the ladder tops at h62)."""
    env = _pbrs_env()
    env._waterfall_paid = False
    env._start_h = 38                      # wait-spot start
    p = _state(mario_y=178, mario_x=60)    # h62: ladder top
    s = _state(mario_y=176, mario_x=64)    # h64 on girder 3
    env._prev = p
    r1, _ = env._reward(s)
    env._prev = s
    r2, _ = env._reward(_state(mario_y=175, mario_x=68))
    assert r1 - r2 > 4.0, f"first pass should carry the one-shot: {r1} vs {r2}"
    env2 = _pbrs_env()
    env2._waterfall_paid = False
    env2._start_h = 164                    # tower start: must NOT fire
    env2._prev = p
    r3, _ = env2._reward(dict(s))
    assert r3 < 4.0, f"tower starts must not earn the pass bonus: {r3}"


def test_g3_traverse_pays_right_and_no_pump():
    """Rightward movement on girder 3 pays; the trimmed girder-2 band no
    longer pays leftward there (the two would have formed an oscillation
    pump in the old h36-65 overlap)."""
    env = _pbrs_env()
    p = _state(mario_y=176, mario_x=90)    # h64, girder 3
    s = _state(mario_y=176, mario_x=100)   # 10px right
    env._prev = p
    r_right, _ = env._reward(s)
    assert r_right > 0.3, f"g3 rightward should pay, got {r_right}"
    env2 = _pbrs_env()
    env2._prev = _state(mario_y=176, mario_x=100)
    r_left, _ = env2._reward(_state(mario_y=176, mario_x=90))
    assert r_left < 0.1, f"g3 leftward must not pay (pump check), got {r_left}"


def test_x131_doctrine_gated():
    """The middle ladder gets the full doctrine: clean-gap climbs pay,
    gambles do not (generalized time-race, 2026-07-16)."""
    env = _pbrs_env()
    p = _state(mario_y=150, mario_x=131)   # h90, on the x131 ladder
    s = _state(mario_y=146, mario_x=131)
    env._prev = p
    r_clear, _ = env._reward(s)
    env._prev = p
    r_block, _ = env._reward(dict(s, barrel0_st=1, barrel0_x=140, barrel0_y=110))
    assert r_clear - r_block > 0.3, f"{r_clear} vs {r_block}"


def test_hammer_rush_tax_range_gated():
    """Advancing on an in-range threat with the hammer UP costs; the same
    approach from outside smashing range is free (user refinement)."""
    def run(threat_dx, hammer_up):
        env = _pbrs_env()
        p = _state(mario_y=175, mario_x=100, has_hammer=1)
        s = _state(mario_y=175, mario_x=104, has_hammer=1)  # moving right
        s["hammer_y"] = 175 - (16 if hammer_up else 2)
        s["fireball0_st"] = 1
        s["fireball0_x"] = 104 + threat_dx
        s["fireball0_y"] = 175
        env._prev = p
        r, _ = env._reward(s)
        return r
    in_range_up = run(10, True)
    far_up = run(40, True)
    in_range_down = run(10, False)
    assert far_up - in_range_up > 0.03, f"{far_up} vs {in_range_up}"
    assert in_range_down - in_range_up > 0.03, f"{in_range_down} vs {in_range_up}"


def test_guard_execution_never_cheapest_death():
    """A guard execution at the floor must cost MORE than an honest floor
    death (-10 -5), or suicide-at-the-stub becomes the optimal exit from
    the poverty trap (30l post-mortem: wave rose 33->93% at low clip)."""
    env = _pbrs_env()
    p = _state(mario_x=99, mario_y=236)
    s = _state(mario_x=99, mario_y=228)  # 8px x-pinned ascend off-envelope
    s["is_dead"] = p["is_dead"] = 1      # 0x6200: 1 = alive
    env._prev = p
    env._glitch_px = 0
    env._reward_max_h = 12               # floor episode -> low-height extra
    r, _ = env._reward(s)
    assert env._glitch_kill, "guard should have executed"
    assert r <= -24, f"execution too cheap: {r}"


def test_x82_stub_rent():
    """Hanging on the g3 broken-ladder stub costs rent; the same spot while
    jumping (g3 jump arcs read h68-77) and the girder beside it are free."""
    def r_at(y, x, jumping=0):
        env = _pbrs_env()
        s = _state(mario_y=y, mario_x=x, is_jumping=jumping)
        env._prev = _state(mario_y=y, mario_x=x, is_jumping=jumping)
        r, _ = env._reward(s)
        return r
    on_stub  = r_at(170, 82)          # h70, mid-stub
    jumping  = r_at(170, 82, 1)       # same spot, jump arc
    assert jumping - on_stub >= 0.05, f"{jumping} vs {on_stub}"


def test_edge_jump_tax():
    """Initiating a jump toward an open girder edge costs the tax; the same
    jump mid-girder is free (user rule: nothing acrobatic at the edge)."""
    def r_jump(x0, x1, tax):
        env = _pbrs_env()
        env.EDGE_JUMP_TAX = tax     # isolate the tax from PBRS arc noise
        p = _state(mario_y=200, mario_x=x0, is_jumping=0)   # 2nd-girder band
        s = _state(mario_y=196, mario_x=x1, is_jumping=1)
        env._prev = p
        r, _ = env._reward(s)
        return r
    # right edge, rightward jump: exactly the tax
    assert abs((r_jump(216, 219, 0.0) - r_jump(216, 219, 2.0)) - 2.0) < 1e-6
    # left edge, leftward jump: covered too (user question 2026-07-18)
    assert abs((r_jump(24, 21, 0.0) - r_jump(24, 21, 2.0)) - 2.0) < 1e-6
    # mid-girder jump: tax setting is irrelevant (scoring skill untouched)
    assert abs(r_jump(120, 123, 0.0) - r_jump(120, 123, 2.0)) < 1e-6


def test_green_light_shifts_peak_up_ladder():
    """With the x53 column clear, potential rises with climb progress above
    the wait-spot peak; with a threat in the column, the ladder holds the
    old flat saturation (waiting stays optimal)."""
    env = _pbrs_env()
    def phi(y, threat):
        s = _state(mario_y=y, mario_x=53)
        if threat:
            s["barrel0_st"] = 1
            s["barrel0_x"] = 53
            s["barrel0_y"] = 150   # above Mario, in-column
        return env._phi(s)
    # clear column: climbing pays in potential
    assert phi(184, False) > phi(192, False) > phi(196, False)
    # blocked column: flat (no pull up the ladder)
    assert abs(phi(184, True) - phi(196, True)) < 1e-9
    # green peak exceeds the wait-spot value
    s_wait = _state(mario_y=202, mario_x=59)
    assert phi(180, False) > env._phi(s_wait)
    # extension: the mid crossing (x115) gets the same pull when clear
    s115 = _state(mario_y=190, mario_x=115)
    s115_hi = _state(mario_y=182, mario_x=115)
    assert env._phi(s115_hi) > env._phi(s115)


def test_margin_feature_agrees_with_gate():
    """The exported climb margins and the reward gates must share a zero
    crossing (review r18: a 20px band used to read 'safe' while the gate
    blocked)."""
    env = _pbrs_env()
    for bx_off in (5, 15, 19, 21, 30, 50):
        s = _state(mario_y=196, mario_x=131)
        s["barrel0_st"] = 1
        s["barrel0_x"] = 131 + (196 - 141) + bx_off   # remaining + offset
        s["barrel0_y"] = 150
        gate = env._ladder_gap_clear(s, 131, 141)
        marg = env._ladder_margin(s, 131, 141)
        if abs(bx_off - env.GAP_MARGIN_PX) > 2:       # off the exact boundary
            assert gate == (marg > 0), f"off {bx_off}: gate {gate} marg {marg}"


def test_x200_east_ladder_climb_is_legal():
    """Climbing the legalized east route ladder must not trip the guard
    (review r18: legal but untested)."""
    env = _pbrs_env()
    env._glitch_px = 0
    p = _state(mario_y=98, mario_x=200)
    s = _state(mario_y=90, mario_x=200)     # 8px ascend, x pinned, in-envelope
    s["is_dead"] = p["is_dead"] = 1
    env._prev = p
    env._reward(s)
    assert not env._glitch_kill, "guard fired on the legal east ladder"


def test_ladder_safety_shading():
    """Legal ladders dim in the threat channel when their column fails the
    time-race gate; bright when clear (user: 'needs to know when it's safe
    to climb vs not' — all nine ladders, not just x53/x131)."""
    import numpy as np
    env = _pbrs_env()
    env._geom = {"w": 224, "h": 256}
    env._ladder_map = env._build_ladder_map()
    env._prev = None
    pix = bytes(224 * 256 * 4)
    clear_s = _state(mario_y=230, mario_x=100)
    obs_clear = env._preprocess(pix, clear_s)
    blocked_s = _state(mario_y=230, mario_x=100)
    blocked_s["barrel0_st"] = 1
    blocked_s["barrel0_x"] = 116          # sits on the x116 column
    blocked_s["barrel0_y"] = 170          # above, within reach window
    obs_blk = env._preprocess(pix, blocked_s)
    sx, sy = 84.0/256.0, 84.0/224.0
    cx = int(round(116 * sx))
    ly = int(184 * sy)                    # mid-x116-ladder row
    band_clear = obs_clear["image"][ly, cx-1:cx+2, 1]
    band_blk = obs_blk["image"][ly, cx-1:cx+2, 1]
    assert band_clear.max() == 255, f"expected bright ladder, {band_clear}"
    assert band_blk.max() == 60, f"expected dimmed ladder, {band_blk}"


def test_gate_sees_fireballs_and_wild_barrels():
    """The safety race counts fireballs (same race) and wild barrels
    (+/-32px berth, no race — they bounce, user doctrine)."""
    env = _pbrs_env()
    base = _state(mario_y=196, mario_x=131)
    assert env._ladder_gap_clear(base, 131, 118)
    fb = dict(base); fb["fireball0_st"] = 1
    fb["fireball0_x"] = 131 + 30; fb["fireball0_y"] = 150
    assert not env._ladder_gap_clear(fb, 131, 118), "fireball in-race missed"
    wild = dict(base); wild["barrel0_st"] = 1; wild["barrel0_crazy"] = 1
    wild["barrel0_x"] = 131 - 28; wild["barrel0_y"] = 150
    assert not env._ladder_gap_clear(wild, 131, 118), "wild in berth missed"
    tame = dict(wild); tame["barrel0_crazy"] = 0
    assert env._ladder_gap_clear(tame, 131, 118), \
        "normal barrel left of the column should not block"


def test_clean_jump_bonus_paid_capped_unfarmable():
    """A barrel passing beneath a jump pays once per arc, caps at 3/episode;
    no barrel = no pay (user: must pale vs the climb, never farmable)."""
    env = _pbrs_env()
    env._clean_jumps = 0
    env._jump_paid = False
    def leap(with_barrel):
        s = _state(mario_y=190, mario_x=102, is_jumping=1)   # x drifts: held dir
        p = _state(mario_y=196, mario_x=100, is_jumping=0)
        if with_barrel:
            s["barrel0_st"] = 1
            s["barrel0_x"] = 102
            s["barrel0_y"] = 205
        env._prev = p
        r, _ = env._reward(s)
        # land (resets the latch)
        env._prev = _state(mario_y=196, mario_x=100, is_jumping=0)
        land = _state(mario_y=196, mario_x=100, is_jumping=0)
        env._reward(land)
        return r
    base = leap(False)
    paid = leap(True)
    assert paid - base >= 0.25, f"{paid} vs {base}"
    # cap: two more pay, the fourth does not
    leap(True); leap(True)
    fourth = leap(True)
    assert fourth - base < 0.05, f"cap failed: {fourth} vs {base}"


def test_clean_jump_requires_held_direction():
    """A vertical hop over a barrel pays nothing — the scan box only
    extends on directional jumps (user pro tip)."""
    env = _pbrs_env()
    env._clean_jumps = 0; env._jump_paid = False
    s = _state(mario_y=190, mario_x=100, is_jumping=1)   # no x drift
    p = _state(mario_y=196, mario_x=100, is_jumping=0)
    s["barrel0_st"] = 1; s["barrel0_x"] = 102; s["barrel0_y"] = 205
    env._prev = p
    env._reward(s)
    assert env._clean_jumps == 0, "vertical hop must not pay"


def test_g2_pocket_rent_asymmetric():
    """Staying in the left pocket costs rent; stepping RIGHT (escaping
    toward the ladder) is free — and PBRS pays the escape (user rule)."""
    env = _pbrs_env()
    def move(px, sx):
        e = _pbrs_env()
        p = _state(mario_y=205, mario_x=px)
        s = _state(mario_y=205, mario_x=sx)
        e._prev = p
        r, _ = e._reward(s)
        return r
    stay = move(30, 30)
    left = move(32, 30)
    right = move(30, 33)
    assert right > stay, f"escape should beat staying: {right} vs {stay}"
    assert right > left, f"escape should beat drifting left: {right} vs {left}"
