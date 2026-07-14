"""Integration-style tests for episode SEMANTICS (reset paths, labeling,
forced-action application, crash recovery) against a fully faked bridge.

The project's worst bugs were never reward math — they were reset/labeling
semantics: the _begin_episode ordering trap (hit by burn-in 28c AND approach
replay 28e), silent fallback mislabeling, and crash rows describing the
post-recovery state. These tests pin each of those down with a scripted
DonkeyKongEnv whose MAME/socket layer is replaced by canned state dicts.

Run with: python -m pytest tests/test_env_semantics.py
"""
import json
import types

import numpy as np
import pytest

import os

from dkong_ai.mame_env import DonkeyKongEnv, ACTIONS


def _state(y, x=100, **kw):
    s = {"mario_y": y, "mario_x": x, "lives": 3, "screen_id": 1,
         "is_dead": 1, "is_jumping": 0, "score": None, "game_start": 1,
         "difficulty": 1}
    s.update(kw)
    return s


BOTTOM = _state(240, 90)
CELL_A = _state(80, 173)    # h160 — carries an approach [2, 2, 3]
CELL_B = _state(70, 150)    # h170 — no approach (burn-in path)


class FakeRNG:
    """np_random stand-in: random() -> 0.0 (every coin lands 'yes' for
    <-comparisons), integers(lo, hi) -> lo (first chain/pos/no jitter)."""

    def random(self):
        return 0.0

    def integers(self, lo, hi=None):
        if hi is None:              # numpy semantics: integers(n) -> [0, n)
            return 0
        return lo


class FakeDK(DonkeyKongEnv):
    def __init__(self, manifest):
        super().__init__(rom_dir="/nonexistent", port=5999, record=False,
                         backward_manifest=manifest, bridge="unused")
        self.sent: list[int] = []
        self.loaded: list[str] = []
        self.responsive = True
        self.crash_next_read = False
        self._sock = types.SimpleNamespace(
            sendall=lambda b: self.sent.append(b[0]),
            close=lambda: None)

    # ---- everything that would touch MAME ------------------------------
    def _launch_mame(self):
        self._proc = types.SimpleNamespace(kill=lambda: None,
                                           wait=lambda timeout=None: None)

    def _connect(self):
        pass

    def _read_handshake(self):
        pass

    def _start_game(self):
        return dict(BOTTOM), b""

    def _save_state(self):
        self._has_state = True

    def _load_state(self):
        return dict(BOTTOM), b""

    def load_state_file(self, path):
        name = os.path.basename(path)   # portable: Windows joins with backslashes
        self.loaded.append(name)
        st = {"cellA.sta": CELL_A, "cellB.sta": CELL_B,
              "anchorA.sta": CELL_A}.get(name, BOTTOM)
        self._cur = dict(st)
        return dict(st), b""

    def _exchange(self, a):
        return dict(getattr(self, "_cur", BOTTOM)), b""

    def _hold(self, a, n):
        return dict(getattr(self, "_cur", BOTTOM)), b""

    def _is_responsive(self):
        st = dict(getattr(self, "_cur", BOTTOM))
        return self.responsive, st, b""

    def _is_live(self, s0):
        st = dict(getattr(self, "_cur", BOTTOM))
        return self.responsive, st, b""

    def _read_obs(self):
        if self.crash_next_read:
            self.crash_next_read = False
            raise ConnectionError("scripted crash")
        return b"", b""

    def _decode_state(self, ram):
        return dict(self._prev)     # hold position: no reward-side motion

    def _preprocess(self, pix, state=None):
        return {"image": np.zeros((84, 84, 2), np.uint8),
                "ram": np.zeros((self.RAM_FEATURE_DIM,), np.float32)}


@pytest.fixture()
def env(tmp_path):
    mani = {"chains": [{"cells": [
        {"sta": "cellA.sta", "height": 160,
         "approach": {"anchor": "anchorA.sta", "acts": [2, 2, 3]}},
        {"sta": "cellB.sta", "height": 170},
    ]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(mani))
    e = FakeDK(str(tmp_path / "manifest.json"))
    e._p_curric = 1.0
    e.P_NO_BARRELS = 0.0
    e.reset()                       # first reset: intro/bottom branch
    e.np_random = FakeRNG()         # deterministic coins from here on
    return e


def test_burnin_survives_begin_episode(env):
    """The 28c ordering trap: the burn-in drawn during the curriculum load
    must still be armed after _begin_episode zeroes per-episode state."""
    env.pin_backward_cell(0, 1)     # cellB: no approach -> burn-in path
    obs, info = env.reset()
    assert env._start_type == "curriculum"
    # Fourth ordering-trap victim: _begin_episode once wiped this, silently
    # marking every success record inexact and starving the harvester.
    assert env._ep_start_sta is not None
    assert env._burnin_left == env.BURN_IN_STEPS
    assert info["burnin"] == env.BURN_IN_STEPS
    env.sent.clear()
    env.step(5)                     # agent asks for JUMP...
    assert env.sent[0] == ACTIONS[0]  # ...burn-in forces NOOP


def test_approach_stash_applied_after_begin_episode(env):
    """The 28e ordering trap: forced approach actions must survive
    _begin_episode and execute verbatim from step 1."""
    env.pin_backward_cell(0, 0)     # cellA: approach [2, 2, 3]
    obs, info = env.reset()
    assert env._forced_actions == [2, 2, 3]
    assert info["approach_len"] == 3
    assert env._pending_approach is None
    assert env._burnin_left == 0    # approach supersedes burn-in
    env.sent.clear()
    env.step(5)
    assert env.sent[0] == ACTIONS[2]  # forced, not the agent's 5


def test_fallback_labels_bottomup_and_attributes_chain(env):
    """A failed curriculum load must become a correctly-labeled bottom
    start: no curriculum labels, no inherited queues, chain attributed."""
    env.pin_backward_cell(0, 0)
    env.responsive = False          # every probe fails -> fallback
    obs, info = env.reset()
    assert env._start_type == "bottomup"
    assert info["bw_start"] is None
    assert info["bw_fallback_chain"] == 0
    assert env._forced_actions == [] and env._pending_approach is None
    assert env._burnin_left == 0    # burn-in is curriculum-only


def test_crash_info_describes_crashed_episode(env):
    """A mid-episode MAME death must log the CRASHED episode's row (labels
    + start position), while the env itself recovers to a bottom start."""
    env.pin_backward_cell(0, 0)
    env.reset()
    assert env._start_type == "curriculum"
    env.crash_next_read = True
    obs, r, term, trunc, info = env.step(0)
    assert term is True
    assert info["start_type"] == "curriculum"       # the crashed episode
    assert info["start_y"] == CELL_A["mario_y"]
    assert env._start_type == "bottomup"            # the recovered env
    assert env._bw_start is None
