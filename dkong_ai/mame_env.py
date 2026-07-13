"""Gymnasium environment that drives MAME (dkong) over the bridge socket.

MAME runs the Lua bridge as a listening server; this env connects as the client,
parses the handshake (screen geometry + discovered input fields), then exchanges
one action byte for one observation per step (lock-step).

Observation: 84x84x2 uint8 — channel 0: grayscale game pixels (4-frame stacked
  by VecFrameStack); channel 1: static ladder-position map (complete ladders only,
  pre-computed from expert corridor — same every frame since barrel board is fixed).
Action: Discrete index -> bitmask over the bridge's CONTROLLED_FIELDS.
Reward: computed from the RAM bytes the bridge ships (see memory_map).

One persistent MAME per env (socket lives for the whole run); episodes reset via
an in-emulator soft-reset. MAME children get PR_SET_PDEATHSIG so they can never
outlive the worker that spawned them.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import struct
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import memory_map

# Action set for the barrel stage: noop + the 5 primitive inputs.
# Index -> bitmask bits matching scripts/bridge.lua CONTROLLED_FIELDS order
# (left, right, up, down, jump).
ACTIONS = [
    0b00000,  # noop
    0b00001,  # left
    0b00010,  # right
    0b00100,  # up
    0b01000,  # down
    0b10000,  # jump
    0b10001,  # jump + left
    0b10010,  # jump + right
]


def _die_with_parent():
    """preexec_fn: ask the kernel to SIGKILL this child if its parent dies.

    Backstop against orphaned MAME processes — if a worker is hard-killed or
    crashes before env.close() can request a clean quit, the kernel reaps MAME."""
    if sys.platform.startswith("linux"):
        PR_SET_PDEATHSIG = 1
        try:
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(
                PR_SET_PDEATHSIG, signal.SIGKILL)
        except OSError:
            pass


class DonkeyKongEnv(gym.Env):
    metadata = {"render_modes": []}

    # Barrel-board ladders that span a full girder gap (x, y_top, y_bottom in
    # game pixels). Used by the channel-1 ladder map AND the broken-ladder
    # glitch guard in _reward: climbing anywhere else is physically possible
    # only via the frame-perfect broken-ladder exploit (x=99 stub et al.),
    # which the go-explore winners and then the policy both discovered.
    COMPLETE_LADDERS = [
        # FLOOR LADDER CORRECTED (2026-07-13): the only complete floor →
        # 2nd-girder ladder is on the RIGHT at x=203 (user ground truth +
        # pro-recording climbs + live probe: x pinned at 203, y 236→211).
        # The old entry (143, 175, 224) was a PHANTOM — its consumers formed
        # a perfect self-built trap: the guard glitch-killed climbs at the
        # real rail, the corner tax covered its base, and PBRS paid Mario to
        # walk away from it. 20K floor draws capped at 8px because of THIS.
        (203, 211, 236),  # floor → 2nd girder, RIGHT side (the real one)
        ( 53, 155, 196),  # 2nd → 3rd girder, FAR LEFT (critical)
        (131, 118, 158),  # 3rd → 4th girder, right-ish
        ( 67,  85, 125),  # 4th → 5th girder, left
        (147,  48, 100),  # top section to Pauline
    ]

    # Max cumulative off-ladder climb pixels per episode before the glitch
    # guard kills (see the guard in _reward). Legal play never accumulates
    # any: complete-ladder climbing is envelope-exempt, jump arcs are
    # is_jumping-gated, and girder walking moves x. 6px ≈ two ratchet pulls.
    GLITCH_PX_MAX = 6

    # Potential-based floor shaping (run 29): the floor "poverty trap" —
    # honest play at the bottom nets negative expected reward because the
    # crossing to the x=143 ladder pays nothing and risks death, so the
    # policy prefers familiar exploits/camping (the ratchet survived 18h of
    # -10 kills because nothing better existed). PBRS (Ng et al.) fills the
    # trap without farmability: r += gamma*PHI(s') - PHI(s) telescopes, so
    # loops/camping pay ~0 (gamma<1 gives a tiny anti-camping bleed) and
    # only genuine crossing progress pays — once. PHI saturates once off
    # the floor so climbing out never LOSES potential; height progress
    # above is the milestone system's job.
    PBRS_COEF = 0.04          # full crossing ~ +5 total
    PBRS_GAMMA = 0.999        # must match training gamma
    # Floor band height. NOT 8 (external review round 5, real bug): the
    # floor girder SLOPES — Mario stands at h~6 on the right end but h~16
    # at the x=143 ladder base — so an 8px band saturated the potential a
    # third of the way into the crossing and the shaping was inert exactly
    # where it mattered. 25 covers the whole slope (and CORNER_H_MAX);
    # the potential still saturates a few px up the ladder, where the
    # milestone system takes over.
    PBRS_FLOOR_H = 25
    PBRS_LADDER_X = 203       # the REAL first ladder (right side; was the
                              # phantom 143 — shaping steered Mario away
                              # from the correct rail into barrel traffic)
    # Stage 2 (2026-07-13, user film review of the h~63 wall): between floor
    # saturation (h25) and the x53 ladder base (h44) lay a reward desert —
    # the girder-2 walk toward the far-left ladder paid nothing, so Mario
    # loitered under the girder-3 edge (barrels that roll off it REVERSE on
    # landing and kill him) instead of reaching the ladder CLIMB_BONUS
    # covers. Geometry verified before shaping (phantom-ladder rule): the 3
    # deepest crossing replays all peak ON the ladder column (x51-65,
    # y163-175). Above h44 the potential is x-independent, so tower play
    # and descents are unaffected.
    PBRS_G2_H = 44            # x53 ladder base; CLIMB_BONUS owns the climb
    PBRS_G2_SPAN = 160        # covers |x-53| back to the ladder-203 arrival

    def _phi(self, s):
        """Crossing-progress potential. State function only — no memory."""
        if s.get("screen_id", 1) != 1 or not s.get("mario_y"):
            return None                       # off-board/off-field: no term
        height = max(0, self.BASE_Y - s["mario_y"])
        if height < self.PBRS_FLOOR_H:        # floor: toward the x203 ladder
            prog = 128 - min(abs(s["mario_x"] - self.PBRS_LADDER_X), 128)
        elif height < self.PBRS_G2_H:         # girder 2: toward the x53 ladder
            prog = 128 + (self.PBRS_G2_SPAN
                          - min(abs(s["mario_x"] - self.LAD53_X),
                                self.PBRS_G2_SPAN))
        else:                                 # saturated above the ladder base
            prog = 128 + self.PBRS_G2_SPAN
        return self.PBRS_COEF * prog

    # _p_curric is a class-level default (like P_NO_BARRELS). Override it by
    # setting an INSTANCE attribute on each constructed env (train.py does
    # this inside the make_env thunk so it happens in the worker process).
    # Mutating the CLASS attribute in the launcher process does NOT work
    # under SubprocVecEnv spawn: workers re-import this module and see the
    # default again — the bug that ran the 27 series at 15%/15%.
    _p_curric = 0.15

    def __init__(self, rom_dir: str, port: int = 5000, frameskip: int = 4,
                 headless: bool = True, mame_bin: str = "mame",
                 bridge: str | None = None, record: bool = True,
                 backward_manifest: str | None = None,
                 extra_mame_args: list[str] | None = None):
        super().__init__()
        self.rom_dir = rom_dir
        self.extra_mame_args = list(extra_mame_args or [])
        self.port = port
        self.frameskip = frameskip
        self.headless = headless
        self.mame_bin = mame_bin
        self.record = record
        self.bridge = bridge or os.path.join(
            os.path.dirname(__file__), "..", "scripts", "bridge.lua")
        self._proc: subprocess.Popen | None = None
        self._inp_path: str | None = None
        self._has_state = False
        self._rxbuf = b""
        self._corridor = self._load_corridor()   # height-band -> expert target x
        self._n_curric = self._count_curriculum()  # expert upper-board start states
        # Backward-algorithm curriculum (phase 2 of Go-Explore): chains of
        # save-states along PROVEN bottom-up winner routes. When set, episodes
        # start from a chain cell no deeper than _bw_level allows; the trainer
        # raises the level (via set_backward_level) as the clear rate rises,
        # walking the start back toward the bottom. Supersedes the legacy
        # expert-demo curric_<idx> states.
        self._bw_chains: list[list[dict]] | None = None
        self._bw_chains_all: list[list[dict]] | None = None  # pin/unpin API
        self._bw_level = 0
        self._bw_levels: list[int] | None = None   # per-chain (preferred)
        self._bw_start: tuple | None = None
        self._bw_fallback_chain = -1
        self._pending_approach: list | None = None   # set by _load_backward_start
        self._bottom_backup_ok = False   # bottom_<port>.sta from THIS instance
        if backward_manifest:
            if record:
                raise ValueError(
                    "backward_manifest requires record=False: recording uses "
                    "clean intro resets (no save-state loads), so the backward "
                    "curriculum could never activate")
            with open(backward_manifest) as f:
                mani = json.load(f)
            base = os.path.dirname(os.path.abspath(backward_manifest))
            self._bw_chains = [
                [{"sta": os.path.join(base, c["sta"]),
                  "height": c.get("height", 0),
                  # approach (optional, augment_approaches.py): mid-leg
                  # anchor + proven action indices arriving at the cell.
                  "approach": ({"anchor": os.path.join(
                                    base, c["approach"]["anchor"]),
                                "acts": c["approach"]["acts"]}
                               if "approach" in c else None)}
                 for c in ch["cells"]]
                for ch in mani["chains"]] or None
            if self._bw_chains is None:
                print("[env] WARNING: backward manifest has no chains — "
                      "backward curriculum disabled", flush=True)
        # _p_curric is a CLASS attribute — do not set self._p_curric here
        self._sock: socket.socket | None = None
        self._geom: dict | None = None
        self._prev: dict | None = None
        self._no_barrels = False   # set per-episode by reset(); drives 0xF8/0xF7 cmd
        self._ladder_map = self._build_ladder_map()  # static; barrel board never changes

        # Dict obs: "image" = 84×84×2 (pixels + threat/ladder/fall-zone map, stacked ×n
        # by DkFrameStackWrapper); "ram" = 75 normalised RAM features giving
        # explicit barrel/fireball/hammer positions, velocities, edge proximity,
        # and barrel type flags (crazy/wild, blue).
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (84, 84, 2), dtype=np.uint8),
            "ram":   spaces.Box(-1.0, 1.0, (self.RAM_FEATURE_DIM,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(len(ACTIONS))

    # ---- process / socket lifecycle -------------------------------------
    def _launch_mame(self):
        logdir = os.path.join(os.path.dirname(self.bridge), "..", "logs")
        os.makedirs(logdir, exist_ok=True)
        bridge_log = os.path.join(logdir, f"bridge_{self.port}.log")
        env = dict(os.environ, DK_BRIDGE_PORT=str(self.port),
                   DK_FRAMESKIP=str(self.frameskip),
                   DK_BRIDGE_LOG=bridge_log)
        statedir = os.path.abspath(os.path.join(os.path.dirname(self.bridge),
                                                "..", "artifacts", "states"))
        os.makedirs(statedir, exist_ok=True)
        # nice +10: the bridge's read_exact busy-spins a full core per MAME
        # while waiting for the next action byte (no blocking read in MAME's
        # Lua sandbox). Profiling (2026-07-11) shows the spin does NOT cap
        # throughput (CNN-era runs hit 900 fps through it; current fps is
        # model-side), but polite spinners keep the trainer's scheduling
        # clean as env count grows.
        args = ["nice", "-n", "10", self.mame_bin, "dkong",
                "-rompath", os.path.abspath(self.rom_dir),
                "-state_directory", statedir,   # shared: per-port + curriculum states
                "-autoboot_script", self.bridge,
                "-autoboot_delay", "0",
                "-skip_gameinfo", "-nothrottle"]
        if self.headless:
            args += ["-video", "none", "-sound", "none"]
            # Sever ALL ties to an X/Wayland display: even with -video none, SDL
            # otherwise opens an X connection to :0, and a display hiccup (WSLg
            # restart, sleep) then kills MAME mid-training. Dummy SDL driver +
            # no DISPLAY = truly headless, immune to display events.
            env["SDL_VIDEODRIVER"] = "dummy"
            env["SDL_AUDIODRIVER"] = "dummy"
            env.pop("DISPLAY", None)
            env.pop("WAYLAND_DISPLAY", None)
        if self.record:
            # Record this MAME session to a .inp for later playback. Finalized
            # only on a clean exit -> close() sends ACT_QUIT. Files are tagged by
            # port + launch time so parallel envs don't collide.
            recdir = os.path.join(os.path.dirname(self.bridge), "..", "artifacts",
                                  "recordings")
            os.makedirs(recdir, exist_ok=True)
            fname = f"dkong_p{self.port}_{int(time.time())}.inp"
            self._inp_path = os.path.abspath(os.path.join(recdir, fname))
            args += ["-input_directory", os.path.abspath(recdir),
                     "-record", fname]
        # Caller-supplied extras (e.g. -aviwrite, -throttle) go LAST so they
        # override any earlier flag — including the headless and record blocks
        # above — since MAME takes the later value for a repeated option.
        args += self.extra_mame_args
        # Redirect MAME output to a per-port file. An unread PIPE here can fill
        # and deadlock MAME before it serves the socket (esp. with many envs).
        self._mame_out = open(os.path.join(logdir, f"mame_{self.port}.out"), "w")
        self._proc = subprocess.Popen(args, env=env, stdout=self._mame_out,
                                      stderr=subprocess.STDOUT,
                                      preexec_fn=_die_with_parent)

    def _connect(self, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=2)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(30.0)  # generous per-step timeout once connected
                self._sock = s
                self._rxbuf = b""   # leftover bytes (e.g. obs piggybacking handshake)
                return
            except OSError:
                time.sleep(0.2)
        raise TimeoutError("could not connect to MAME bridge")

    def _recv_exact(self, n: int) -> bytes:
        # Serve from the leftover buffer first, then the socket. Critical: the
        # handshake read can over-read into the first obs frame; those bytes live
        # in _rxbuf and must not be lost (else the binary stream desyncs).
        buf = bytearray(self._rxbuf[:n])
        self._rxbuf = self._rxbuf[n:]
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("bridge closed")
            buf += chunk
        return bytes(buf)

    def _read_handshake(self):
        # Client speaks first so MAME knows a peer is present, then reads the
        # two newline-terminated text lines: HELLO ... and FIELDS=...
        self._sock.sendall(b"H")
        data = b""
        while data.count(b"\n") < 2:
            chunk = self._sock.recv(4096)
            if not chunk:
                # Peer closed mid-handshake: recv returns b"" forever —
                # without this the loop spins indefinitely instead of
                # letting reset()'s recovery path relaunch MAME.
                raise ConnectionError("bridge closed during handshake")
            data += chunk
        hello, fields, rest = data.split(b"\n", 2)
        self._rxbuf = rest          # keep any obs bytes that piggybacked the handshake
        kv = dict(tok.split(b"=", 1) for tok in hello.split()[1:])
        self._geom = {
            "w": int(kv[b"W"]), "h": int(kv[b"H"]),
            "bpp": int(kv[b"BPP"]), "frameskip": int(kv[b"FRAMESKIP"]),
            "fields": fields.decode().removeprefix("FIELDS=").split(";;"),
        }

    # ---- observation / reward -------------------------------------------
    def _read_obs(self):
        (length,) = struct.unpack(">I", self._recv_exact(4))
        payload = self._recv_exact(length)
        (ram_len,) = struct.unpack(">H", payload[:2])
        ram = payload[2:2 + ram_len]
        pix = payload[2 + ram_len:]
        return ram, pix

    def _decode_state(self, ram: bytes) -> dict:
        # Every watched address is a single byte, in WATCH_ORDER.
        state = {name: ram[i] for i, name in enumerate(memory_map.WATCH_ORDER)}
        state["score"] = memory_map.decode_score(state)   # int or None
        return state

    def _preprocess(self, pix: bytes, state: dict | None = None) -> np.ndarray:
        w, h = self._geom["w"], self._geom["h"]
        arr = np.frombuffer(pix, dtype=np.uint8).reshape(h, w, 4)  # ARGB
        gray = cv2.cvtColor(arr[..., 1:4], cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # Second channel: static ladder map + dynamic barrel/fireball overlays.
        # Ladders=255, active barrels=180, fireball=120. The CNN sees both
        # fixed structure (where ladders are) and live threats simultaneously.
        threat = self._ladder_map.copy()
        if state:
            sx, sy = 84.0 / 256.0, 84.0 / 224.0
            for i in range(6):
                st = state.get(f"barrel{i}_st", 0)
                if st not in (1, 2):
                    continue
                bx, by = state.get(f"barrel{i}_x", 0), state.get(f"barrel{i}_y", 0)
                if not bx or not by:
                    continue
                cx = int(round(bx * sx))
                cy = int(round(by * sy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        threat[max(0, min(83, cy+dy)), max(0, min(83, cx+dx)), 0] = 180
            for _fi in range(5):
                if state.get(f"fireball{_fi}_st", 0):
                    fx = state.get(f"fireball{_fi}_x", 0)
                    fy = state.get(f"fireball{_fi}_y", 0)
                    if fx and fy:
                        cx = int(round(fx * sx))
                        cy = int(round(fy * sy))
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                threat[max(0, min(83, cy+dy)), max(0, min(83, cx+dx)), 0] = 120
            # Hammer pickup: bright dot (80) when available on the board.
            if not state.get("has_hammer", 0):
                hx = state.get("hammer_x", 0)
                hy = state.get("hammer_y", 0)
                if hx and hy:
                    cx = int(round(hx * sx))
                    cy = int(round(hy * sy))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            threat[max(0, min(83, cy+dy)), max(0, min(83, cx+dx)), 0] = 80
            # Fall-zone overlay (intensity 200): when a barrel is within EDGE_PROX
            # pixels of the girder edge it's heading toward, mark the landing zone
            # on the next girder below. Lets the agent see danger arriving from above
            # before the barrel appears — critical for planning around edge falls.
            prev = self._prev or {}
            for i in range(6):
                st = state.get(f"barrel{i}_st", 0)
                if st not in (1, 2):
                    continue
                bx = state.get(f"barrel{i}_x", 0)
                by = state.get(f"barrel{i}_y", 0)
                pbx = prev.get(f"barrel{i}_x", bx)
                pvx = bx - pbx      # positive = moving right
                if not bx or not by or pvx == 0:
                    continue
                for (y_lo, y_hi, x_left, x_right, land_y) in self.GIRDER_EDGES:
                    if not (y_lo <= by < y_hi):
                        continue
                    land_x = x_left if pvx < 0 else x_right
                    dist = (bx - x_left) if pvx < 0 else (x_right - bx)
                    if dist <= self.EDGE_PROX:
                        lx84 = int(round(land_x * sx))
                        ly84 = min(83, int(round(land_y * sy)))
                        for fdy in range(-2, 3):
                            for fdx in range(-4, 5):
                                row = max(0, min(83, ly84 + fdy))
                                col = max(0, min(83, lx84 + fdx))
                                if threat[row, col, 0] < 200:
                                    threat[row, col, 0] = 200
                    break
        image = np.concatenate([small[..., None], threat], axis=-1)  # (84,84,2)
        ram = self._build_ram_features(state) if state else np.zeros(
            self.RAM_FEATURE_DIM, dtype=np.float32)
        return {"image": image, "ram": ram}

    @staticmethod
    def _build_ladder_map() -> np.ndarray:
        """Static 84×84 channel marking ladder positions.

        Complete ladders (spanning a full girder gap) → intensity 255.
        Broken stubs (short segments that can't be climbed) → intensity 128.
        Broken stubs are included so the CNN knows barrel-steering targets:
        a barrel rolling over a broken stub falls through harmlessly.

        Positions: (x_game, y_top_game, y_bot_game) in game pixel coords.
        mario_x ≈ screen_x (0–224); mario_y ≈ screen_y (48=top, 240=floor).
        Derived from live tilemap analysis (0x7400-0x7800) — core ladder tile
        codes C0-C7 / D0-D7 in contiguous column runs identify each segment."""
        m = np.zeros((84, 84), dtype=np.uint8)
        COMPLETE_LADDERS = DonkeyKongEnv.COMPLETE_LADDERS
        BROKEN_STUBS = DonkeyKongEnv.BROKEN_STUBS
        sx, sy = 84.0 / 256.0, 84.0 / 224.0
        for x_g, yt_g, yb_g in COMPLETE_LADDERS:
            x84 = int(round(x_g * sx))
            y_top = int(round(yt_g * sy))
            y_bot = min(83, int(round(yb_g * sy)))
            for dx in (-1, 0, 1):  # 3px wide so stripes survive bilinear upscaling
                col = min(83, max(0, x84 + dx))
                m[y_top:y_bot + 1, col] = 255
        for x_g, yt_g, yb_g in BROKEN_STUBS:
            x84 = int(round(x_g * sx))
            y_top = int(round(yt_g * sy))
            y_bot = min(83, int(round(yb_g * sy)))
            for dx in (-1, 0, 1):
                col = min(83, max(0, x84 + dx))
                # Don't overwrite complete-ladder pixels already drawn at 255
                for row in range(y_top, y_bot + 1):
                    if m[row, col] == 0:
                        m[row, col] = 128
        return m[..., None]  # (84, 84, 1)

    # RAM feature layout (74 values, all in [-1, 1]):
    #   [mario_x/255, mario_y/240]                                              —  2
    #   [Δx/128, Δy/120, vx/8, vy/20, lad53/64, edge_dist, active,
    #    crazy, blue]                            × 6 barrels                   — 54
    #   [Δx/128, Δy/120, active]                × 5 fireballs                  — 15
    #   [Δx/128, Δy/120, has_hammer]            × 1 hammer pickup              —  3
    # vx/vy: per-step velocity (frameskip=4). Horiz max ~2px/frame → norm by 8;
    # vertical ~5px/frame falling → norm by 20.
    # lad53: barrel x-distance to the critical left ladder at x=53, norm by 64.
    # edge_dist: normalised [0,1] distance to the girder edge the barrel is heading
    # toward (0 = at edge / about to fall, 1 = far away). Paired with the fall-zone
    # image overlay so the agent can anticipate barrels appearing from above.
    RAM_FEATURE_DIM = 75   # 2 mario + 6 barrels x 9 + 5 fireballs x 3 + 3 hammer
                           # + 1 difficulty (regime-conditional play: wild-barrel
                           # behaviour differs sharply by 0x6380 regime, and the
                           # diff-5 counter-move is unlearnable without knowing
                           # the regime — user lore, see memory/gameplay tips)
    LAD53_X = 53    # x of the critical left ladder (2nd→3rd girder)

    # Girder edge lookup: (barrel_y_lo, barrel_y_hi, x_left, x_right, landing_y).
    # barrel_y in same coord system as mario_y (0=top, 240=floor).
    # landing_y is the y of the next girder below where the barrel lands.
    GIRDER_EDGES = (
        ( 50,  90, 16, 224,  96),   # near-top area → 5th girder
        ( 88, 122, 16, 224, 128),   # 5th → 4th girder
        (120, 155, 16, 224, 162),   # 4th → 3rd girder
        (152, 186, 16, 224, 196),   # 3rd → 2nd girder
        (184, 225, 16, 224, 240),   # 2nd → floor
    )
    EDGE_PROX = 40   # game-pixel radius within which barrel is "approaching edge"

    def _build_ram_features(self, state: dict) -> np.ndarray:
        """Normalised RAM threat features relative to Mario's position."""
        mx = state.get("mario_x", 0) or 0
        my = state.get("mario_y", 0) or 0
        feats: list[float] = [mx / 255.0, my / 240.0]
        prev = self._prev or {}
        for i in range(6):
            st = state.get(f"barrel{i}_st", 0)
            if st in (1, 2):
                bx = state.get(f"barrel{i}_x", 0)
                by = state.get(f"barrel{i}_y", 0)
                dx    = float(np.clip((bx - mx)           / 128.0, -1, 1))
                dy    = float(np.clip((by - my)           / 120.0, -1, 1))
                pbx   = prev.get(f"barrel{i}_x", bx)
                pby   = prev.get(f"barrel{i}_y", by)
                vx    = float(np.clip((bx - pbx)          /   8.0, -1, 1))
                vy    = float(np.clip((by - pby)          /  20.0, -1, 1))
                lad53 = float(np.clip((bx - self.LAD53_X) /  64.0, -1, 1))
                pvx_sign = bx - pbx
                edge_dist = 1.0
                for (y_lo, y_hi, x_left, x_right, _) in self.GIRDER_EDGES:
                    if y_lo <= by < y_hi:
                        if pvx_sign < 0:
                            edge_dist = float(np.clip(
                                (bx - x_left) / self.EDGE_PROX, 0.0, 1.0))
                        elif pvx_sign > 0:
                            edge_dist = float(np.clip(
                                (x_right - bx) / self.EDGE_PROX, 0.0, 1.0))
                        break
                # Type flags straight from the game's barrel object: the wild
                # (crazy) barrel bounces vertically down the board, so every
                # rolling-barrel prior (edge_dist, fall zones, lad53) is wrong
                # for it; blue barrels end in the oil drum -> fireball spawn.
                crazy = float(state.get(f"barrel{i}_crazy", 0) > 0)
                blue  = float(state.get(f"barrel{i}_blue", 0) > 0)
            else:
                dx = dy = vx = vy = lad53 = edge_dist = crazy = blue = 0.0
            feats.extend([dx, dy, vx, vy, lad53, edge_dist, float(st > 0),
                          crazy, blue])
        for i in range(5):
            fst = state.get(f"fireball{i}_st", 0)
            if fst:
                dx = float(np.clip((state.get(f"fireball{i}_x", 0) - mx) / 128.0, -1, 1))
                dy = float(np.clip((state.get(f"fireball{i}_y", 0) - my) / 120.0, -1, 1))
            else:
                dx = dy = 0.0
            feats.extend([dx, dy, float(fst > 0)])
        has_h = state.get("has_hammer", 0)
        if not has_h:
            dx = float(np.clip((state.get("hammer_x", 0) - mx) / 128.0, -1, 1))
            dy = float(np.clip((state.get("hammer_y", 0) - my) / 120.0, -1, 1))
        else:
            dx = dy = 0.0
        feats.extend([dx, dy, float(bool(has_h))])
        feats.append(state.get("difficulty", 1) / 5.0)
        return np.array(feats, dtype=np.float32)

    # ---- reward shaping helpers -----------------------------------------
    CELL = 16   # (x, height) grid size for the novelty/exploration bonus

    # Zig-zag waypoint milestones: (height_min, is_x_lt, x_threshold, bonus).
    # Each fires at most once per episode — not farmable.
    # WP1a/WP1b split the old single +25 into two stages: +10 for reaching
    # x<75 (agent already does this in run 9) and +20 for the final 22px to
    # the actual ladder entrance at x≈53. Forces the full traverse.
    WAYPOINTS = (
        (36,  True,  140,  5.0),  # WP0: 2nd girder, heading left (x < 140)
        (45,  True,   75, 10.0),  # WP1a: approaching ladder (x < 75)
        (45,  True,   58, 75.0),  # WP1b: AT the ladder entrance (x < 58)
        (65,  False, 100,  8.0),  # WP2: 3rd girder after traverse (x > 100)
        (100, True,   85,  8.0),  # WP3: 3rd-girder left traverse (x < 85)
        (150, False, 130,  8.0),  # WP4: on the final ladder (x > 130)
        (170, False, 100, 20.0),  # WP5: near Pauline — high on the final ladder
    )

    # Girder-level milestones: fire once per episode when Mario first reaches
    # each girder. Bonuses scale up with each level — making higher girders
    # progressively more valuable than farming at the bottom.
    # Heights: 2nd girder~44, 3rd~78, 4th~112, 5th~144, top~182.
    GIRDER_MILESTONES = (
        ( 44, 10.0),   # 2nd girder
        ( 78, 30.0),   # 3rd girder
        (112, 40.0),   # 4th girder
        (144, 55.0),   # 5th girder
        (182, 70.0),   # top / Pauline level
    )
    # Indices 100-104 in _wp_hit to avoid collision with WAYPOINTS (0-5).

    # Per-step climb bonus for the FIRST ladder (ground floor → 2nd girder, x≈143).
    # Same mechanic as CLIMB_BONUS. Without this, the only incentive to climb the
    # first ladder is the one-shot girder milestone (+10), which can't compete with
    # repeatable ground-floor barrel-jump rewards (+0.3 each).
    FIRST_CLIMB_X_LO, FIRST_CLIMB_X_HI = 196, 210   # REAL first ladder (x=203)
    FIRST_CLIMB_H_LO, FIRST_CLIMB_H_HI =   2,  30   # floor → 2nd girder (y236→211)
    FIRST_CLIMB_BONUS      = 0.30
    FIRST_LADDER_IDLE_COST = 0.05

    # Per-step ladder-climb bonus: fires every step Mario is actively ascending
    # the 2nd→3rd girder ladder (mario_y decreasing while at x≈53).
    # Rewards the ACT of climbing — not just approaching — and can't be farmed
    # (requires mario_y to decrease = actually gaining height on the ladder).
    CLIMB_X_LO, CLIMB_X_HI = 43, 68   # ladder column ± tolerance
    CLIMB_H_LO, CLIMB_H_HI = 40, 100  # height band: start of 2nd girder → 3rd
    CLIMB_BONUS      = 0.30            # per step while actively ascending
    LADDER_IDLE_COST = 0.05            # per step in ladder zone but y unchanged

    # Upper-section final-ladder climb bonus: same mechanic as CLIMB_BONUS but
    # for the top ladder at x≈147 (5th girder → Pauline). Fills the reward
    # desert above height 144 where only the sparse height milestone fires.
    UPPER_CLIMB_X_LO, UPPER_CLIMB_X_HI = 137, 160   # top ladder ± tolerance
    # H_HI 192→200 (2026-07-09): cover the last rungs to the clear trigger.
    # BONUS 0.30→0.50, IDLE 0.05→0.15 (user film review: Mario mounts the
    # final ladder then stops/dismounts — "you're on the final ladder,
    # climb it until it's over").
    UPPER_CLIMB_H_LO, UPPER_CLIMB_H_HI = 138, 200   # 5th girder → past Pauline
    UPPER_CLIMB_BONUS      = 0.50
    UPPER_LADDER_IDLE_COST = 0.15

    # Top-rung dead-end tax (2026-07-09, user film review): an INVISIBLE
    # BARRIER stops Mario at x≈107 on the top walkway (measured empirically —
    # 90 LEFT inputs from x155 pin at x107); Kong is unreachable, and the
    # stretch left of the final-ladder mount zone is pure pacing waste. The
    # top walkway sits at mario_y ≤ 80; the 5th girder below starts y ≥ 85,
    # so the y-gate keeps legitimate mid-traverse untaxed.
    TOP_DEADEND_X = 133
    TOP_DEADEND_Y = 80
    TOP_DEADEND_COST = 0.08

    # Dense leftward-progress reward on the 2nd girder: +TRAVERSE_PROGRESS per
    # pixel moved left while in the traverse zone. Provides gradient on EVERY
    # step toward the ladder — even failed attempts that end in death generate
    # useful signal rather than just -10. Complements the one-shot waypoints.
    TRAVERSE_H_LO, TRAVERSE_H_HI = 36, 65
    TRAVERSE_X_LO, TRAVERSE_X_HI = 53, 143   # ladder entrance to right edge
    TRAVERSE_PROGRESS = 0.05                   # reward per pixel moved left

    # Dense rightward-progress reward on the 5th girder: mirrors TRAVERSE_PROGRESS
    # but for the rightward traverse from the 4th→5th ladder (x≈67) to the top
    # ladder entrance (x≈147). This is exactly the stall zone for run 22 (height
    # 146) — no dense signal existed for this 80px traverse.
    UPPER_TRAVERSE_H_LO, UPPER_TRAVERSE_H_HI = 140, 158   # 5th girder band
    UPPER_TRAVERSE_X_LO, UPPER_TRAVERSE_X_HI =  67, 147   # arrival → top ladder
    UPPER_TRAVERSE_PROGRESS = 0.05                         # per pixel moved right

    # Anti-camping: height band and x threshold for the per-step girder penalty.
    # Applies only to the right side of the 2nd girder — the zone where the agent
    # farms barrels instead of traversing to the ladder. Traversing left is free.
    CAMP_H_LO, CAMP_H_HI, CAMP_X, CAMP_COST = 36, 65, 130, 0.01

    # Bottom-floor corner penalty: per-step cost for being in the dead-end corners
    # of the ground floor. Left corner = past the left wall (x<30); right corner =
    # past the first ladder at x≈143 heading toward the right wall (x>160).
    # CORNER_H_MAX=25: ground floor is mario_y≈220-224 → height≈16-20, so the
    # threshold must be >20 to actually fire. 15 was too low and never triggered.
    CORNER_H_MAX   = 25
    CORNER_X_LEFT  = 30
    # 160 -> 156 (2026-07-07): FIRST_CLIMB zone ends at 155, so x in (155,160]
    # was a 5px no-penalty safe harbor beside the ladder base — and eval film
    # showed the policy camping exactly there. The box now starts where the
    # climb zone ends.
    # 156 -> 214 (2026-07-13, floor-geometry correction): the real first
    # ladder lives at x=203 — the old box taxed Mario for standing AT its
    # base. The true dead-end corner is only the sliver past the ladder.
    CORNER_X_RIGHT = 214
    CORNER_COST    = 0.20

    # Broken-ladder stub tax (2026-07-07): the x=99 stub's lower rungs are
    # legal ladder tiles, so the glitch guard can't fire on them — eval film
    # showed ritual re-climbing (8.4%% of bottom-ups still END via the guard).
    # Loitering in the stub zone now costs per step; the legit floor ladder
    # (x=143) is bonus-paid, so the gradient points away from the stub.
    STUB_X_LO, STUB_X_HI = 92, 106
    STUB_H_LO, STUB_H_HI = 10, 40
    # Oil-can corner (bottom-left): film review #5 (user, 2026-07-13) caught
    # floor spawns wandering LEFT and jumping into the can. Deaths there
    # already pay -10/-15 and PBRS drains leftward movement — the gap was
    # the novelty bonus SUBSIDIZING first visits to the death zone. Same
    # remedy as the stub zone: no novelty/corridor pay, no extra penalty
    # (the pro route legitimately touches this region when point-milking).
    OIL_X_HI = 44
    OIL_H_HI = 20
    STUB_COST = 0.08

    # Short stubs that hang from a girder but don't connect below.
    # Barrels can fall through these; Mario cannot climb them.
    # Positions from tilemap core-code run analysis (col*8 ≈ x, row*8 ≈ y).
    # Used by the threat map AND by the generalized stub tax (2026-07-09:
    # film review by the user showed the mid-board walls at h131/h163 were
    # Mario ritual-climbing untaxed mid-board stubs instead of traversing to
    # the complete ladder — the floor-stub disease, one flight up).
    BROKEN_STUBS = (
        ( 64, 144, 152),  # 4th→3rd gap at x≈64
        (104,  56,  72),  # top area stub at x≈104
        (120, 160, 184),  # 3rd→2nd partial at x≈120
        (144, 104, 120),  # 5th→4th gap at x≈144
        (160, 112, 136),  # through 4th girder at x≈160
        (168, 176, 192),  # 3rd→2nd partial at x≈168
        (200, 120, 152),  # through 4th girder at x≈200
    )
    STUB_X_TOL = 7        # |mario_x - stub_x| within which the tax applies
    STUB_Y_PAD = 6        # stub y-span padding (mount/dismount frames)

    # Score-gating zone: block barrel-jump score reward only when camping
    # (height<65, x>115, AND not moving left). Traversing left through the zone
    # unlocks score so the agent is rewarded for jumping barrels en route — the
    # same mechanic it already knows but now applied during the traverse.
    SCORE_GATE_H, SCORE_GATE_X = 65, 115

    # Episode height timeout: if Mario hasn't crossed height HEIGHT_TIMEOUT_H
    # within HEIGHT_TIMEOUT_STEPS steps, force-terminate the episode with a
    # large penalty. Prevents the hammer-farm-until-dead strategy because every
    # episode now has a hard deadline for making crossing progress.
    HEIGHT_TIMEOUT_STEPS   = 800
    HEIGHT_TIMEOUT_H       = 60
    HEIGHT_TIMEOUT_PENALTY = 15.0

    # Hammer-at-left-wall penalty: per-step cost when Mario holds the hammer
    # AND is parked at the left wall above ground-floor height. This directly
    # penalises the observed failure mode: grab hammer → run left → stand at
    # wall → wait for hammer to expire → die at x≈20.
    HAMMER_WALL_X    = 45     # left of this = "parked at left wall"
    HAMMER_WALL_H_LO = 25     # only above ground floor (farming zone)
    HAMMER_WALL_COST = 0.05   # per-step penalty

    def _count_curriculum(self):
        d = os.path.join(os.path.dirname(self.bridge), "..", "artifacts",
                         "states", "dkong")
        try:
            return len([f for f in os.listdir(os.path.abspath(d))
                        if f.startswith("curric_") and f.endswith(".sta")])
        except OSError:
            return 0

    def _load_corridor(self):
        path = os.path.join(os.path.dirname(self.bridge), "..", "artifacts",
                            "expert_corridor.json")
        try:
            with open(os.path.abspath(path)) as f:
                return json.load(f)            # [{h_lo,h_hi,x_med,...}, ...]
        except OSError:
            return None

    def _target_x(self, height: int):
        if not self._corridor:
            return None
        for c in self._corridor:
            if c["h_lo"] <= height < c["h_hi"]:
                return c["x_med"]
        return self._corridor[-1]["x_med"]      # above the top band -> Pauline x

    def _reward(self, s: dict) -> tuple[float, bool]:
        """Exploration-driven climb shaping, to break the right-camping local
        optimum (agent farmed barrels at height ~47 and never traversed left to
        the ladder). Terms:
          * HEIGHT MILESTONE: 0.5 per new pixel of max height (top ~+100).
          * NOVELTY: first visit to an (x,height) grid cell this episode pays
            0.2, plus up to 0.3 more if that cell is on the expert's route
            (corridor). Paid once per cell -> not farmable; rewards exploring
            new ground, especially left/up toward the ladders.
          * SCORE (de-weighted, artifact-guarded), death -10, clear +100.
        Descending to dodge is free (no penalty)."""
        if self._prev is None:
            return 0.0, False
        p = self._prev
        died = s["lives"] < p["lives"]
        r_pbrs = 0.0
        phi_s, phi_p = self._phi(s), self._phi(p)
        if phi_s is not None and phi_p is not None:
            r_pbrs = self.PBRS_GAMMA * phi_s - phi_p
        cleared = s["screen_id"] > p["screen_id"]
        r = r_pbrs
        if not died and not cleared:
            if s["mario_y"]:
                height = self.BASE_Y - s["mario_y"]
                not_jumping = not s.get("is_jumping", 0)
                # Height milestone: pay only for NEW max height, and only when
                # not mid-jump — jump apexes are transient and shouldn't give
                # credit for heights Mario hasn't actually stood on.
                if height > self._reward_max_h and not_jumping:
                    r += (height - self._reward_max_h) * 0.5
                    self._reward_max_h = height
                # Per-step height bonus: small continuous reward for being higher.
                # Gives the value function a gradient through the wall so the agent
                # "knows" height 100 > height 54 without ever having cleared.
                r += 0.003 * height / 100.0
                # Zig-zag waypoint milestones: fire once per episode at each
                # inflection point of the expert route. The first two pull
                # specifically toward the 2nd-girder left traverse.
                for _i, (_hmin, _lt, _xv, _bon) in enumerate(self.WAYPOINTS):
                    if _i not in self._wp_hit and height >= _hmin:
                        if (s["mario_x"] < _xv) if _lt else (s["mario_x"] > _xv):
                            self._wp_hit.add(_i)
                            r += _bon
                # Girder-level milestones: progressive bonuses for reaching each
                # new girder — higher levels pay more, making climbing always
                # worth more than staying at the bottom to farm.
                for _j, (_hmin, _bon) in enumerate(self.GIRDER_MILESTONES):
                    _key = 100 + _j
                    if _key not in self._wp_hit and height >= _hmin:
                        self._wp_hit.add(_key)
                        r += _bon
                # Anti-camping: small per-step cost for lingering on the right
                # side of the 2nd girder (where the agent farms barrel-jumps
                # instead of traversing left to the ladder). Traversing left
                # past CAMP_X removes the cost immediately.
                if (self.CAMP_H_LO <= height <= self.CAMP_H_HI
                        and s["mario_x"] > self.CAMP_X
                        and not s.get("has_hammer", 0)):
                    r -= self.CAMP_COST
                # Bottom-floor corner penalty: dead-end corners on the ground
                # floor only (height<15). Left of x=30 or right of x=190
                # (past the ladder) serve no purpose.
                if (height < self.CORNER_H_MAX
                        and (s["mario_x"] < self.CORNER_X_LEFT
                             or s["mario_x"] > self.CORNER_X_RIGHT)):
                    r -= self.CORNER_COST
                # Broken-ladder stub tax: see STUB_* constants.
                if (self.STUB_H_LO <= height <= self.STUB_H_HI
                        and self.STUB_X_LO <= s["mario_x"] <= self.STUB_X_HI):
                    r -= self.STUB_COST
                # Generalized mid-board stub tax: loitering at ANY broken
                # stub costs the same rent. Small enough that a 1-2s
                # dodge-mount stays worth it vs a death; large enough that
                # ritual re-climbing (the h131/h163 walls) stops being free.
                else:
                    for sx_, yt_, yb_ in self.BROKEN_STUBS:
                        if (abs(s["mario_x"] - sx_) <= self.STUB_X_TOL
                                and yt_ - self.STUB_Y_PAD <= s["mario_y"]
                                    <= yb_ + self.STUB_Y_PAD):
                            r -= self.STUB_COST
                            break
                # Top-rung dead-end tax: see TOP_DEADEND_* constants.
                if (s["mario_y"] and s["mario_y"] <= self.TOP_DEADEND_Y
                        and s["mario_x"] < self.TOP_DEADEND_X):
                    r -= self.TOP_DEADEND_COST
                # Dense traverse progress: +TRAVERSE_PROGRESS per pixel moved
                # left on the 2nd girder. Gives gradient on every failed attempt
                # (not just when the agent survives all the way to x=53).
                if (self.TRAVERSE_H_LO <= height <= self.TRAVERSE_H_HI
                        and self.TRAVERSE_X_LO < s["mario_x"] <= self.TRAVERSE_X_HI
                        and s["mario_x"] < p["mario_x"]):
                    r += (p["mario_x"] - s["mario_x"]) * self.TRAVERSE_PROGRESS
                # Dense rightward-progress reward on the 5th girder: same
                # mechanic for the traverse from the 4th→5th ladder to the top
                # ladder entrance (the stall zone above height 140).
                if (self.UPPER_TRAVERSE_H_LO <= height <= self.UPPER_TRAVERSE_H_HI
                        and self.UPPER_TRAVERSE_X_LO <= s["mario_x"] < self.UPPER_TRAVERSE_X_HI
                        and s["mario_x"] > p["mario_x"]):
                    r += (s["mario_x"] - p["mario_x"]) * self.UPPER_TRAVERSE_PROGRESS
                # First-ladder climb bonus: per-step reward for actively ascending
                # the ground-floor → 2nd girder ladder at x≈143. Without this,
                # the only incentive to climb is the one-shot girder milestone,
                # which loses to repeatable ground-floor barrel-jump farming.
                # Gated on is_jumping==0 so the bonus only fires during real
                # ladder climbing, not during jump arcs at the ladder base.
                if (not_jumping
                        and self.FIRST_CLIMB_H_LO <= height <= self.FIRST_CLIMB_H_HI
                        and self.FIRST_CLIMB_X_LO <= s["mario_x"] <= self.FIRST_CLIMB_X_HI):
                    if s["mario_y"] < p["mario_y"]:
                        r += self.FIRST_CLIMB_BONUS
                    elif s["mario_y"] == p["mario_y"]:
                        r -= self.FIRST_LADDER_IDLE_COST
                # Ladder-climb bonus: per-step reward for actively ascending the
                # 2nd→3rd girder ladder. Requires mario_y to decrease (gaining
                # height) while positioned at the ladder column — can't be farmed
                # by standing still.
                if (not_jumping
                        and self.CLIMB_H_LO <= height <= self.CLIMB_H_HI
                        and self.CLIMB_X_LO <= s["mario_x"] <= self.CLIMB_X_HI):
                    if s["mario_y"] < p["mario_y"]:
                        r += self.CLIMB_BONUS
                    elif s["mario_y"] == p["mario_y"]:
                        r -= self.LADDER_IDLE_COST
                # Upper ladder climb bonus: same mechanic for the final ladder
                # (x≈147, 5th girder → Pauline). Fills the reward desert above
                # height 144 where only the sparse height milestone fires.
                if (not_jumping
                        and self.UPPER_CLIMB_H_LO <= height <= self.UPPER_CLIMB_H_HI
                        and self.UPPER_CLIMB_X_LO <= s["mario_x"] <= self.UPPER_CLIMB_X_HI):
                    if s["mario_y"] < p["mario_y"]:
                        r += self.UPPER_CLIMB_BONUS
                    elif s["mario_y"] == p["mario_y"]:
                        r -= self.UPPER_LADDER_IDLE_COST
                # Novelty + corridor: reward first visit to a new (x,height)
                # cell, more if it's on the expert route.
                cell = (s["mario_x"] // self.CELL, height // self.CELL)
                if cell not in self._visited:
                    self._visited.add(cell)
                    # No novelty/corridor pay inside the broken-stub zone —
                    # it was a small per-episode bounty for visiting the
                    # glitch ladder's neighbourhood. Ditto the oil-can
                    # corner (film review #5: subsidized wandering into
                    # fireball/can deaths at the bottom-left).
                    if not (self.STUB_H_LO <= height <= self.STUB_H_HI
                            and self.STUB_X_LO <= s["mario_x"] <= self.STUB_X_HI) \
                       and not (height <= self.OIL_H_HI
                                and s["mario_x"] <= self.OIL_X_HI):
                        r += 0.2
                        tx = self._target_x(height)
                        if tx is not None:
                            r += 0.3 * max(0.0, 1.0 - abs(s["mario_x"] - tx) / 48.0)
            # Score gains, de-weighted. Guard ignores <=0 (incl. the pre-game
            # 3700->0 reset) and implausible jumps (artifact / glitch).
            # Score is GATED in the camp zone: no score reward when height < 65
            # and mario_x > 115. This removes barrel-jump income from the right
            # side of the 2nd girder — the camping penalty is no longer offset.
            if s["score"] is not None and p["score"] is not None:
                gained = s["score"] - p["score"]
                # Gate: block score when camping (low + right side + not moving
                # left). Also unconditionally block in the right corner (x>190,
                # height<15) — no barrel-jump reward for the dead-end wall.
                in_corner = (s["mario_y"] is not None
                             and s["mario_y"] > self.BASE_Y - self.CORNER_H_MAX
                             and s["mario_x"] > self.CORNER_X_RIGHT)
                in_gate = in_corner or (
                           s["mario_y"] is not None
                           and s["mario_y"] > self.BASE_Y - self.SCORE_GATE_H
                           and s["mario_x"] > self.SCORE_GATE_X
                           and s["mario_x"] >= p["mario_x"])
                if 0 < gained <= 2000 and not in_gate:
                    r += gained * 0.003         # +0.3 per 100-pt barrel jump
            # Hammer-at-left-wall penalty: penalise the exact failure we
            # observed — grab hammer, run to left wall, stand waiting for
            # it to expire. Only fires above ground-floor height so the
            # ground-floor corner penalty isn't double-counted.
            # Deliberately OUTSIDE the score-validity gate above: it has no
            # score dependency, and score decode legitimately returns None on
            # volatile HUD frames — an indentation accident had it skipping
            # exactly then (external review finding, fixed 2026-07-10).
            if (s.get("has_hammer", 0)
                    and s["mario_x"] < self.HAMMER_WALL_X
                    and s["mario_y"] is not None
                    and s["mario_y"] < self.BASE_Y - self.HAMMER_WALL_H_LO):
                r -= self.HAMMER_WALL_COST
        # Broken-ladder glitch guard: sustained climbing (y falling, x pinned,
        # not a jump arc, alive — 0x6200 is 1=alive) anywhere outside a
        # complete ladder's envelope is the frame-perfect broken-ladder
        # exploit (all 20/20 probed bottom episodes rode the x=99 stub).
        # Three consecutive glitch-climb steps end the episode as a death:
        # the exploit simply doesn't exist in our version of the game.
        glitch_killed = False
        if (not died and not cleared and s["screen_id"] == 1
                and s.get("is_dead", 0) == 1 and not s.get("is_jumping", 0)
                and s["mario_y"] and p["mario_y"]
                and s["mario_y"] < p["mario_y"]
                and s["mario_x"] == p["mario_x"]
                and not any(abs(s["mario_x"] - lx) <= 6
                            and yt - 6 <= s["mario_y"] <= yb + 6
                            for lx, yt, yb in self.COMPLETE_LADDERS)):
            # CUMULATIVE pixels, not consecutive steps (fixed 2026-07-11,
            # film review #4): the old 3-consecutive-streak guard reset on
            # every pause, so the policy learned a climb-pause-climb RATCHET
            # — a trajectory audit showed 100% of bottom-up height gain
            # coming from the x=99 stub in 20/20 episodes, only 1 of which
            # tripped the streak guard. Pausing must not launder the climb.
            self._glitch_px += p["mario_y"] - s["mario_y"]
            if self._glitch_px >= self.GLITCH_PX_MAX:
                glitch_killed = True
                self._glitch_kill = True
        if glitch_killed:
            r -= 10.0
        if died:
            r -= 10.0
            # Extra penalty when the agent never left the farming zone this episode.
            # Uses max height reached (not death position) so a genuine crossing
            # attempt that fails mid-route isn't punished — only pure farming deaths.
            if self._reward_max_h < 40:
                r -= 5.0
        if cleared:
            r += 100.0                          # reaching Pauline is the goal
        # Episode height timeout: force-terminate if Mario hasn't reached the
        # ladder crossing zone within HEIGHT_TIMEOUT_STEPS steps. Removes the
        # option of farming indefinitely; every episode now has a deadline.
        self._episode_steps += 1
        timed_out = (self._episode_steps >= self.HEIGHT_TIMEOUT_STEPS
                     and self._reward_max_h < self.HEIGHT_TIMEOUT_H)
        if timed_out:
            r -= self.HEIGHT_TIMEOUT_PENALTY
        # Single-life episodes: ANY death terminates. Multi-life episodes were
        # a relic of 19s intro resets (fewer intros per env-step); with 0.03s
        # save-state resets they only hurt — especially backward-curriculum
        # starts, where dying at the top silently respawned Mario at the
        # BOTTOM and the episode continued as a mislabeled, unclearable
        # bottom-up run that drowned the frontier gradient.
        done = died or cleared or timed_out or glitch_killed
        return r, done

    # ---- gym API ---------------------------------------------------------
    # Action byte constants understood by the bridge.
    A_NOOP, A_COIN, A_START, A_RESET, A_QUIT = 0x00, 0xF1, 0xF2, 0xFE, 0xFD
    A_SAVE, A_LOAD = 0xFC, 0xFB
    A_RIGHT = 0b00010   # play bitmask: move right (controllability probe)
    # Training-wheels barrel freeze: 0xF8 zeroes all barrel+fireball status bytes
    # in MAME RAM each frame so they can't kill Mario. 0xF7 re-enables them.
    # Python sends one or the other at the start of every episode.
    A_FREEZE_BARRELS, A_UNFREEZE_BARRELS = 0xF8, 0xF7
    P_NO_BARRELS = 0.15  # fraction of episodes that run with barrels disabled

    def _exchange(self, action_byte: int):
        self._sock.sendall(bytes([action_byte]))
        ram, pix = self._read_obs()
        return self._decode_state(ram), pix

    def _hold(self, action_byte: int, n: int):
        state = pix = None
        for _ in range(n):
            state, pix = self._exchange(action_byte)
        return state, pix

    def _start_game(self):
        """Insert coin, press start, and WAIT OUT the ~19s intro until Mario is
        actually controllable.

        Detection uses input RESPONSE, not RAM flags: 0x6200 ("is_dead") is
        unreliable here — Mario is alive, at the start, and walks under our input
        while it still reads 1. So we just hold RIGHT and watch for Mario WALKING
        (sustained increasing mario_x at the start row). The frozen intro-Mario
        ignores input; the brief (0,0) hand-over blip isn't a sustained walk.
        Holding right is also the safe opening (away from the left fireball)."""
        self._hold(self.A_NOOP, 200)            # let boot RAM-test + attract settle
        for _attempt in range(3):
            self._hold(self.A_COIN, 15)
            self._hold(self.A_NOOP, 10)
            self._hold(self.A_START, 15)
            inc = 0
            state, pix = self._exchange(self.A_RIGHT)
            for _ in range(500):                # up to ~33s; intro is ~19s
                px = state["mario_x"]
                state, pix = self._exchange(self.A_RIGHT)
                if state["mario_x"] > px and state["mario_y"] > 150:
                    inc += 1
                    if inc >= 4:                # walking right -> level is live
                        return state, pix
                else:
                    inc = 0
        return state, pix   # give up gracefully; caller still gets a frame

    def _save_state(self):
        for _ in range(4):                   # save is scheduled; let it complete
            self._exchange(self.A_SAVE)
        self._hold(self.A_NOOP, 3)
        self._has_state = True
        if self._bw_chains is not None:
            # Backward curriculum swaps chain states onto this slot file, so
            # keep a pristine copy of the bottom start; load_state_file puts
            # it back after each swap, keeping "load slot" == "bottom".
            shutil.copyfile(self._slot_sta_path(), self._bottom_sta_path())
            self._bottom_backup_ok = True

    def _load_state(self):
        state = pix = None
        for _ in range(3):                   # load is scheduled; let it complete
            state, pix = self._exchange(self.A_LOAD)
        state, pix = self._hold(self.A_NOOP, 2)
        return state, pix

    def set_backward_level(self, level: int):
        """Trainer hook (env_method): allow starts this many cells back from
        each chain's end. Level 0 = final cell only (nearest the goal)."""
        self._bw_level = int(level)

    def set_backward_levels(self, levels):
        """Per-chain walk-back levels (list aligned with the manifest's
        chains). Lets each route descend independently: one hard/awkward
        frontier cell stalls only its own chain, and the walk-back flows
        down the easiest route first — one route to the bottom is enough."""
        self._bw_levels = [int(x) for x in levels]

    def _slot_sta_path(self):
        statedir = os.path.abspath(os.path.join(os.path.dirname(self.bridge),
                                                "..", "artifacts", "states"))
        return os.path.join(statedir, "dkong", f"dk_{self.port}.sta")

    def _bottom_sta_path(self):
        return self._slot_sta_path().replace(f"dk_{self.port}",
                                             f"bottom_{self.port}")

    def load_state_file(self, sta_path: str):
        """Load an arbitrary .sta through this port's slot — the ONE primitive
        every slot-swap consumer must use (backward curriculum, go_explore
        restores, replays). Swaps the file in, loads it, then puts the bottom
        backup straight back (the loads already consumed the swapped file),
        so the invariant "slot file == bottom start" survives; see the
        slot-clobber bug in HANDOFF.md for why that matters."""
        if not os.path.exists(sta_path):
            # Deliberately NOT an OSError: reset()'s except treats OSError as
            # "MAME died" and would relaunch MAME forever; a missing state
            # file is a config error that must fail fast and readably.
            raise RuntimeError(f"state file missing: {sta_path}")
        shutil.copyfile(sta_path, self._slot_sta_path())
        self._has_state = True
        state, pix = self._load_state()
        if self._bottom_backup_ok:
            shutil.copyfile(self._bottom_sta_path(), self._slot_sta_path())
        return state, pix

    BW_REHEARSAL_CAP = 8   # rehearse only the K cells above the frontier

    # ---- curriculum control API (eval_battery, film_cells, probes) -------
    # Tooling used to mutate _bw_chains/_bw_levels/_p_curric directly, which
    # is brittle (a smoke-test pin once grabbed the wrong cell) and easy to
    # regress. These methods own the invariants; use them instead.

    def pin_backward_cell(self, ci: int, pos: int):
        """Restrict curriculum draws to exactly one cell (filming/probing).
        The full chain set is preserved for unpin_backward()."""
        if not hasattr(self, "_bw_chains_all") or self._bw_chains_all is None:
            self._bw_chains_all = self._bw_chains
        cell = self._bw_chains_all[ci][pos]
        self._bw_chains = [[cell]]
        self._bw_levels = [0]
        self._p_curric = 1.0

    def set_bottomup_eval(self):
        """All episodes start from the bottom (honest-floor evaluation)."""
        self._p_curric = 0.0

    def unpin_backward(self):
        """Restore the full chain set after pin_backward_cell()."""
        if getattr(self, "_bw_chains_all", None) is not None:
            self._bw_chains = self._bw_chains_all

    def _load_backward_start(self):
        """Start from a random winner-chain cell within the walk-back window
        [n-1-_bw_level, n-1]. Returns (state, pix), or (None, None) if the
        snapshot is unresponsive (caller falls back to the bottom start)."""
        ci = int(self.np_random.integers(len(self._bw_chains)))
        chain = self._bw_chains[ci]
        n = len(chain)
        level = (self._bw_levels[ci] if self._bw_levels is not None
                 else self._bw_level)
        lo = max(0, n - 1 - level)
        # Most draws drill the frontier tier (deepest allowed cell); the
        # rest rehearse the whole window uniformly. Pure-uniform sampling
        # starves the frontier of practice as the window grows (1/(k+1) of
        # draws), which is exactly the tier that needs the gradient.
        # 0.5 -> 0.7 (2026-07-05): the shelf tiers (x=147 ladder-grab
        # precision) sat at 2-5% for ~20M steps — the failing skill needs
        # reps more than the mastered tower needs extra rehearsal; the
        # consolidation governor guards the tower side.
        # 0.7 -> 0.5 (2026-07-05 eve, run27r post-mortem): at 0.7 the governor
        # did NOT guard the tower — 70% of draws grinding an ~8% frontier
        # (c446 pool) decayed the adjacent promoted tier 43%->5% in 2.4M steps
        # (gradient interference: the states are 4 macro-steps apart), which
        # pinned pooled rehearsal at ~0.55 < CONSOL_OFF 0.68, freezing
        # promotions for the whole run. Freezing stops promotion, not decay;
        # only rehearsal share does. Deadlock breaker, not a tuning taste.
        if self.np_random.random() < 0.5:
            pos = lo
        else:
            # Rehearsal capped to the K cells just above the frontier
            # (2026-07-10, run 28): uniform-over-all-passed spreads the
            # maintenance budget over the whole tower, so per-cell volume
            # ~50%/level -> starvation at deep levels — measured as the
            # "hollow tower" (tiers gated at 0.31 decayed to 0-13% with
            # ~20 draws/h each). Recently-passed cells are the fragile
            # ones; the old deep cells are the near-top band, which every
            # chain's window overlaps anyway.
            hi = min(lo + self.BW_REHEARSAL_CAP, n - 1)
            pos = int(self.np_random.integers(lo, hi + 1))
        cell = chain[pos]
        # Freeze mode is PERSISTENT in the bridge until explicitly changed,
        # and the per-episode mode command is sent AFTER this load — so a
        # barrel-frozen previous episode would leave the world frozen during
        # the motion-based liveness probe and every valid state would fail
        # to bottom (external review round 6; latent while P_NO_BARRELS=0).
        # CONDITIONAL on the stale flag: an unconditional unfreeze would add
        # an exchange to every load, phase-shifting all fixed-RNG cells and
        # invalidating recorded success reproductions.
        if self._no_barrels:
            self._exchange(self.A_UNFREEZE_BARRELS)
        approach = cell.get("approach")
        # Approach draws are STOCHASTIC (run 28g): when the approach
        # monopolized spawns, a single bad approach starved the cell of the
        # clean spawns it could already clear (c446_d5: approach 1% over
        # 228 draws vs clean 67% — the gate could never fire). Coin-flip
        # keeps both modes flowing and the CSV split attributes them.
        if approach and self.np_random.random() < 0.5:
            # APPROACH REPLAY (run 28e, film reviews #3/#4): instead of
            # cold-dropping the policy on the cell with a zeroed LSTM, load
            # the mid-leg anchor and FORCE the proven approach actions for
            # the first steps of the episode — the LSTM fills with the real
            # arrival context and control lands at the cell's original game
            # time (zero bonus-timer cost, unlike the NOOP burn-in).
            # Handover is randomized (drop a random 0..HANDOVER_JITTER
            # suffix): diversity along a proven-survivable path — the safe
            # replacement for the idle-jitter removed in 27j.
            s0, _ = self.load_state_file(approach["anchor"])
            self._ep_start_sta = approach["anchor"]
            ok, state, pix = self._is_live(s0)
            if not ok:
                self._bw_fallback_chain = ci
                self._ep_start_sta = None
                return None, None
            drop = int(self.np_random.integers(0, self.HANDOVER_JITTER + 1))
            acts = approach["acts"]
            # Stashed, not applied: _begin_episode (called later in reset)
            # zeroes the per-episode action queue — reset() applies this
            # AFTER _begin_episode, mirroring the burn-in ordering.
            self._pending_approach = list(acts[:max(1, len(acts) - drop)])
            self._bw_start = (ci, pos, n, cell["height"])
            return state, pix
        s0, _ = self.load_state_file(cell["sta"])
        self._ep_start_sta = cell["sta"]
        ok, state, pix = self._is_live(s0)
        if not ok:
            self._ep_start_sta = None
            # Attribute the fallback: the episode silently becomes a bottom
            # start, and per-chain fallback rates are invisible without this
            # (floor chains drew 23 eps/h instead of ~350 — WC states are
            # probe-flaky as a class and we couldn't see which).
            self._bw_fallback_chain = ci
            return None, None
        self._bw_start = (ci, pos, n, cell["height"])
        return state, pix

    def _load_curriculum(self, idx):
        """Load expert upper-board start-state curric_<idx> (bridge cmd 0xE0+idx)."""
        state = pix = None
        for _ in range(3):
            state, pix = self._exchange(0xE0 + idx)
        return self._hold(self.A_NOOP, 2)

    def _is_live(self, s0):
        """Frozen-snapshot check WITHOUT driving Mario: advance 2 NOOP
        exchanges and confirm the WORLD moves (any barrel/fireball position
        changes). The old input-response probe walked Mario blindly into
        traffic — at floor-band cells a barrel reaches the spawn inside the
        probe window on fixed RNG, so the probe itself died on ~85% of
        floor-chain loads (run 28g doom-triage: wc_011 probe-suicide 4/4).
        Static pre-checks: Mario alive (0x6200==1) and on-field (y!=0).
        Returns (ok, state, pix)."""
        if not s0.get("mario_y") or s0.get("is_dead", 1) != 1:
            return False, s0, b""
        dyn = [f"barrel{i}_{a}" for i in range(6) for a in ("x", "y")] + \
              [f"fireball{i}_{a}" for i in range(5) for a in ("x", "y")]
        st, pix = self._exchange(self.A_NOOP)
        for _ in range(2):
            st2, pix = self._exchange(self.A_NOOP)
            if any(st2.get(k, 0) != st.get(k, 0) for k in dyn):
                return True, st2, pix
            st = st2
        return False, st, pix

    def _is_responsive(self):
        """Probe whether Mario actually responds to input (some curriculum
        snapshots caught a mid-transition frozen frame). Mixed inputs cover both
        on-girder (L/R move x) and on-ladder (U moves y). Returns (ok, state, pix).
        NOTE: curriculum loads use _is_live instead (this probe walks Mario
        into traffic — probe-suicide at floor cells); this remains for the
        bottom-start controllability check and offline tooling."""
        s0, pix = self._exchange(self.A_NOOP)
        x0, y0 = s0["mario_x"], s0["mario_y"]
        st = s0
        for a in (0b00100, 0b00100, 0b00010, 0b00001, 0b00100):  # up,up,right,left,up
            st, pix = self._exchange(a)
            if not st["is_dead"]:
                # 0x6200 is 1=ALIVE, 0=dead/inactive (polarity is inverted
                # vs the field name). A death tumble moves x/y without input,
                # so without this check a dying Mario "passes" the probe.
                return False, st, pix
            if st["mario_x"] != x0 or st["mario_y"] != y0:
                return True, st, pix
        return False, st, pix

    def _recover(self):
        """A MAME instance died (e.g. crash / display hiccup) -> tear it down and
        relaunch a fresh one with a new save-state. Keeps a long unattended run
        alive: one crash costs one episode, not the whole training run."""
        try:
            if self._sock:
                self._sock.close()
        except OSError:
            pass
        self._sock = None
        if self._proc:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        if getattr(self, "_mame_out", None):
            try:
                self._mame_out.close()
            except OSError:
                pass
            self._mame_out = None
        self._has_state = False
        self._rxbuf = b""
        self._launch_mame()
        self._connect()
        self._read_handshake()
        state, pix = self._start_game()
        if not self.record:
            self._save_state()
        # The recovered episode is a fresh BOTTOM start on a fresh MAME:
        # correct the episode labels (a death mid-curriculum-reset would
        # otherwise keep _start_type="curriculum"/_bw_start and poison
        # per-cell audits) and re-apply the per-episode barrel mode (the
        # fresh MAME runs bridge-default live barrels regardless of what
        # self._no_barrels said — resample and send the command so label
        # and reality agree). External review finding, fixed 2026-07-10.
        self._start_type = "bottomup"
        self._bw_start = None
        self._no_barrels = self.np_random.random() < self.P_NO_BARRELS
        state, pix = self._exchange(
            self.A_FREEZE_BARRELS if self._no_barrels
            else self.A_UNFREEZE_BARRELS)
        return state, pix

    def _begin_episode(self, state):
        self._prev = state
        # Per-episode progress trackers (BASE_Y is the start-row mario_y).
        self._min_y = state["mario_y"] if state["mario_y"] else self.BASE_Y
        self._max_screen = state["screen_id"]
        # Ground truth of where the episode really began, for the per-episode
        # monitor CSVs: a "bottomup" row with start_y far above BASE_Y, or a
        # cleared row whose start_screen != 1, is a mislabeled/phantom episode.
        self._start_y = state["mario_y"]
        self._start_screen = state["screen_id"]
        self._difficulty_start = state.get("difficulty", 0)
        # Height already paid by the milestone reward (start height -> no pay).
        self._reward_max_h = max(0, self.BASE_Y - self._min_y)
        self._visited = set()                # novelty bonus: cells seen this episode
        self._wp_hit = set()                 # waypoint milestones fired this episode
        self._episode_steps = 0              # for height-timeout termination
        self._glitch_px = 0                  # cumulative off-ladder climb pixels
        self._glitch_kill = False            # episode ended by the glitch guard
        self._burnin_left = 0                # LSTM spawn burn-in (set in reset)
        self._burnin_drawn = 0               # this episode's drawn burn-in length
        self._forced_actions = []            # approach replay queue (see reset)
        self._approach_len = 0               # this episode's forced-approach length
        # Success recording (run 29): executed-action log + start descriptor.
        # A successful curriculum episode is fully reproducible offline
        # (fixed .sta + recorded acts, no jitter on curriculum loads), which
        # makes every rare win harvestable: SIL replay data, new curriculum
        # rungs, and approach bytes — see harvest_successes.py.
        self._ep_acts: list[int] = []        # executed-action log
        # NB: _ep_start_sta is deliberately NOT cleared here — it is set by
        # _load_backward_start, which runs BEFORE _begin_episode (the §12
        # ordering trap, FOURTH occurrence: this very line used to zero it
        # every episode, silently marking all success records inexact and
        # starving the harvester). reset() clears it at the top instead.
        self._last_exec = 0                  # action actually executed this step

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._start_type = "bottomup"
        self._bw_start = None
        # Reset here, NOT in _begin_episode: these are set during
        # _load_backward_start, which runs before _begin_episode (the
        # per-episode-state ordering trap, §12 of HANDOFF — third time).
        self._bw_fallback_chain = -1
        self._ep_start_sta = None
        try:
            if self._proc is None:           # first reset: launch persistent MAME
                self._launch_mame()
                self._connect()
                self._read_handshake()
                state, pix = self._start_game()   # full ~19s intro, once
                if not self.record:
                    # Snapshot the controllable start so later resets just LOAD
                    # it (~0.04s vs ~1.5s). Disabled when recording (a state load
                    # isn't an input event and would break .inp playback).
                    self._save_state()
                # Apply the per-episode barrel mode here too — otherwise each
                # worker's FIRST episode always runs bridge-default (live)
                # barrels regardless of P_NO_BARRELS.
                self._no_barrels = self.np_random.random() < self.P_NO_BARRELS
                state, pix = self._exchange(
                    self.A_FREEZE_BARRELS if self._no_barrels
                    else self.A_UNFREEZE_BARRELS)
            elif self._has_state:            # fast path: instant state reload
                if self._bw_chains is not None:
                    # Backward-algorithm curriculum: with prob p_curric start
                    # from a winner-chain state (walk-back window set by the
                    # trainer via set_backward_level); else bottom start.
                    state = pix = None
                    if self.np_random.random() < self._p_curric:
                        state, pix = self._load_backward_start()
                        if state is not None:
                            self._start_type = "curriculum"
                    if state is None:
                        state, pix = self._load_state()
                else:
                    # Legacy self-curriculum: with prob p_curric, start partway
                    # up from an expert-demo state. Bad (frozen) snapshots are
                    # detected and fall back to the bottom start. Wall-zone:
                    # restrict to the lowest 5 states (heights 35-52), which
                    # start Mario mid-traverse with live barrels.
                    n_wall = min(5, self._n_curric)
                    if n_wall and self.np_random.random() < self._p_curric:
                        idx = int(self.np_random.integers(n_wall))
                        self._load_curriculum(idx)
                        ok, state, pix = self._is_responsive()
                        if not ok:
                            state, pix = self._load_state()
                        else:
                            self._start_type = "curriculum"
                    else:
                        state, pix = self._load_state()
                # Per-episode barrel-freeze: 50% of episodes disable all barrels
                # and the fireball so the agent can freely explore the left traverse
                # and ladder-climb without danger. The remaining 50% run live.
                self._no_barrels = self.np_random.random() < self.P_NO_BARRELS
                mode_cmd = (self.A_FREEZE_BARRELS if self._no_barrels
                            else self.A_UNFREEZE_BARRELS)
                state, pix = self._exchange(mode_cmd)
                # The save-state restores a FIXED barrel/RNG state; advance a
                # random number of idle steps so each episode's barrel pattern
                # differs (generalization across DK's RNG, not one fixed
                # start). UNITS: exchanges (frameskip frames each) — at
                # frameskip 4, 0-20 exchanges = 0-80 frames ≈ up to 1.3s.
                # BOTTOM STARTS ONLY: idling at the spawn is safe (no barrel
                # reaches it for ~5s). For curriculum cells it was a death
                # sentence — winner-chain states on the top girder sit in the
                # barrel-spawn lane, and holding NOOP there killed Mario
                # DURING reset (game respawns him at the bottom on a stored
                # life -> ghost episode: max_height frozen at start height,
                # ~1% frontier clears, walk-back stalled). Curriculum draws
                # get their diversity from action sampling + 12 chains.
                if self._start_type != "curriculum":
                    n = int(self.np_random.integers(0, 21))
                    if n:
                        state, pix = self._hold(self.A_NOOP, n)
            else:                            # recording path: clean soft-reset+intro
                self._exchange(self.A_RESET)
                state, pix = self._start_game()
                # Still need to set barrel mode for recording sessions.
                self._no_barrels = self.np_random.random() < self.P_NO_BARRELS
                mode_cmd = (self.A_FREEZE_BARRELS if self._no_barrels
                            else self.A_UNFREEZE_BARRELS)
                state, pix = self._exchange(mode_cmd)
        except (ConnectionError, OSError):   # MAME died -> relaunch fresh
            state, pix = self._recover()
        self._begin_episode(state)
        # LSTM spawn burn-in: a state-load drops the policy mid-traffic with a
        # zeroed LSTM — it must act before it has read any barrel velocities.
        # Establishment-wall signature across runs 27*: spawn, dodge-in-place
        # ~9s, die with zero height gain. For the first BURN_IN steps the env
        # executes NOOP regardless of the chosen action, so the LSTM fills
        # with real observations before decisions count. ONLY below the
        # mastered band: top-girder cells (h>=172) sit in the barrel-spawn
        # lane where standing still kills (see the reset-jitter comment
        # above), and they clear 90-100% without help. Trade-off: the ~8
        # burn-in transitions store the CHOSEN action but executed NOOP —
        # slight off-policy mislabeling, accepted at ~2% of episode steps.
        # STOCHASTIC 50/50 (run 28c, user film review + beeline probe,
        # 2026-07-10): a FIXED 8-step freeze phase-shifts the whole run into
        # the cell's deterministic traffic pattern — 2 of 3 stuck frontiers
        # were PROVEN clearable by immediate play and doomed by the delay
        # (delayed-beeline deaths matched the policy's film frame-for-frame).
        # Half the spawns now get instant control (delay-critical cells cap
        # ~0.35-0.45, above the 0.31 gate); half keep the LSTM warm-up.
        # Bonus: restores traffic-phase diversity to fixed-RNG cells.
        if self._pending_approach is not None:
            # Approach replay supersedes burn-in for this episode; only
            # applies to genuine curriculum starts (a mid-reset _recover
            # falls back to bottomup and must not inherit the queue).
            if self._start_type == "curriculum":
                self._forced_actions = self._pending_approach
                self._approach_len = len(self._forced_actions)
            self._pending_approach = None
        elif (self._start_type == "curriculum"
                and state["mario_y"] > self.BASE_Y - 172
                and self.np_random.random() < 0.5):
            self._burnin_left = self.BURN_IN_STEPS
            self._burnin_drawn = self.BURN_IN_STEPS
        return self._preprocess(pix, state), self._info(state)

    BASE_Y = 240   # Mario's start-row y; height climbed = BASE_Y - min_y reached

    def _info(self, state: dict) -> dict:
        # max_height = pixels climbed above the start row (higher = better);
        # cleared = reached a later screen this episode.
        return {"state": state,
                "max_height": max(0, self.BASE_Y - self._min_y),
                "cleared": int(self._max_screen > 1),
                "start_type": self._start_type,
                "bw_start": self._bw_start,   # (chain, pos, len, height)|None
                "no_barrels": self._no_barrels,
                "start_y": self._start_y,
                "start_screen": self._start_screen,
                "end_screen": state["screen_id"],
                "bw_pos": self._bw_start[1] if self._bw_start else -1,
                # Chain id alongside pos: (bw_pos, start_y) audits get
                # contaminated by cross-chain position collisions + height
                # label lag — this column makes per-cell attribution exact.
                "bw_chain": self._bw_start[0] if self._bw_start else -1,
                "glitch_kill": int(self._glitch_kill),
                # Drawn burn-in length (0 or BURN_IN_STEPS): per-phase clear
                # attribution for the stochastic spawn burn-in.
                "burnin": self._burnin_drawn,
                # Forced-approach length (0 = no approach replay): per-cell
                # attribution for approach-replay vs burn-in episodes.
                "approach_len": self._approach_len,
                # Chain whose curriculum load fell back to a bottom start
                # this episode (-1 = no fallback): per-chain flakiness rates.
                "bw_fallback_chain": self._bw_fallback_chain,
                # Action actually EXECUTED this step (forced-approach/burn-in
                # may override the agent's choice) — SIL must imitate this.
                "exec_action": self._last_exec,
                # Internal difficulty (1-5, tracked-only): curriculum states
                # inherit game time from their phase-1 trajectories, so deep
                # cells start at HIGHER difficulty than the bottom start —
                # these columns expose that confound per episode.
                "difficulty_start": self._difficulty_start,
                "difficulty_end": state.get("difficulty", 0)}

    BURN_IN_STEPS = 8   # ~0.53s at frameskip 4; long enough to read velocities
    HANDOVER_JITTER = 6  # approach replay: drop a random 0..J action suffix

    def step(self, action: int):
        if self._forced_actions:             # approach replay (see
            action = self._forced_actions.pop(0)   # _load_backward_start)
        elif self._burnin_left > 0:          # LSTM spawn burn-in (see reset)
            self._burnin_left -= 1
            action = 0                       # ACTIONS[0] = noop
        self._last_exec = int(action)
        self._ep_acts.append(int(action))
        try:
            self._sock.sendall(bytes([ACTIONS[int(action)]]))
            ram, pix = self._read_obs()
        except (ConnectionError, OSError):
            # MAME died mid-episode (crash / timeout): relaunch and end this
            # episode cleanly so training continues on a fresh instance.
            # The terminal info must describe the CRASHED episode (last good
            # state + its trackers), not the fresh post-recovery start —
            # capture it BEFORE _recover()/_begin_episode reset the labels,
            # or the Monitor CSV logs a phantom row (external review, 2026-07-10).
            crash_info = self._info(self._prev)
            state, pix = self._recover()
            self._begin_episode(state)
            return self._preprocess(pix, state), 0.0, True, False, crash_info
        state = self._decode_state(ram)
        reward, terminated = self._reward(state)
        if state["mario_y"] and not state.get("is_jumping", 0):
            self._min_y = min(self._min_y, state["mario_y"])
        self._max_screen = max(self._max_screen, state["screen_id"])
        obs = self._preprocess(pix, state)   # uses self._prev (old) for vx/vy/fall-zone
        self._prev = state
        if terminated:
            self._maybe_record_success(state)
        return obs, reward, terminated, False, self._info(state)

    def _maybe_record_success(self, state):
        """Append a reproducible success record (start .sta + executed acts)
        to logs/successes/dk_<port>.jsonl. Curriculum successes replay
        deterministically (fixed-RNG .sta, no jitter, forced prefixes are in
        the act log); bottom starts are flagged approximate (reset jitter is
        not in the log). Harvested offline by harvest_successes.py into SIL
        food, new rungs, and approach bytes."""
        if self._no_barrels or self._glitch_kill or len(self._ep_acts) > 1500:
            return
        cleared = int(self._max_screen > 1)
        gain = max(0, self.BASE_Y - self._min_y) - max(
            0, self.BASE_Y - (self._start_y or self.BASE_Y))
        curric = self._start_type == "curriculum"
        if not (cleared or (curric and gain >= 40)):
            return
        rec = {"ts": time.time(), "port": self.port,
               "start": (os.path.basename(self._ep_start_sta)
                         if self._ep_start_sta else "bottom"),
               "exact": bool(self._ep_start_sta),
               "acts": list(self._ep_acts), "cleared": cleared, "gain": gain,
               "bw": list(self._bw_start[:2]) if self._bw_start else None}
        try:
            os.makedirs("logs/successes", exist_ok=True)
            with open(f"logs/successes/dk_{self.port}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass                       # recording must never kill training

    def close(self):
        if self._sock:
            # Ask MAME to exit cleanly so the -record .inp is flushed/finalized;
            # only then close the socket.
            try:
                self._sock.sendall(bytes([self.A_QUIT]))
                self._sock.settimeout(3.0)
                try:
                    while self._sock.recv(4096):
                        pass            # drain until MAME closes the connection
                except OSError:
                    pass
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc:
            try:
                self._proc.wait(timeout=8)   # let the clean exit complete
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
        if getattr(self, "_mame_out", None):
            try:
                self._mame_out.close()
            except OSError:
                pass
            self._mame_out = None
