"""Seed the persistent SIL buffer from the professional .inp demo.

The pro recording (demos/dkong.inp, 70+ barrel boards) is replayed through
the bridge's EXTRACT mode; every frameskip'd step is rebuilt into the
CURRENT training observation (84x84x2 image incl. threat channel, 102 RAM
features, 2-frame stack — byte-identical to what SILCallback captures from
obs_tensor), and the pro's held inputs are mapped onto our 8 discrete
actions. Barrel-board segments that END in a board clear (screen 1 ->
other) are kept, earliest boards first (closest to our L1 difficulty
regime), and written into the `clear_bottomup` class of the persistent SIL
buffer — the slot that has never held a real episode (no honest bottom-up
clear exists yet). SIL replays them gently (coef 0.05) alongside the
policy's own successes; the class's FIFO cap means his own clears evict
the pro's the moment they exist.

Run-5 BC failed by INITIALIZING the net from this demo (brittle); this is
replay-alongside, not initialization.

    python -m dkong_ai.seed_sil_from_inp --rom-dir ./roms \
        --sil artifacts/ppo_dkong_run31_sil.pkl [--max-eps 13]
"""
import argparse
import os
import pickle
import socket
import struct
import subprocess
import time
from collections import deque

import numpy as np

from . import memory_map
from .extract_bc import mask_to_action
from .mame_env import DonkeyKongEnv, _die_with_parent

PORT = 5250
N_WATCH = len(memory_map.WATCH_ORDER)
MAX_EP_LEN = 600          # must match SILCallback.max_ep_len


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--inp", default="dkong.inp")
    ap.add_argument("--sil", default="artifacts/ppo_dkong_run31_sil.pkl")
    ap.add_argument("--max-eps", type=int, default=13)
    ap.add_argument("--stack", type=int, default=2)
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    bridge = os.path.abspath(os.path.join(here, "..", "scripts", "bridge.lua"))
    demos = os.path.abspath(os.path.join(here, "..", "demos"))
    env_vars = dict(os.environ, DK_BRIDGE_PORT=str(PORT), DK_EXTRACT="1",
                    DK_FRAMESKIP="4", SDL_VIDEODRIVER="dummy",
                    SDL_AUDIODRIVER="dummy")
    env_vars.pop("DISPLAY", None); env_vars.pop("WAYLAND_DISPLAY", None)
    proc = subprocess.Popen(
        ["mame", "dkong", "-rompath", os.path.abspath(args.rom_dir),
         "-input_directory", demos, "-playback", args.inp,
         "-autoboot_script", bridge, "-autoboot_delay", "0",
         "-skip_gameinfo", "-nothrottle", "-video", "none", "-sound", "none",
         "-exit_after_playback"],
        env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=_die_with_parent)

    sock = None
    for _ in range(150):
        try:
            sock = socket.create_connection(("127.0.0.1", PORT), timeout=2)
            break
        except OSError:
            time.sleep(0.2)
    if sock is None:
        raise TimeoutError("could not connect to extraction bridge")
    sock.settimeout(30.0)
    sock.sendall(b"H")
    rx = b""
    while rx.count(b"\n") < 2:
        rx += sock.recv(4096)
    hello, _, rest = rx.split(b"\n", 2)
    kv = dict(t.split(b"=", 1) for t in hello.split()[1:])
    W, H = int(kv[b"W"]), int(kv[b"H"])
    buf = rest

    def recvn(n):
        nonlocal buf
        while len(buf) < n:
            chunk = sock.recv(max(n - len(buf), 4096))
            if not chunk:
                raise OSError("stream ended")
            buf += chunk
        out, buf2 = buf[:n], buf[n:]
        buf = buf2
        return out

    # Obs factory: a real env instance, never launched/connected — we drive
    # its decode/preprocess machinery directly on the extract stream.
    oe = DonkeyKongEnv(args.rom_dir, port=PORT + 1, record=False)
    oe._geom = {"w": W, "h": H, "bpp": 4, "frameskip": 4, "fields": []}

    episodes = []            # kept clear episodes: list[list[(obs, act)]]
    floor_eps = []           # head segments of long boards (floor play)
    cur = None               # current segment accumulator
    stackbuf = None
    prev_screen = None
    boards_seen = 0
    steps = 0

    def close_segment(cleared):
        nonlocal cur
        if cur and cleared and len(cur) > 4:
            episodes.append(cur[-MAX_EP_LEN:])
            if len(cur) > MAX_EP_LEN:
                # the tail keeps the finishing climb; the HEAD is the floor
                # play + first climbs — the exact section he's stuck on —
                # and it feeds the (empty) floor class instead of the bin
                floor_eps.append(cur[:MAX_EP_LEN])
            print(f"[seed] board {boards_seen}: kept clear episode "
                  f"({len(cur)} steps{', trimmed' if len(cur) > MAX_EP_LEN else ''})")
        cur = None

    while len(episodes) < args.max_eps:
        try:
            sock.sendall(b"\x00")
            (length,) = struct.unpack(">I", recvn(4))
            payload = recvn(length)
        except (OSError, struct.error):
            break
        (ram_len,) = struct.unpack(">H", payload[:2])
        ram = payload[2:2 + ram_len]
        pix = payload[2 + ram_len:]
        state = oe._decode_state(ram)
        mask = ram[N_WATCH]
        screen = state["screen_id"]
        steps += 1

        if screen == 1 and prev_screen != 1:
            boards_seen += 1
            cur = []
            stackbuf = deque([np.zeros((84, 84, 2), np.uint8)] * args.stack,
                             maxlen=args.stack)
            oe._begin_episode(state)
        elif screen != 1 and prev_screen == 1:
            close_segment(cleared=True)      # 1 -> elsewhere = board cleared
        prev_screen = screen

        if cur is None or screen != 1:
            continue
        # A death mid-board restarts the same screen: detect via life loss
        # and drop the segment (we only seed CLEAN clears).
        if oe._prev and state.get("lives", 3) < oe._prev.get("lives", 3):
            cur = None
            continue
        obs = oe._preprocess(pix, state)     # {"image":(84,84,2), "ram":(102,)}
        oe._prev = state                     # same ordering as env.step()
        stackbuf.append(obs["image"])
        stacked = np.concatenate(list(stackbuf), axis=-1)   # (84,84,2*stack)
        # SB3 wraps the venv in VecTransposeImage: SILCallback captures
        # obs_tensor images CHANNEL-FIRST. Match it exactly (verified
        # against training-captured episodes: (4,84,84) uint8).
        stacked = np.ascontiguousarray(stacked.transpose(2, 0, 1))
        cur.append(({"image": stacked, "ram": obs["ram"].copy()},
                    mask_to_action(mask)))

    proc.terminate()
    print(f"[seed] stream done: {steps} steps, {boards_seen} boards seen, "
          f"{len(episodes)} clear episodes kept")
    if not episodes:
        raise SystemExit("[seed] nothing to seed")

    d = {"ram_dim": DonkeyKongEnv.RAM_FEATURE_DIM,
         "bufs": {"floor": [], "clear_bottomup": [], "clear": []}}
    if os.path.exists(args.sil):
        try:
            with open(args.sil, "rb") as fh:
                old = pickle.load(fh)
            if old.get("ram_dim") == d["ram_dim"]:
                d["bufs"].update(old.get("bufs", {}))
        except Exception as e:
            print(f"[seed] existing buffer unreadable ({e}) — starting fresh")
    d["bufs"]["clear_bottomup"] = episodes[:args.max_eps]
    if floor_eps:
        d["bufs"]["floor"] = floor_eps[:args.max_eps]
    tmp = args.sil + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(d, fh, protocol=5)
    os.replace(tmp, args.sil)
    n = {k: len(v) for k, v in d["bufs"].items()}
    print(f"[seed] wrote {args.sil}: {n}")


if __name__ == "__main__":
    main()
