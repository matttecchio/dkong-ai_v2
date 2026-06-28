"""Extract a behavioral-cloning dataset from the expert .inp.

Replays the demo through the bridge in EXTRACT mode: each step yields the 84x84
obs + the play-input bitmask the playback is applying. We keep the FIRST barrel
board (screen_id==1) and map each bitmask to our discrete action index, then save
(frames, actions) for BC training. Frame-stacking is done later in train_bc.

    python -m dkong_ai.extract_bc --rom-dir ./roms --inp dkong.inp
"""
import argparse
import os
import socket
import struct
import subprocess
import time

import cv2
import numpy as np

from .mame_env import _die_with_parent, ACTIONS
from . import memory_map

PORT = 5200
N_WATCH = len(memory_map.WATCH_ORDER)          # ram bytes before the input byte
SCREEN_IDX = memory_map.WATCH_ORDER.index("screen_id")


def mask_to_action(mask: int) -> int:
    """Nearest ACTIONS entry by Hamming distance (expert combos -> our 8 actions)."""
    best, bestd = 0, 99
    for i, a in enumerate(ACTIONS):
        d = bin(mask ^ a).count("1")
        if d < bestd:
            best, bestd = i, d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--inp", default="dkong.inp")
    ap.add_argument("--out", default="artifacts/bc_data.npz")
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    bridge = os.path.abspath(os.path.join(here, "..", "scripts", "bridge.lua"))
    demos = os.path.abspath(os.path.join(here, "..", "demos"))
    env = dict(os.environ, DK_BRIDGE_PORT=str(PORT), DK_EXTRACT="1",
               DK_FRAMESKIP="4", SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    env.pop("DISPLAY", None); env.pop("WAYLAND_DISPLAY", None)
    args_mame = ["mame", "dkong", "-rompath", os.path.abspath(args.rom_dir),
                 "-input_directory", demos, "-playback", args.inp,
                 "-autoboot_script", bridge, "-autoboot_delay", "0",
                 "-skip_gameinfo", "-nothrottle", "-video", "none", "-sound", "none",
                 "-exit_after_playback"]
    proc = subprocess.Popen(args_mame, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, preexec_fn=_die_with_parent)

    # connect as client
    sock = None
    for _ in range(150):
        try:
            sock = socket.create_connection(("127.0.0.1", PORT), timeout=2)
            break
        except OSError:
            time.sleep(0.2)
    if sock is None:
        raise TimeoutError("could not connect to extraction bridge")
    sock.settimeout(30.0); sock.sendall(b"H")
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
            buf += sock.recv(max(n - len(buf), 4096))
        out, buf = buf[:n], buf[n:]
        return out

    frames, actions = [], []
    seen_board = False
    while True:
        sock.sendall(b"\x00")                       # advance ack
        try:
            (length,) = struct.unpack(">I", recvn(4))
            payload = recvn(length)
        except (OSError, struct.error):
            break                                    # playback ended
        (ram_len,) = struct.unpack(">H", payload[:2])
        ram = payload[2:2 + ram_len]
        pix = payload[2 + ram_len:]
        screen = ram[SCREEN_IDX]
        mask = ram[N_WATCH]                          # appended input byte
        if screen == 1:
            seen_board = True
            arr = np.frombuffer(pix, np.uint8).reshape(H, W, 4)
            gray = cv2.cvtColor(arr[..., 1:4], cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            frames.append(small)
            actions.append(mask_to_action(mask))
        elif seen_board:
            break                                    # first barrel board done

    proc.terminate()
    frames = np.array(frames, np.uint8)
    actions = np.array(actions, np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, frames=frames, actions=actions)
    import collections
    dist = collections.Counter(actions.tolist())
    print(f"saved {len(frames)} (obs,action) pairs -> {args.out}")
    print("action distribution (idx:count):", dict(sorted(dist.items())))
    print("ACTIONS legend:", {i: bin(a) for i, a in enumerate(ACTIONS)})


if __name__ == "__main__":
    main()
