"""Mine APPROACH bytes for wc_* curriculum cells from the world-class .inp.

The WC cells were snapshotted every 45 frames of pro playback but carry no
trajectory bytes — so they never got approach replay, and the floor chains'
states spawn Mario cold where the pro was MID-STRIDE. This tool replays the
.inp through the bridge's EXTRACT mode at the training frameskip, logging
(x, y, action) per step on the first barrel board, aligns each wc cell to
the step stream by position (sequential search — wc numbering is frame
order), and writes verified approaches: anchor = the previous wc state,
acts = the pro's actual inputs between the two states.

    python -m dkong_ai.extract_wc_approaches --rom-dir ./roms \
        --manifest-dir artifacts/backward_dense13
"""
import argparse
import json
import os
import socket
import struct
import subprocess
import time

from .mame_env import DonkeyKongEnv, _die_with_parent, ACTIONS
from . import memory_map

PORT = 5200
N_WATCH = len(memory_map.WATCH_ORDER)
IDX_SCREEN = memory_map.WATCH_ORDER.index("screen_id")
IDX_X = memory_map.WATCH_ORDER.index("mario_x")
IDX_Y = memory_map.WATCH_ORDER.index("mario_y")
IDX_START = memory_map.WATCH_ORDER.index("game_start")

MAX_SEG = 20        # skip curation gaps longer than this many steps
MATCH_TOL = 10      # |dx| + 2|dy| position-match tolerance
APPROACH_MAX = 14


def mask_to_action(mask: int) -> int:
    return min(range(len(ACTIONS)), key=lambda i: bin(ACTIONS[i] ^ mask).count("1"))


def extract_steps(rom_dir: str, inp: str):
    """Replay the .inp in EXTRACT mode; yield (x, y, act) per 4-frame step
    for the first barrel board."""
    here = os.path.dirname(__file__)
    bridge = os.path.abspath(os.path.join(here, "..", "scripts", "bridge.lua"))
    demos = os.path.abspath(os.path.join(here, "..", "demos"))
    env = dict(os.environ, DK_BRIDGE_PORT=str(PORT), DK_EXTRACT="1",
               DK_FRAMESKIP="4", SDL_VIDEODRIVER="dummy",
               SDL_AUDIODRIVER="dummy")
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    cmd = ["mame", "dkong", "-rompath", os.path.abspath(rom_dir),
           "-input_directory", demos, "-playback", inp,
           "-autoboot_script", bridge, "-autoboot_delay", "0",
           "-skip_gameinfo", "-nothrottle", "-video", "none", "-sound",
           "none", "-exit_after_playback"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            preexec_fn=_die_with_parent)
    sock = None
    for _ in range(150):
        try:
            sock = socket.create_connection(("127.0.0.1", PORT), timeout=2)
            break
        except OSError:
            time.sleep(0.2)
    if sock is None:
        proc.kill()
        raise TimeoutError("could not connect to extraction bridge")
    sock.settimeout(30.0)
    sock.sendall(b"H")
    rx = b""
    while rx.count(b"\n") < 2:
        rx += sock.recv(4096)
    _, _, buf = rx.split(b"\n", 2)

    def recvn(n):
        nonlocal buf
        while len(buf) < n:
            buf += sock.recv(max(n - len(buf), 4096))
        out, buf2 = buf[:n], buf[n:]
        buf = buf2
        return out

    steps, seen = [], False
    while True:
        sock.sendall(b"\x00")
        try:
            (length,) = struct.unpack(">I", recvn(4))
            payload = recvn(length)
        except (OSError, struct.error):
            break
        (ram_len,) = struct.unpack(">H", payload[:2])
        ram = payload[2:2 + ram_len]
        # game_start gates out the ATTRACT-MODE demo, which also shows
        # screen==1 with a walking Mario (first extraction run matched
        # nothing: it had captured the attract loop, not the game).
        if ram[IDX_SCREEN] == 1 and ram[IDX_START] == 1:
            seen = True
            steps.append((ram[IDX_X], ram[IDX_Y],
                          mask_to_action(ram[N_WATCH])))
        elif seen:
            break
    proc.terminate()
    # Trim the intro: Kong's climb runs ~285 steps with screen==1 and
    # game_start==1 but Mario OFF-FIELD (x=y=0 sentinel) — leaving it in
    # pushed the first real-play step past the sequential search window and
    # cascaded the whole alignment into 57/57 no-match.
    first = next((i for i, (x, y, _) in enumerate(steps) if y > 0 and x > 0),
                 0)
    return steps[first:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--inp", default="dkong.inp")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--port", type=int, default=5100,
                    help="scratch env for cell measurement + verification")
    args = ap.parse_args()

    steps = extract_steps(args.rom_dir, args.inp)
    print(f"[wc-extract] {len(steps)} steps on the first barrel board")

    d = args.manifest_dir
    mani = json.load(open(os.path.join(d, "manifest.json")))
    wc_names = sorted({c["sta"] for ch in mani["chains"] for c in ch["cells"]
                       if c["sta"].startswith("wc_") and "approach" not in c})
    env = DonkeyKongEnv(args.rom_dir, port=args.port, record=False)
    env._p_curric = 0.0
    env.P_NO_BARRELS = 0.0
    env.reset()

    def pos(sta):
        st, _ = env.load_state_file(os.path.join(d, sta))
        return st["mario_x"], st["mario_y"]

    # Sequential position alignment: wc numbering is frame order and states
    # were mined every 45 frames = ~11.25 steps, so the expected stream
    # position scales with the INDEX GAP between surviving states (curation
    # dropped 92/162 — fixed windows cascaded into no-match after each gap).
    matches: dict[str, int] = {}
    j_prev, k_prev = 0, None
    for name in wc_names:
        k = int(name[3:6])
        cx, cy = pos(name)
        exp = j_prev + (11.25 * (k - k_prev) if k_prev is not None else 11.25)
        lo = max(1, int(exp) - 60)
        hi = min(len(steps), int(exp) + 60)
        best, bestd = None, 999
        for j in range(lo, hi):
            x, y, _ = steps[j]
            dist = abs(x - cx) + 2 * abs(y - cy)
            if dist < bestd:
                best, bestd = j, dist
        if best is None or bestd > MATCH_TOL:
            print(f"[wc-extract] NO MATCH {name} (best dist {bestd} "
                  f"around step {int(exp)})")
            # keep the EXPECTED position so one bad cell doesn't derail
            # the running alignment for everything after it
            j_prev, k_prev = int(exp), k
            continue
        matches[name] = best
        j_prev, k_prev = best, k

    n_ok = n_skip = 0
    approaches: dict[str, dict] = {}
    ordered = [n for n in wc_names if n in matches]
    for k, name in enumerate(ordered):
        j = matches[name]
        if k == 0:
            continue
        anchor = ordered[k - 1]
        ja = matches[anchor]
        seg = j - ja
        if not (3 <= seg <= MAX_SEG):
            n_skip += 1
            continue
        acts = [steps[t][2] for t in range(ja, j)][-APPROACH_MAX:]
        # Verify: replay anchor + acts, compare landing to the cell.
        cx, cy = pos(name)
        st, _ = env.load_state_file(os.path.join(d, anchor))
        for a in acts:
            st, _ = env._exchange(ACTIONS[a])
        if (not st["mario_y"] or abs(st["mario_x"] - cx) > 20
                or abs(st["mario_y"] - cy) > 10):
            print(f"[wc-extract] VERIFY FAIL {name}: lands "
                  f"({st['mario_x']},{st['mario_y']}) vs ({cx},{cy})")
            n_skip += 1
            continue
        ax, ay = pos(anchor)
        if ay < cy - 4 or (abs(ax - cx) + abs(ay - cy)) < 8:
            n_skip += 1        # descending or stationary — bad pedagogy
            continue
        approaches[name] = {"anchor": anchor, "acts": acts}
        n_ok += 1
    env.close()

    for ch in mani["chains"]:
        for c in ch["cells"]:
            if c["sta"] in approaches:
                c["approach"] = approaches[c["sta"]]
    json.dump(mani, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    print(f"[wc-extract] wrote {n_ok} approaches ({n_skip} skipped, "
          f"{len(wc_names) - len(matches)} unmatched) -> {d}/manifest.json")


if __name__ == "__main__":
    main()
