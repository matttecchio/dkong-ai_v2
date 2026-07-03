"""Replay a Go-Explore winner as one continuous, watchable bottom-up run.

Walks the winner's ancestor chain and re-sends every recorded action byte,
re-syncing on each cell's save-state exactly the way exploration did. Each
restore loads the state the machine is already in, so the result looks like
one seamless climb from the bottom girder to the screen change.

Modes (combinable):
  --avi NAME.avi   MAME writes a video of the session to artifacts/recordings/
                   (headless + unthrottled unless --watch: renders in seconds,
                   video plays at normal speed; auto-converts to .mp4 if
                   ffmpeg is installed)
  --watch          windowed at normal speed (WSLg) to watch it live

Example:
  python -m dkong_ai.replay_winner --rom-dir ./roms \
      --archive-dir artifacts/go_explore --winner 0 --avi first_clear.avi
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time

from .go_explore import Archive, restore_file, _repo_dir
from .mame_env import DonkeyKongEnv


def replay(env: DonkeyKongEnv, arch: Archive, w: dict, tail_noops: int = 900):
    chain = arch.ancestors(w["parent"])
    state = None
    for cell in chain[1:]:
        restore_file(env, arch.sta_path(cell["parent"]))
        for b in cell["bytes"]:
            state, _ = env._exchange(b)
    restore_file(env, arch.sta_path(w["parent"]))
    for b in w["bytes"]:
        state, _ = env._exchange(b)
    # Let the clear cutscene (and next stage's start) play out on film.
    for _ in range(tail_noops):
        state, _ = env._exchange(env.A_NOOP)
    return chain, state


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--archive-dir",
                    default=os.path.join(_repo_dir(), "artifacts", "go_explore"))
    ap.add_argument("--winner", type=int, default=0)
    ap.add_argument("--avi", default=None,
                    help="AVI filename (written to artifacts/recordings/)")
    ap.add_argument("--watch", action="store_true",
                    help="windowed, normal speed")
    ap.add_argument("--port", type=int, default=5300)
    args = ap.parse_args()

    arch = Archive(args.archive_dir)
    if not arch.load():
        raise SystemExit(f"no archive.json in {args.archive_dir}")
    if not arch.winners:
        raise SystemExit("archive has no winners yet")
    w = arch.winners[args.winner]

    recdir = os.path.join(_repo_dir(), "artifacts", "recordings")
    os.makedirs(recdir, exist_ok=True)
    extra = []
    if args.avi:
        extra += ["-snapshot_directory", recdir, "-aviwrite", args.avi]
    if args.watch:
        extra += ["-throttle"]
    env = DonkeyKongEnv(args.rom_dir, port=args.port, record=False,
                        headless=not args.watch, extra_mame_args=extra)
    env._p_curric = 0.0
    env.P_NO_BARRELS = 0.0

    try:
        env.reset()                      # boots MAME; intro plays onto the film
        t0 = time.time()
        chain, state = replay(env, arch, w)
        hs = [c["height"] for c in chain]
        print(f"replayed winner #{args.winner}: {len(chain)} cells, "
              f"heights {hs}")
        print(f"final: screen_id={state['screen_id']} lives={state['lives']} "
              f"({time.time() - t0:.0f}s)")
        if state["screen_id"] == 1:
            print("WARNING: screen_id still 1 — replay desynced?")
    finally:
        env.close()                      # clean quit finalizes the AVI

    if args.avi:
        avi = os.path.join(recdir, args.avi)
        if os.path.exists(avi) and shutil.which("ffmpeg"):
            mp4 = os.path.splitext(avi)[0] + ".mp4"
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", avi,
                                "-c:v", "libx264", "-crf", "20",
                                "-pix_fmt", "yuv420p", mp4])
            if r.returncode == 0:
                os.remove(avi)
                print(f"video -> {mp4}")
            else:
                print(f"video -> {avi} (ffmpeg convert failed)")
        else:
            print(f"video -> {avi}")


if __name__ == "__main__":
    main()
