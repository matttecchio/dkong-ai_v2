"""Augment a backward-curriculum manifest with per-cell APPROACH data.

For each cell whose .sta name encodes archive provenance (a{ai}_c{IDX} or
a{ai}_c{IDX}_d{J}), derive the proven play sequence that arrives at it:
anchor = the archive parent's state, acts = the (sanitized) play bytes of the
segment into c{IDX} (truncated at J for _dJ rungs). Every candidate approach
is REPLAY-VERIFIED in a scratch MAME against the cell's actual loaded
position — wrong provenance (multi-generation rungs, desynced archives)
self-excludes and the cell simply keeps the burn-in path.

Approach-replay motivation (film reviews #3/#4 + battery, 2026-07-10/11):
different stuck cells want different halves of the warm-context/clean-phase
trade; forced replay of the real approach provides both, at the cell's
original game time (zero bonus-timer cost), with handover randomization as
safe jitter along a proven path.

    python -m dkong_ai.augment_approaches --src artifacts/backward_dense12 \
        --out artifacts/backward_dense13 \
        --archive-a0 artifacts/go_explore_run3 --archive-a1 artifacts/go_explore
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil

from .go_explore import Archive
from .mame_env import DonkeyKongEnv, ACTIONS

NAME_RE = re.compile(r"^a(\d+)_c(\d+)(?:_d(\d+))?\.sta$")
APPROACH_MAX = 14      # env feeds at most this many trailing acts
TOL_X, TOL_Y = 20, 10  # landing tolerance vs the cell's loaded position


def sanitize(byts):
    """Raw bridge bytes -> (play_prefix_raw, act_indices). Stops at the first
    control byte (trailing snapshot overhead: A_SAVE + NOOPs)."""
    raw, acts = [], []
    for b in byts:
        if b > 0x1F:
            break
        raw.append(b)
        if b in ACTIONS:
            acts.append(ACTIONS.index(b))
        else:  # nearest by Hamming distance (extract_bc pattern)
            acts.append(min(range(len(ACTIONS)),
                            key=lambda i: bin(ACTIONS[i] ^ b).count("1")))
    return raw, acts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--archive-a0", default="artifacts/go_explore_run3")
    ap.add_argument("--archive-a1", default="artifacts/go_explore")
    ap.add_argument("--rom-dir", default="./roms")
    ap.add_argument("--port", type=int, default=5100)
    args = ap.parse_args()

    archives = {}
    for ai, adir in ((0, args.archive_a0), (1, args.archive_a1)):
        arch = Archive(adir)
        if arch.load():
            archives[ai] = arch
        else:
            print(f"[augment] archive a{ai} ({adir}) failed to load — "
                  f"its cells keep burn-in")

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.src):
        shutil.copyfile(os.path.join(args.src, f), os.path.join(args.out, f))
    mani = json.load(open(os.path.join(args.out, "manifest.json")))

    env = DonkeyKongEnv(args.rom_dir, port=args.port, record=False)
    env._p_curric = 0.0
    env.P_NO_BARRELS = 0.0
    env.reset()

    done_cells: dict[str, dict | None] = {}   # sta name -> approach | None
    n_ok = n_skip = n_fail = 0
    for ch in mani["chains"]:
        for cell in ch["cells"]:
            name = cell["sta"]
            if name in done_cells:
                if done_cells[name]:
                    cell["approach"] = done_cells[name]
                continue
            done_cells[name] = None
            m = NAME_RE.match(name)
            if not m:
                n_skip += 1              # wc_*/wcf_* — no bytes until mined
                continue
            ai, idx, j = int(m.group(1)), int(m.group(2)), m.group(3)
            arch = archives.get(ai)
            rec = arch.by_idx.get(idx) if arch else None
            if rec is None or rec.get("parent") is None:
                n_skip += 1
                continue
            byts = list(rec["bytes"])
            if j is not None:
                byts = byts[:int(j) + 1]
            raw, acts = sanitize(byts)
            if len(acts) < 4:
                n_skip += 1
                continue
            # Replay 1 — VERIFY: cell's actual loaded position vs landing.
            st_cell, _ = env.load_state_file(os.path.join(args.out, name))
            cx, cy = st_cell["mario_x"], st_cell["mario_y"]
            anchor_src = arch.sta_path(rec["parent"])
            leg_anchor = os.path.join(args.out, f"ap_a{ai}_leg{idx}.sta")
            shutil.copyfile(anchor_src, leg_anchor)
            env.load_state_file(leg_anchor)
            st = None
            for b in raw:
                st, _ = env._exchange(b)
            if (st is None or not st["mario_y"]
                    or abs(st["mario_x"] - cx) > TOL_X
                    or abs(st["mario_y"] - cy) > TOL_Y):
                lx = st["mario_x"] if st else "?"
                ly = st["mario_y"] if st else "?"
                print(f"[augment] VERIFY FAIL {name}: replay lands "
                      f"({lx},{ly}) vs cell ({cx},{cy}) — burn-in kept")
                n_fail += 1
                os.unlink(leg_anchor)
                continue
            # Direction/motion filters (run 28g, learned the hard way):
            # a DESCENDING approach hands over with downward momentum and
            # misleading context; a STATIONARY one is a pure delay — the
            # phase-doom the beeline probe convicted (c446_d5: approach 1%
            # vs clean-spawn 67% over 228 draws). Both are archive truth
            # (the explorer wandered/idled) but terrible pedagogy.
            st0, _ = env.load_state_file(anchor_src)
            if st0["mario_y"] and st0["mario_y"] < cy - 4:
                print(f"[augment] DROP descending approach {name}")
                n_fail += 1
                os.unlink(leg_anchor)
                continue
            if (abs(st0["mario_x"] - cx) + abs(st0["mario_y"] - cy)) < 8:
                print(f"[augment] DROP stationary approach {name}")
                n_fail += 1
                os.unlink(leg_anchor)
                continue
            # Replay 2 — MINT a mid-leg anchor so training feeds at most
            # APPROACH_MAX forced steps. (Fresh replay: a mid-run A_SAVE
            # costs ~7 extra exchanges and would desync replay 1's landing.)
            cut = max(0, len(raw) - APPROACH_MAX)
            anchor, feed = None, acts
            if cut == 0:
                anchor = f"ap_{name}"           # short leg: full-leg anchor
                shutil.copyfile(anchor_src, os.path.join(args.out, anchor))
            else:
                env.load_state_file(leg_anchor)
                stc = None
                for b in raw[:cut]:
                    stc, _ = env._exchange(b)
                # Only snapshot a clean frame (alive, not mid-jump); an
                # unclean cut falls back to the full-leg anchor + full acts.
                if stc and stc["is_dead"] == 1 and not stc["is_jumping"]:
                    for _ in range(4):
                        env._exchange(env.A_SAVE)
                    env._hold(env.A_NOOP, 3)
                    anchor = f"ap_{name}"
                    shutil.copyfile(env._slot_sta_path(),
                                    os.path.join(args.out, anchor))
                    feed = acts[cut:]
                else:
                    anchor = f"ap_full_{name}"
                    shutil.copyfile(anchor_src, os.path.join(args.out, anchor))
            os.unlink(leg_anchor)
            approach = {"anchor": anchor, "acts": feed}
            cell["approach"] = approach
            done_cells[name] = approach
            n_ok += 1
    env.close()

    json.dump(mani, open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1)
    print(f"[augment] approaches: {n_ok} verified, {n_fail} failed "
          f"verification, {n_skip} no-provenance -> {args.out}")


if __name__ == "__main__":
    main()
