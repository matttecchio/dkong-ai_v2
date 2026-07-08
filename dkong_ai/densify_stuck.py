"""Surgical walk-back densifier: insert fine replay rungs immediately
DOWNSTREAM of stuck frontier cells, doom-screen them, and splice a new
backward dir.

Unlike export_chains --densify (band-wide, fixed stride), this targets named
cells where the walk-back gate has stalled and fills only the gap between the
stuck cell and its already-promoted neighbour with every-step rungs.

Levels are end-relative, so inserting cells between the frontier and its
mastered neighbour keeps a saved levels.json valid: each affected chain's
frontier pointer lands on the easiest new rung, and the stuck cell moves
deeper by the number of rungs inserted.

    python -m dkong_ai.densify_stuck \
        --src artifacts/backward_dense2 --out artifacts/backward_dense3 \
        --archive artifacts/go_explore_run1 --archive artifacts/go_explore \
        --stuck a1_c446.sta --stuck a0_c469.sta --stuck a1_c445_d4.sta
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil

from .go_explore import Archive
from .mame_env import ACTIONS, DonkeyKongEnv

BAND_LO, BAND_HI = 140, 200      # clean-frame height band for new rungs
MAX_RUNGS_PER_LEG = 4

_NAME = re.compile(r"a(\d+)_c(\d+)(?:_d(\d+))?\.sta$")


def parse_name(sta: str):
    m = _NAME.match(sta)
    if not m:
        raise ValueError(f"unparseable state name {sta}")
    ai, idx, j = m.groups()
    return int(ai), int(idx), (int(j) if j is not None else None)


def _height(st) -> int:
    return max(0, 240 - st["mario_y"]) if st["mario_y"] else 0


def replay(env, a_sta_path: str, byts, upto: int):
    """Fresh replay: load leg start, feed byts[:upto+1]; return final state."""
    env.load_state_file(a_sta_path)
    st = None
    for b in byts[: upto + 1]:
        st, _ = env._exchange(b)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="artifacts/backward_dense2")
    ap.add_argument("--out", default="artifacts/backward_dense3")
    ap.add_argument("--archive", action="append", required=True,
                    help="phase-1 archive dirs in the SAME ORDER the src "
                         "manifest was exported with (a0, a1, ...)")
    ap.add_argument("--stuck", action="append", required=True,
                    help="state name (as in manifest) of a stalled frontier")
    ap.add_argument("--rom-dir", default="./roms")
    ap.add_argument("--port", type=int, default=5301)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--doom-steps", type=int, default=12)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.src, "manifest.json")))
    chains = manifest["chains"]
    archives = []
    for d in args.archive:
        arch = Archive(d)
        if not arch.load():
            raise SystemExit(f"[densify] cannot load archive {d}")
        archives.append(arch)

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.src):
        shutil.copyfile(os.path.join(args.src, f), os.path.join(args.out, f))

    env = DonkeyKongEnv(args.rom_dir, port=args.port, record=False)
    env._p_curric = 0.0
    env.P_NO_BARRELS = 0.0
    env.reset()

    minted: dict[tuple, dict | None] = {}   # (ai, leg_idx, j) -> cell | None

    def mint_leg(ai: int, leg_idx: int, a_path: str, j_lo: int, j_hi: int,
                 check_j: int | None, check_sta: str):
        """Mint rungs j in [j_lo, j_hi) along leg -> leg_idx, replayed from
        a_path. Frame-exactness probe: replay to the successor's known point
        and compare against the successor state's ACTUAL loaded height —
        manifest height labels can lag the snapshot by 5-15px (a0 gotcha),
        so the .sta itself is the only trustworthy anchor."""
        rec = archives[ai].by_idx[leg_idx]
        byts = rec["bytes"]
        st = replay(env, a_path, byts,
                    check_j if check_j is not None else len(byts) - 1)
        replay_h = _height(st)
        st_true, _ = env.load_state_file(os.path.join(args.out, check_sta))
        true_h = _height(st_true)
        if abs(replay_h - true_h) > 2:
            print(f"[densify] LEG DESYNC a{ai} c{leg_idx}: replay lands "
                  f"h{replay_h}, successor {check_sta} loads h{true_h} "
                  f"— leg skipped")
            return
        js = list(range(j_lo, j_hi))
        step = max(1, len(js) // MAX_RUNGS_PER_LEG)
        for j in js[::step]:
            key = (ai, leg_idx, j)
            if key in minted:
                continue
            st = replay(env, a_path, byts, j)
            h = _height(st)
            if not (BAND_LO <= h <= BAND_HI and st["is_dead"] == 1
                    and not st["is_jumping"]):
                minted[key] = None       # mid-jump/dead/out-of-band frame
                continue
            name = f"a{ai}_c{leg_idx}_d{j}.sta"
            for _ in range(4):
                env._exchange(env.A_SAVE)
            env._hold(env.A_NOOP, 3)
            shutil.copyfile(env._slot_sta_path(),
                            os.path.join(args.out, name))
            minted[key] = {"sta": name, "height": h}

    # Pass 1: mint rungs for every (stuck, successor) adjacency in any chain.
    for ch in chains:
        cells = ch["cells"]
        for i, c in enumerate(cells[:-1]):
            if c["sta"] not in args.stuck:
                continue
            x_ai, x_idx, x_j = parse_name(c["sta"])
            s_ai, s_idx, s_j = parse_name(cells[i + 1]["sta"])
            if s_ai != x_ai:
                continue
            if s_idx != x_idx:
                # Successor sits on a DIFFERENT leg (stuck cell is a leg end
                # or a rung before a pruned adjacency). Rungs go on the
                # successor's leg: j in [1, J or leg end).
                # With --prune-descents exports the manifest-adjacent cell is
                # not always the archive parent — replay the successor's
                # bytes from its TRUE parent (whose .sta lives in the
                # archive), or they land somewhere else entirely.
                rec = archives[s_ai].by_idx[s_idx]
                a_path = (os.path.join(args.out, c["sta"])
                          if rec["parent"] == x_idx and x_j is None
                          else archives[s_ai].sta_path(rec["parent"]))
                j_hi = s_j if s_j is not None else len(rec["bytes"]) - 1
                mint_leg(s_ai, s_idx, a_path, 1, j_hi,
                         s_j, cells[i + 1]["sta"])
            elif x_j is not None:
                # Same leg: stuck rung _dJ → later rung _dJ2 or the leg end.
                rec = archives[x_ai].by_idx[x_idx]
                j_hi = s_j if s_j is not None else len(rec["bytes"]) - 1
                mint_leg(x_ai, x_idx, archives[x_ai].sta_path(rec["parent"]),
                         x_j + 1, j_hi, s_j, cells[i + 1]["sta"])

    # Doom screen: drop rungs where a do-little Mario nearly always dies at
    # once — those add gate noise no policy can learn through. Threshold is
    # deliberately loose (5/6): hard-but-winnable spawns are the curriculum.
    kept = dropped = 0
    for key, cell in list(minted.items()):
        if cell is None:
            continue
        path = os.path.join(args.out, cell["sta"])
        deaths = 0
        for t in range(args.trials):
            st, _ = env.load_state_file(path)
            if t == 0:
                ok, st, _ = env._is_responsive()
                if not ok:
                    deaths = args.trials
                    break
            lives0 = st["lives"]
            jitter = int(env.np_random.integers(0, 48))
            for k in range(jitter + args.doom_steps):
                a = env.A_NOOP if k < jitter else \
                    ACTIONS[int(env.np_random.integers(len(ACTIONS)))]
                st, _ = env._exchange(a)
                if st["lives"] < lives0:
                    deaths += 1
                    break
        if deaths >= args.trials - 1:
            os.remove(path)
            minted[key] = None
            dropped += 1
            print(f"[densify] DROP doomed rung {cell['sta']} "
                  f"({deaths}/{args.trials} quick deaths)")
        else:
            kept += 1
    env.close()

    # Pass 2: splice surviving rungs into every chain at the adjacency,
    # ascending j (= trajectory order: later rung is closer to the goal).
    added_slots = 0
    for ch in chains:
        out_cells = []
        cells = ch["cells"]
        for i, c in enumerate(cells):
            out_cells.append(c)
            if c["sta"] not in args.stuck or i + 1 >= len(cells):
                continue
            x_ai, x_idx, x_j = parse_name(c["sta"])
            s_ai, s_idx, s_j = parse_name(cells[i + 1]["sta"])
            leg_idx = s_idx if x_j is None else x_idx
            rungs = sorted(
                (k[2], cell) for k, cell in minted.items()
                if cell is not None and k[0] == s_ai and k[1] == leg_idx
                and (x_j or 0) < k[2] < (s_j if s_j is not None else 10 ** 9))
            out_cells.extend(cell for _, cell in rungs)
            added_slots += len(rungs)
        ch["cells"] = out_cells

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"chains": chains}, f, indent=1)
    print(f"[densify] {kept} rungs minted ({dropped} doomed dropped), "
          f"{added_slots} cell slots spliced -> {args.out}/manifest.json")


if __name__ == "__main__":
    main()
