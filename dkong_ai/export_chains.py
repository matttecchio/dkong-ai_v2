"""Export Go-Explore winner chains as a backward-algorithm curriculum.

Reads one or more phase-1 archives, picks a diverse subset of winners (at most
one per distinct final cell), and copies the union of their ancestor-chain
save-states into a self-contained directory with a manifest:

    manifest.json  {"chains": [{"cells": [{"sta": "...", "height": h}, ...]}]}

Each chain's cells are root-first start states along a PROVEN bottom-up route.
Phase-2 training (train.py --backward-dir) starts episodes from the deepest
allowed cell of a random chain and walks the start back toward the bottom as
the clear rate rises.

    python -m dkong_ai.export_chains --out artifacts/backward \
        --archive artifacts/go_explore_run1 --archive artifacts/go_explore
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from .go_explore import Archive


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", action="append", required=True,
                    help="phase-1 archive dir (repeatable)")
    ap.add_argument("--out", default="artifacts/backward")
    ap.add_argument("--per-archive", type=int, default=8,
                    help="max winners exported per archive")
    ap.add_argument("--verify-states", action="store_true",
                    help="load every exported state in a scratch MAME and drop "
                         "unresponsive ones (frozen cutscene/transition "
                         "snapshots always fall back to bottom starts in "
                         "training — pure waste)")
    ap.add_argument("--rom-dir", default="./roms",
                    help="ROM dir for --verify-states / --densify")
    ap.add_argument("--verify-port", type=int, default=5301)
    ap.add_argument("--densify", default=None, metavar="LO:HI:K",
                    help="insert extra cells through a hard height band: for "
                         "each chain leg touching heights [LO,HI], replay the "
                         "leg's recorded actions and save a state every K "
                         "steps. Motivation: the walk-back stalled where "
                         "consecutive cells were ~20 steps apart across the "
                         "top-girder barrel lane (run 27j: 2.5%% clears flat "
                         "over 12k draws) — a finer staircase gives each tier "
                         "a learnable increment. Try 130:190:5.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    chains = []
    for ai, adir in enumerate(args.archive):
        arch = Archive(adir)
        if not arch.load() or not arch.winners:
            print(f"[export] {adir}: no winners — skipped")
            continue
        # Diversity: at most one winner per distinct final cell (parent).
        seen_parents = set()
        picked = []
        for w in arch.winners:
            if w["parent"] in seen_parents:
                continue
            seen_parents.add(w["parent"])
            picked.append(w)
            if len(picked) >= args.per_archive:
                break
        for w in picked:
            cells = []
            prev_rec = None
            for rec in arch.ancestors(w["parent"]):
                name = f"a{ai}_c{rec['idx']}.sta"
                # Always overwrite: names encode archive position + cell idx,
                # which repeat when an archive is wiped and regenerated — a
                # skip-if-exists here would silently pair the new manifest
                # with stale state bytes from the previous export.
                shutil.copyfile(arch.sta_path(rec["idx"]),
                                os.path.join(args.out, name))
                cell = {"sta": name, "height": rec["height"]}
                if prev_rec is not None:
                    # Remember the leg so --densify can replay prev -> rec.
                    cell["_leg"] = (ai, prev_rec["idx"], rec["idx"],
                                    tuple(rec["bytes"]),
                                    prev_rec["height"], rec["height"])
                cells.append(cell)
                prev_rec = rec
            chains.append({"cells": cells})
        print(f"[export] {adir}: {len(picked)} winners "
              f"(of {len(arch.winners)}, {len(seen_parents)} distinct exits)")

    env = None
    if (args.densify or args.verify_states) and chains:
        from .mame_env import DonkeyKongEnv
        env = DonkeyKongEnv(args.rom_dir, port=args.verify_port, record=False)
        env._p_curric = 0.0
        env.P_NO_BARRELS = 0.0
        env.reset()

    if args.densify and chains:
        lo, hi, k = (int(x) for x in args.densify.split(":"))
        dense_cache: dict = {}   # (ai, from_idx, to_idx) -> [cell, ...]
        n_new = 0
        for ch in chains:
            out_cells = []
            for cell in ch["cells"]:
                leg = cell.pop("_leg", None)
                if leg is not None:
                    ai, a_idx, b_idx, byts, h_a, h_b = leg
                    band = (lo <= h_a <= hi or lo <= h_b <= hi
                            or (h_a < lo < h_b) or (h_b < lo < h_a))
                    if band:
                        key = (ai, a_idx, b_idx)
                        if key not in dense_cache:
                            dense_cache[key] = []
                            a_sta = os.path.join(args.out,
                                                 f"a{ai}_c{a_idx}.sta")
                            # One fresh replay per save point: saving mid-run
                            # costs ~7 extra exchanges (A_SAVE + settle) which
                            # would desync the rest of the leg from the
                            # recorded bytes.
                            for j in range(k - 1, len(byts) - 1, k):
                                env.load_state_file(a_sta)
                                st = None
                                for b in byts[:j + 1]:
                                    st, _ = env._exchange(b)
                                h = max(0, 240 - st["mario_y"]) \
                                    if st["mario_y"] else 0
                                # Save only clean frames: alive (0x6200 is
                                # 1=alive) and not mid-jump-arc.
                                if not (lo <= h <= hi and st["is_dead"] == 1
                                        and not st["is_jumping"]):
                                    continue
                                name = f"a{ai}_c{b_idx}_d{j}.sta"
                                for _ in range(4):
                                    env._exchange(env.A_SAVE)
                                env._hold(env.A_NOOP, 3)
                                shutil.copyfile(
                                    env._slot_sta_path(),
                                    os.path.join(args.out, name))
                                dense_cache[key].append(
                                    {"sta": name, "height": h})
                        out_cells.extend(dense_cache[key])
                        n_new += len(dense_cache[key])
                out_cells.append(cell)
            ch["cells"] = out_cells
        print(f"[export] densify {lo}-{hi} every {k}: "
              f"{sum(len(v) for v in dense_cache.values())} unique new states, "
              f"{n_new} cell slots added across chains")
    else:
        for ch in chains:
            for cell in ch["cells"]:
                cell.pop("_leg", None)

    if args.verify_states and chains:
        bad = set()
        for name in sorted({c["sta"] for ch in chains for c in ch["cells"]}):
            env.load_state_file(os.path.join(args.out, name))
            ok, _, _ = env._is_responsive()
            if not ok:
                bad.add(name)
                print(f"[export] DROP unresponsive state {name}")
        for ch in chains:
            ch["cells"] = [c for c in ch["cells"] if c["sta"] not in bad]
        chains = [ch for ch in chains if ch["cells"]]
        print(f"[export] state verification: {len(bad)} unresponsive dropped")

    if env is not None:
        env.close()

    if not chains:
        raise SystemExit("[export] no winners in any archive — refusing to "
                         "write an empty manifest (it would silently disable "
                         "the backward curriculum)")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"chains": chains}, f, indent=1)
    n_sta = len([f for f in os.listdir(args.out) if f.endswith(".sta")])
    print(f"[export] wrote {len(chains)} chains, {n_sta} unique states "
          f"-> {args.out}/manifest.json")


if __name__ == "__main__":
    main()
