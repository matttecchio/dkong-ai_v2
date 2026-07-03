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
                    help="ROM dir for --verify-states")
    ap.add_argument("--verify-port", type=int, default=5301)
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
            for rec in arch.ancestors(w["parent"]):
                name = f"a{ai}_c{rec['idx']}.sta"
                # Always overwrite: names encode archive position + cell idx,
                # which repeat when an archive is wiped and regenerated — a
                # skip-if-exists here would silently pair the new manifest
                # with stale state bytes from the previous export.
                shutil.copyfile(arch.sta_path(rec["idx"]),
                                os.path.join(args.out, name))
                cells.append({"sta": name, "height": rec["height"]})
            chains.append({"cells": cells})
        print(f"[export] {adir}: {len(picked)} winners "
              f"(of {len(arch.winners)}, {len(seen_parents)} distinct exits)")

    if args.verify_states and chains:
        from .mame_env import DonkeyKongEnv
        env = DonkeyKongEnv(args.rom_dir, port=args.verify_port, record=False)
        env._p_curric = 0.0
        env.P_NO_BARRELS = 0.0
        try:
            env.reset()
            bad = set()
            for name in sorted({c["sta"] for ch in chains for c in ch["cells"]}):
                env.load_state_file(os.path.join(args.out, name))
                ok, _, _ = env._is_responsive()
                if not ok:
                    bad.add(name)
                    print(f"[export] DROP unresponsive state {name}")
        finally:
            env.close()
        for ch in chains:
            ch["cells"] = [c for c in ch["cells"] if c["sta"] not in bad]
        chains = [ch for ch in chains if ch["cells"]]
        print(f"[export] state verification: {len(bad)} unresponsive dropped")

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
