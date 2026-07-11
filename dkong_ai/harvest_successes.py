"""Harvest the policy's OWN successes into curriculum material.

Training records every honest success (logs/successes/dk_*.jsonl: start .sta
+ executed action log — curriculum episodes replay deterministically). This
tool replays them offline and mints:
  * RUNGS: clean-frame snapshots through a requested height band — fresher
    and on-distribution vs the phase-1 random-exploration states;
  * APPROACHES: for existing manifest cells that lack one, if a success
    trajectory passes close by, its last steps become a verified approach
    (policy-generated approaches pass the motion filters by construction).

Outputs a staging dir (cells.json + .sta files + approaches.json); merging
into the live manifest is a deliberate manual step.

    python -m dkong_ai.harvest_successes --rom-dir ./roms \
        --manifest-dir artifacts/backward_dense13 --out artifacts/harvest_1 \
        --rungs 100:160:8
"""
import argparse
import glob
import json
import os
import shutil

from .mame_env import DonkeyKongEnv, ACTIONS

APPROACH_MAX = 14
NEAR_X, NEAR_Y = 12, 6


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rungs", default=None, metavar="LO:HI:K",
                    help="mint clean-frame rungs every K steps inside "
                         "heights [LO,HI] along each success")
    ap.add_argument("--max-successes", type=int, default=40)
    ap.add_argument("--port", type=int, default=5100)
    args = ap.parse_args()

    d = args.manifest_dir
    mani = json.load(open(os.path.join(d, "manifest.json")))
    no_approach = {}
    for ch in mani["chains"]:
        for c in ch["cells"]:
            if "approach" not in c and c["sta"] not in no_approach:
                no_approach[c["sta"]] = None   # position measured lazily

    recs, seen = [], set()
    for f in sorted(glob.glob("logs/successes/dk_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("exact"):
                continue                      # bottom starts: jitter not logged
            key = (r["start"], len(r["acts"]), r.get("gain"))
            if key in seen:
                continue
            seen.add(key)
            recs.append(r)
    recs = recs[-args.max_successes:]
    if not recs:
        raise SystemExit("[harvest] no exact success records yet")
    print(f"[harvest] {len(recs)} unique exact successes")

    os.makedirs(args.out, exist_ok=True)
    env = DonkeyKongEnv(args.rom_dir, port=args.port, record=False)
    env._p_curric = 0.0
    env.P_NO_BARRELS = 0.0
    env.reset()

    def cellpos(sta):
        if no_approach[sta] is None:
            st, _ = env.load_state_file(os.path.join(d, sta))
            no_approach[sta] = (st["mario_x"], st["mario_y"])
        return no_approach[sta]

    band = None
    if args.rungs:
        lo, hi, k = (int(x) for x in args.rungs.split(":"))
        band = (lo, hi, k)

    new_cells, new_approaches = [], {}
    n_verified = 0
    for ri, r in enumerate(recs):
        start = os.path.join(d, r["start"])
        if not os.path.exists(start):
            continue
        # Replay 1: verify + track the trajectory.
        st, _ = env.load_state_file(start)
        traj = []
        for a in r["acts"]:
            st, _ = env._exchange(ACTIONS[a])
            traj.append((st["mario_x"], st["mario_y"]))
        end_h = max(0, 240 - st["mario_y"]) if st["mario_y"] else 0
        reproduced = (st["screen_id"] > 1) if r["cleared"] else \
            (end_h - (240 - traj[0][1]) >= r["gain"] - 12 if traj else False)
        if not reproduced:
            continue
        n_verified += 1
        # Approaches for cells this success passes near.
        for sta in list(no_approach):
            if sta in new_approaches:
                continue
            cx, cy = cellpos(sta)
            for j in range(APPROACH_MAX, len(traj)):
                x, y = traj[j]
                if abs(x - cx) <= NEAR_X and abs(y - cy) <= NEAR_Y:
                    ax, ay = traj[j - APPROACH_MAX]
                    if ay < cy - 4 or (abs(ax - cx) + abs(ay - cy)) < 8:
                        break                 # descending/stationary
                    # Mint the anchor at j-APPROACH_MAX (fresh replay: a
                    # mid-run save desyncs the remainder).
                    st2, _ = env.load_state_file(start)
                    for a in r["acts"][:j - APPROACH_MAX]:
                        st2, _ = env._exchange(ACTIONS[a])
                    if st2["is_dead"] != 1 or st2.get("is_jumping", 0):
                        break
                    for _ in range(4):
                        env._exchange(env.A_SAVE)
                    env._hold(env.A_NOOP, 3)
                    aname = f"hva_{ri}_{os.path.splitext(sta)[0]}.sta"
                    shutil.copyfile(env._slot_sta_path(),
                                    os.path.join(args.out, aname))
                    new_approaches[sta] = {
                        "anchor": aname,
                        "acts": r["acts"][j - APPROACH_MAX:j]}
                    break
        # Rungs through the requested band.
        if band:
            lo, hi, k = band
            for j in range(k, len(traj), k):
                h = max(0, 240 - traj[j][1]) if traj[j][1] else 0
                if not (lo <= h <= hi):
                    continue
                st2, _ = env.load_state_file(start)
                for a in r["acts"][:j]:
                    st2, _ = env._exchange(ACTIONS[a])
                if st2["is_dead"] != 1 or st2.get("is_jumping", 0):
                    continue
                for _ in range(4):
                    env._exchange(env.A_SAVE)
                env._hold(env.A_NOOP, 3)
                name = f"hv_{ri}_{j}.sta"
                shutil.copyfile(env._slot_sta_path(),
                                os.path.join(args.out, name))
                new_cells.append({
                    "sta": name, "height": h,
                    "approach": {"anchor": r["start"],
                                 "acts": r["acts"][max(0, j - APPROACH_MAX):j]}
                    if j <= APPROACH_MAX else None})
    env.close()

    json.dump({"cells": new_cells, "approaches": new_approaches},
              open(os.path.join(args.out, "harvest.json"), "w"), indent=1)
    print(f"[harvest] {n_verified}/{len(recs)} reproduced | "
          f"{len(new_cells)} rungs, {len(new_approaches)} approaches "
          f"-> {args.out}/harvest.json (merge manually)")


if __name__ == "__main__":
    main()
