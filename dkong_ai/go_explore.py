"""Go-Explore phase 1 for the Donkey Kong barrel board.

Classic (policy-free) Go-Explore: an archive of cells (discretised Mario
positions), each holding a MAME save-state plus the exact action-byte
trajectory that reaches it from its parent cell. Workers repeatedly:

  1. pick an under-explored cell (count-based weight, mild height bias),
  2. restore its save-state — copy the archived .sta onto the worker's
     per-port slot file (dk_<port>.sta) and send A_LOAD; no bridge changes,
  3. explore with sticky random actions (no neural net, no GPU),
  4. snapshot every newly reached cell back into the archive.

Progress is banked permanently: the agent never has to survive the whole
left-traverse in one attempt, which is what PPO failed at for 26 runs.
Success = screen_id leaving 1 with lives intact (bottom-up clear, live
barrels); the winning byte trajectory and the save-states along its
ancestor chain feed phase 2 (backward-algorithm robustification).

Consistency rules that make this correct:
- Cells are immutable once committed: cell_<idx>.sta is written exactly once,
  so a byte trajectory recorded after restoring a cell stays valid forever.
- The restore prologue is a FIXED command sequence (3xLOAD + 2xNOOP +
  UNFREEZE), so every restore of a cell advances identical frames.
- The A_SAVE/A_NOOP bytes spent snapshotting mid-rollout are appended to the
  running trajectory: restore(parent) + bytes lands frame-exactly on the
  child's .sta state, so trajectories stitch across generations.
- Doomed cells (snapshotted mid-death-animation: restoring them dies almost
  immediately) are detected by early-death statistics and retired, freeing
  their key for a fresh snapshot under a new idx.

Run:      python -m dkong_ai.go_explore --rom-dir ./roms --workers 6
Self-test: add --validate (determinism + cross-port snapshot round-trip).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import threading
import time

from .mame_env import ACTIONS, DonkeyKongEnv

CELL_X = 8      # cell discretisation, game pixels per bin (x)
CELL_H = 8      # cell discretisation, pixels of height per bin
PORT_BASE = 5200                # clear of training (5000-5015) and eval (5100)
EARLY_DEATH_STEPS = 8           # dying within this many steps of a restore is
EARLY_DEATH_RATIO = 0.75        # "early"; retire a cell when >=75% of >=4
EARLY_DEATH_MIN_TRIES = 4       # restores die early (mid-death-animation save)


def _repo_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Archive:
    """Thread-safe Go-Explore cell archive with JSON persistence."""

    def __init__(self, root_dir: str):
        self.dir = root_dir
        self.cells_dir = os.path.join(root_dir, "cells")
        os.makedirs(self.cells_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.cells: dict[str, dict] = {}    # key -> live record
        self.by_idx: dict[int, dict] = {}   # idx -> record (incl. retired)
        self.next_idx = 0
        self.best_height = 0
        self.winners: list[dict] = []
        self.rollouts = 0
        self.steps = 0
        self.dirty = False

    @staticmethod
    def key_of(state: dict) -> str | None:
        y = state["mario_y"]
        if not y:                            # off-playfield sentinel
            return None
        h = max(0, DonkeyKongEnv.BASE_Y - y)
        ham = 1 if state.get("has_hammer") else 0
        return f"{state['mario_x'] // CELL_X},{h // CELL_H},{ham}"

    def sta_path(self, idx: int) -> str:
        return os.path.join(self.cells_dir, f"cell_{idx}.sta")

    def reserve(self, key: str, height: int, parent: int | None,
                nbytes: int) -> int | None:
        """Claim a new idx for `key`, or None if the key is already live."""
        with self.lock:
            if key in self.cells:
                return None
            idx = self.next_idx
            self.next_idx += 1
            base = self.by_idx[parent]["steps"] if parent is not None else 0
            rec = {"idx": idx, "key": key, "height": height, "parent": parent,
                   "steps": base + nbytes, "bytes": None,
                   "chosen": 0, "early_deaths": 0, "dead": False}
            self.cells[key] = rec
            self.by_idx[idx] = rec
            return idx

    def commit(self, idx: int, traj: list[int]):
        with self.lock:
            rec = self.by_idx[idx]
            rec["bytes"] = list(traj)
            rec["steps"] = ((self.by_idx[rec["parent"]]["steps"]
                             if rec["parent"] is not None else 0) + len(traj))
            if rec["height"] > self.best_height:
                self.best_height = rec["height"]
            self.dirty = True

    def abort(self, idx: int):
        with self.lock:
            rec = self.by_idx.pop(idx, None)
            if rec and self.cells.get(rec["key"]) is rec:
                del self.cells[rec["key"]]

    def select(self, rng: random.Random) -> dict | None:
        with self.lock:
            recs = [r for r in self.by_idx.values()
                    if r["bytes"] is not None and not r["dead"]]
            if not recs:
                return None
            # Count-based novelty x mild height bias x chain-length penalty
            # (DK's bonus timer ends the board ~2500 macro-steps in, so very
            # deep chains are dead ends).
            ws = [(1.0 / math.sqrt(1.0 + r["chosen"]))
                  * (1.0 + r["height"] / 40.0)
                  * (2000.0 / (2000.0 + r["steps"])) for r in recs]
            rec = rng.choices(recs, weights=ws, k=1)[0]
            rec["chosen"] += 1
            return {"idx": rec["idx"], "steps": rec["steps"]}

    def mark_early_death(self, idx: int):
        with self.lock:
            rec = self.by_idx.get(idx)
            if rec is None or rec["parent"] is None:   # never retire the root
                return
            rec["early_deaths"] += 1
            if (rec["chosen"] >= EARLY_DEATH_MIN_TRIES
                    and rec["early_deaths"] >= EARLY_DEATH_RATIO * rec["chosen"]):
                rec["dead"] = True
                if self.cells.get(rec["key"]) is rec:
                    del self.cells[rec["key"]]         # key becomes claimable
                self.dirty = True

    def add_winner(self, parent: int, traj: list[int], screen_id: int):
        with self.lock:
            self.winners.append(
                {"parent": parent, "bytes": list(traj), "screen_id": screen_id,
                 "steps": self.by_idx[parent]["steps"] + len(traj)})
            self.dirty = True

    def count(self, alive_only: bool = True) -> int:
        with self.lock:
            return len(self.cells) if alive_only else len(self.by_idx)

    def note_rollout(self, nsteps: int):
        with self.lock:
            self.rollouts += 1
            self.steps += nsteps

    def ancestors(self, idx: int) -> list[dict]:
        """Root-first chain of records ending at idx (phase-2 start states)."""
        with self.lock:
            chain = []
            cur: int | None = idx
            while cur is not None:
                rec = self.by_idx[cur]
                chain.append(rec)
                cur = rec["parent"]
        return list(reversed(chain))

    # -- persistence -------------------------------------------------------
    def save(self, force: bool = False):
        with self.lock:
            if not (self.dirty or force):
                return
            blob = {"next_idx": self.next_idx, "best_height": self.best_height,
                    "rollouts": self.rollouts, "steps": self.steps,
                    "winners": self.winners,
                    "cells": [r for r in self.by_idx.values()
                              if r["bytes"] is not None]}
            self.dirty = False
        tmp = os.path.join(self.dir, "archive.json.tmp")
        with open(tmp, "w") as f:
            json.dump(blob, f)
        os.replace(tmp, os.path.join(self.dir, "archive.json"))

    def load(self) -> bool:
        path = os.path.join(self.dir, "archive.json")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            blob = json.load(f)
        with self.lock:
            self.next_idx = blob["next_idx"]
            self.best_height = blob["best_height"]
            self.rollouts = blob.get("rollouts", 0)
            self.steps = blob.get("steps", 0)
            # Drop cells whose .sta is missing AND (transitively) all their
            # descendants — a parent always has a lower idx than its children,
            # so one ascending pass keeps by_idx ancestor-complete and
            # ancestors() can stay strict. Winners with dropped parents go too.
            dropped = 0
            for rec in sorted(blob["cells"], key=lambda r: r["idx"]):
                parent = rec["parent"]
                if ((parent is not None and parent not in self.by_idx)
                        or not os.path.exists(self.sta_path(rec["idx"]))):
                    dropped += 1
                    continue
                self.by_idx[rec["idx"]] = rec
                if not rec.get("dead"):
                    self.cells[rec["key"]] = rec
            winners = blob.get("winners", [])
            self.winners = [w for w in winners if w["parent"] in self.by_idx]
            lost_w = len(winners) - len(self.winners)
            if dropped or lost_w:
                print(f"[archive] dropped {dropped} cells (missing .sta or "
                      f"orphaned) and {lost_w} winners on load", flush=True)
        return True


def restore_file(env: DonkeyKongEnv, sta: str) -> dict:
    """Load `sta` and unfreeze barrels. FIXED command sequence (3xLOAD +
    2xNOOP + UNFREEZE) so restores are frame-reproducible — load_state_file's
    backup bookkeeping is file-only and adds no frames."""
    env.load_state_file(sta)
    state, _ = env._exchange(env.A_UNFREEZE_BARRELS)
    return state


class Explorer:
    """One worker: a persistent MAME driven in restore->rollout->snapshot
    loops against the shared archive, via the env's existing primitives."""

    def __init__(self, env: DonkeyKongEnv, archive: Archive,
                 rng: random.Random, rollout_steps: int = 100,
                 sticky: float = 0.75):
        self.env = env
        self.arch = archive
        self.rng = rng
        self.rollout_steps = rollout_steps
        self.sticky = sticky

    def snapshot(self, idx: int) -> list[int]:
        """Bank the CURRENT machine state as cell `idx`. Returns the command
        bytes the save spent (they advanced frames and must join the running
        trajectory so replays stay frame-exact)."""
        self.env._save_state()          # 4x A_SAVE + 3x NOOP; .sta now on disk
        shutil.copyfile(self.env._slot_sta_path(), self.arch.sta_path(idx))
        return [self.env.A_SAVE] * 4 + [self.env.A_NOOP] * 3

    def rollout(self) -> str | None:
        sel = self.arch.select(self.rng)
        if sel is None:
            return None
        state = restore_file(self.env, self.arch.sta_path(sel["idx"]))
        prev_lives = state["lives"]
        traj: list[int] = []            # bytes sent since the restore prologue
        act = self.env.A_NOOP
        outcome = None
        for step in range(self.rollout_steps):
            if self.rng.random() > self.sticky:
                act = ACTIONS[self.rng.randrange(len(ACTIONS))]
            state, _ = self.env._exchange(act)
            traj.append(act)
            lives = state["lives"]
            if lives < prev_lives:      # died: bank nothing from here on
                if step < EARLY_DEATH_STEPS:
                    self.arch.mark_early_death(sel["idx"])
                break
            prev_lives = lives
            if state["screen_id"] != 1 and lives > 0:
                # Left the barrel board alive = BOTTOM-UP CLEAR.
                self.arch.add_winner(sel["idx"], traj, state["screen_id"])
                outcome = "winner"
                break
            key = self.arch.key_of(state)
            if key is None:
                continue
            height = max(0, DonkeyKongEnv.BASE_Y - state["mario_y"])
            idx = self.arch.reserve(key, height, sel["idx"], len(traj))
            if idx is not None:
                try:
                    traj.extend(self.snapshot(idx))
                except OSError:
                    self.arch.abort(idx)
                    raise
                self.arch.commit(idx, traj)
        self.arch.note_rollout(len(traj))
        return outcome


def make_env(rom_dir: str, port: int) -> DonkeyKongEnv:
    env = DonkeyKongEnv(rom_dir, port=port, record=False)
    env._p_curric = 0.0          # instance overrides: always bottom start,
    env.P_NO_BARRELS = 0.0       # always live barrels
    env.reset()                  # boot MAME, run intro, save bottom to slot
    return env


def worker_loop(ex: Explorer, stop: threading.Event, deadline: float,
                stop_on_success: bool):
    while not stop.is_set() and time.time() < deadline:
        try:
            if ex.rollout() == "winner" and stop_on_success:
                stop.set()
        except (ConnectionError, OSError):
            try:
                ex.env._recover()
            except Exception as e:          # MAME won't come back: retire worker
                print(f"[worker :{ex.env.port}] recover failed: {e}", flush=True)
                return


def init_root(arch: Archive, env: DonkeyKongEnv):
    """Canonical bottom start = this env's post-intro slot save -> cell 0."""
    state, _ = env._exchange(env.A_NOOP)
    key = arch.key_of(state) or "root"
    idx = arch.reserve(key, max(0, DonkeyKongEnv.BASE_Y - state["mario_y"]),
                       None, 0)
    assert idx == 0
    shutil.copyfile(env._slot_sta_path(), arch.sta_path(0))
    arch.commit(0, [])


def verify_winner(env: DonkeyKongEnv, arch: Archive, w: dict) -> bool:
    """Deterministically replay a winner: restore its parent cell, resend its
    bytes, confirm the screen change reproduces."""
    restore_file(env, arch.sta_path(w["parent"]))
    state = None
    for b in w["bytes"]:
        state, _ = env._exchange(b)
    ok = bool(state and state["screen_id"] != 1 and state["lives"] > 0)
    print(f"[verify] winner replay: screen_id={state['screen_id'] if state else '?'}"
          f" lives={state['lives'] if state else '?'} -> {'PASS' if ok else 'FAIL'}")
    return ok


def validate(envs: list[DonkeyKongEnv], arch: Archive, seed: int) -> bool:
    """Self-test: (1) restore+replay determinism, (2) snapshot round-trip
    across two different MAME instances (cross-port .sta portability)."""
    envA = envs[0]
    envB = envs[1] if len(envs) > 1 else envs[0]
    rng = random.Random(seed)
    seq = []
    act = 0
    for _ in range(150):
        if rng.random() > 0.75:
            act = ACTIONS[rng.randrange(len(ACTIONS))]
        seq.append(act)

    def trace(env, actions):
        restore_file(env, arch.sta_path(0))
        out = []
        for b in actions:
            st, _ = env._exchange(b)
            out.append((st["mario_x"], st["mario_y"], st["lives"]))
        return out

    t1, t2 = trace(envA, seq), trace(envA, seq)
    det = t1 == t2
    if not det:
        i = next(i for i, (a, b) in enumerate(zip(t1, t2)) if a != b)
        print(f"[validate] DETERMINISM FAIL at step {i}: {t1[i]} vs {t2[i]}")
    else:
        print(f"[validate] determinism: PASS (150 steps, identical traces)")

    # Snapshot round-trip: play 60 steps + settle idle, snapshot on A,
    # restore on B, Mario's position must match exactly.
    restore_file(envA, arch.sta_path(0))
    stA = None
    for b in seq[:60] + [0] * 6:
        stA, _ = envA._exchange(b)
    envA._save_state()
    tmp = os.path.join(arch.cells_dir, "validate_tmp.sta")
    shutil.copyfile(envA._slot_sta_path(), tmp)
    stB = restore_file(envB, tmp)
    os.remove(tmp)
    rt = (stA["mario_x"], stA["mario_y"]) == (stB["mario_x"], stB["mario_y"])
    print(f"[validate] cross-port round-trip: A=({stA['mario_x']},{stA['mario_y']}) "
          f"B=({stB['mario_x']},{stB['mario_y']}) lives {stA['lives']}/{stB['lives']}"
          f" -> {'PASS' if rt else 'FAIL'}")
    return det and rt


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rom-dir", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--minutes", type=float, default=1e9,
                   help="wall-clock budget (default: until success/interrupt)")
    p.add_argument("--rollout-steps", type=int, default=100)
    p.add_argument("--sticky", type=float, default=0.75,
                   help="prob of repeating the previous action each step")
    p.add_argument("--archive-dir",
                   default=os.path.join(_repo_dir(), "artifacts", "go_explore"))
    p.add_argument("--port-base", type=int, default=PORT_BASE)
    p.add_argument("--keep-going", action="store_true",
                   help="keep exploring after the first clear")
    p.add_argument("--validate", action="store_true",
                   help="run the determinism/round-trip self-test and exit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arch = Archive(args.archive_dir)
    resumed = arch.load()
    n_envs = 2 if args.validate else args.workers
    envs: list[DonkeyKongEnv] = []
    stop = threading.Event()
    threads: list[threading.Thread] = []
    status = 0
    tb = None
    try:
        for i in range(n_envs):
            envs.append(make_env(args.rom_dir, args.port_base + i))
            print(f"[go-explore] worker :{args.port_base + i} up "
                  f"({len(envs)}/{n_envs})", flush=True)
        if not resumed:
            init_root(arch, envs[0])
            arch.save(force=True)
            print("[go-explore] fresh archive; root cell banked", flush=True)
        else:
            print(f"[go-explore] resumed archive: {arch.count()} cells, "
                  f"best_h={arch.best_height}, {len(arch.winners)} winners",
                  flush=True)

        if args.validate:
            status = 0 if validate(envs, arch, args.seed) else 1
        else:
            from torch.utils.tensorboard import SummaryWriter
            n = 1
            while os.path.exists(os.path.join(_repo_dir(), "logs",
                                              f"GoExplore_{n}")):
                n += 1
            tb = SummaryWriter(os.path.join(_repo_dir(), "logs",
                                            f"GoExplore_{n}"))
            print(f"[go-explore] tensorboard -> logs/GoExplore_{n}", flush=True)
            deadline = time.time() + args.minutes * 60
            for i, env in enumerate(envs):
                ex = Explorer(env, arch, random.Random(args.seed * 1000 + i),
                              args.rollout_steps, args.sticky)
                t = threading.Thread(target=worker_loop, daemon=True,
                                     args=(ex, stop, deadline,
                                           not args.keep_going))
                t.start()
                threads.append(t)

            t0 = last_t = time.time()
            last_steps = arch.steps
            seen_winners = len(arch.winners)
            while not stop.is_set() and time.time() < deadline:
                time.sleep(30)
                now = time.time()
                with arch.lock:
                    steps, cells = arch.steps, len(arch.cells)
                    best, rolls = arch.best_height, arch.rollouts
                    nw = len(arch.winners)
                sps = (steps - last_steps) / max(1e-9, now - last_t)
                last_steps, last_t = steps, now
                print(f"[go-explore] {(now - t0) / 60:6.1f}min cells={cells} "
                      f"best_h={best} rollouts={rolls} steps={steps} "
                      f"({sps:.0f}/s) winners={nw}", flush=True)
                if nw > seen_winners:
                    seen_winners = nw
                    print("[go-explore] *** BOTTOM-UP CLEAR FOUND ***",
                          flush=True)
                if tb:
                    tb.add_scalar("explore/cells", cells, steps)
                    tb.add_scalar("explore/best_height", best, steps)
                    tb.add_scalar("explore/rollouts", rolls, steps)
                    tb.add_scalar("explore/steps_per_s", sps, steps)
                    tb.add_scalar("explore/winners", nw, steps)
                    tb.flush()
                arch.save()
    except KeyboardInterrupt:
        print("[go-explore] interrupted", flush=True)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)
        arch.save(force=True)
        if tb:
            tb.close()
        if arch.winners and not args.validate:
            print(f"[go-explore] {len(arch.winners)} winner(s); verifying #0 …",
                  flush=True)
            try:
                verify_winner(envs[0], arch, arch.winners[0])
                chain = arch.ancestors(arch.winners[0]["parent"])
                hs = [r["height"] for r in chain]
                print(f"[go-explore] winner chain: {len(chain)} cells, "
                      f"heights {hs}", flush=True)
            except (ConnectionError, OSError) as e:
                print(f"[go-explore] verify skipped (env dead: {e})", flush=True)
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        print(f"[go-explore] done: {arch.count()} cells, best_h={arch.best_height},"
              f" winners={len(arch.winners)}; archive -> {args.archive_dir}",
              flush=True)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
