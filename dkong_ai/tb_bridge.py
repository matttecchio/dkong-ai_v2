"""Tail a go_explore.log into TensorBoard events.

Stopgap for go-explore runs launched before go_explore.py wrote TensorBoard
natively: backfills every status line already in the log (reconstructing
wall-clock from the file mtime), then follows the file and streams new lines.

Run:  python -m dkong_ai.tb_bridge --follow
View: tensorboard --logdir logs   ->  run "GoExplore_1", scalars under explore/
"""
from __future__ import annotations

import argparse
import os
import re
import time

from torch.utils.tensorboard import SummaryWriter

PAT = re.compile(r"\[go-explore\]\s+([\d.]+)min cells=(\d+) best_h=(\d+) "
                 r"rollouts=(\d+) steps=(\d+) \((\d+)/s\) winners=(\d+)")


def emit(w: SummaryWriter, m: re.Match, walltime: float):
    mins, cells, best, rolls, steps, sps, wins = (float(m[1]), int(m[2]),
                                                  int(m[3]), int(m[4]),
                                                  int(m[5]), int(m[6]),
                                                  int(m[7]))
    for tag, val in (("explore/cells", cells), ("explore/best_height", best),
                     ("explore/rollouts", rolls), ("explore/steps_per_s", sps),
                     ("explore/winners", wins)):
        w.add_scalar(tag, val, global_step=steps, walltime=walltime)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default="logs/go_explore.log")
    ap.add_argument("--out", default="logs/GoExplore_1")
    ap.add_argument("--follow", action="store_true",
                    help="keep tailing the log after the backfill")
    a = ap.parse_args()

    w = SummaryWriter(a.out)
    with open(a.log) as f:
        history = f.readlines()
        matches = [m for line in history if (m := PAT.search(line))]
        # The last status line was written ~ at the log's mtime; anchor the
        # backfilled wall-clock so relative/wall x-axes look right.
        start = os.path.getmtime(a.log) - (float(matches[-1][1]) * 60
                                           if matches else 0)
        for m in matches:
            emit(w, m, start + float(m[1]) * 60)
        w.flush()
        print(f"[tb-bridge] backfilled {len(matches)} points -> {a.out}",
              flush=True)
        while a.follow:
            line = f.readline()
            if not line:
                w.flush()
                time.sleep(2)
                continue
            m = PAT.search(line)
            if m:
                emit(w, m, time.time())
    w.close()


if __name__ == "__main__":
    main()
