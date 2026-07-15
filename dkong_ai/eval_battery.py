"""Fixed nightly evaluation battery — the project's honest scoreboard.

Runs the same battery every time so results are comparable across days:
  1. Bottom-up episodes (live barrels, no curriculum): half deterministic,
     half stochastic. The honest floor + first-clear signal.
  2. Key-cell probes: fixed cells drawn through the normal curriculum path
     (stochastic burn-in included), rates reported split by drawn burn-in.

Appends one JSON line per run to logs/battery/battery.jsonl and prints a
summary. Uses port 5100 so it can run alongside training.

    .venv/bin/python -m dkong_ai.eval_battery --rom-dir ./roms \
        --model artifacts/ppo_dkong_run28_last
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from .mame_env import DonkeyKongEnv
from .dk_policy import DkFrameStackWrapper
from .eval import _load_model

KEY_CELLS = "0:57,4:41,9:44"   # wc_154, a1_c446_d21, a1_c457_d4 (2026-07-10)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate — honest bounds at the
    small n and rare-event rates this battery deals in."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def run_episode(venv, model, is_lstm, deterministic, max_steps=1200,
                base=None):
    obs = venv.reset()
    # Capture per-episode labels NOW: DummyVecEnv auto-resets on done, so by
    # the time this function returns, the env's mutable fields describe the
    # NEXT episode (external review round 6 — all prior battery burn-in
    # splits were off by one episode).
    meta = {}
    if base is not None:
        meta = {"start_type": base._start_type,
                "burnin": base._burnin_drawn,
                "approach_len": base._approach_len}
    lstm_state, ep_start = None, np.ones((1,), dtype=bool)
    done, steps, best_h, cleared, glitch = False, 0, 0, 0, 0
    while not done and steps < max_steps:
        if is_lstm:
            action, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=ep_start,
                deterministic=deterministic)
            ep_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, dones, infos = venv.step(action)
        done = bool(dones[0])
        steps += 1
        best_h = max(best_h, infos[0].get("max_height", 0))
        cleared = max(cleared, infos[0].get("cleared", 0))
        glitch = max(glitch, infos[0].get("glitch_kill", 0))
    return {"steps": steps, "max_h": best_h, "cleared": cleared, **meta,
            "glitch": glitch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--model", default="artifacts/ppo_dkong_run28_last")
    ap.add_argument("--manifest",
                    default="artifacts/backward_dense12/manifest.json")
    ap.add_argument("--bottomups", type=int, default=60,
                    help="total bottom-up episodes (half det, half stoch)")
    ap.add_argument("--cells", default=KEY_CELLS)
    ap.add_argument("--cell-eps", type=int, default=12)
    ap.add_argument("--stack", type=int, default=2)
    ap.add_argument("--port", type=int, default=5100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = DonkeyKongEnv(rom_dir=args.rom_dir, port=args.port, record=False,
                         backward_manifest=os.path.abspath(args.manifest))
    chains = base._bw_chains
    base.P_NO_BARRELS = 0.0
    venv = DkFrameStackWrapper(DummyVecEnv([lambda: base]), n_stack=args.stack)
    model, is_lstm = _load_model(args.model)
    base.reset(seed=args.seed)

    result = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
              "model": args.model, "seed": args.seed}

    # --- phase 1: bottom-up floor -------------------------------------
    base.set_bottomup_eval()
    for label, det in (("det", True), ("stoch", False)):
        n_eps = args.bottomups // 2 + (args.bottomups % 2 if det else 0)
        eps = [run_episode(venv, model, is_lstm, det)
               for _ in range(n_eps)]
        clean = [e for e in eps if not e["glitch"]]
        clears = sum(e["cleared"] for e in eps)
        entry = {"n": len(eps), "clean_n": len(clean), "clears": clears,
                 "clear_ci": wilson(clears, len(eps))}
        if clean:
            hs = sorted(e["max_h"] for e in clean)
            entry["mean_h"] = round(float(np.mean(hs)), 1)
            entry["median_h"] = hs[len(hs) // 2]
        else:
            # No clean episodes: report null, never a masking 0.0 (a battery
            # smoke run once printed "mean 0.0" that was really "no data").
            entry["mean_h"] = entry["median_h"] = None
            entry["clean_insufficient"] = True
        result[f"bottomup_{label}"] = entry
        print(f"bottomup {label}: clean mean {entry['mean_h']}"
              f" median {entry['median_h']} clears {clears}"
              f" ci={entry['clear_ci']}"
              + (" CLEAN_INSUFFICIENT" if not clean else ""), flush=True)

    # --- phase 2: key cells through the training path ------------------
    result["cells"] = {}
    for spec in args.cells.split(","):
        ci, pos = (int(x) for x in spec.split(":"))
        cell = chains[ci][pos]
        name = os.path.basename(cell["sta"]).replace(".sta", "")
        base.pin_backward_cell(ci, pos)
        by_burnin = {}
        for _ in range(args.cell_eps):
            e = run_episode(venv, model, is_lstm, deterministic=False,
                            max_steps=500, base=base)
            if e.get("start_type") != "curriculum":
                continue                      # unresponsive load fallback
            b = e.get("burnin", 0)
            by_burnin.setdefault(b, []).append(e)
        result["cells"][name] = {
            str(b): {"n": len(v),
                     "clear": round(sum(e["cleared"] for e in v) / len(v), 2),
                     "mean_h": round(float(np.mean([e["max_h"] for e in v])), 0)}
            for b, v in sorted(by_burnin.items())}
        print(f"cell {name}: " + " | ".join(
            f"burnin={b} n={len(v)} clear={sum(e['cleared'] for e in v)/len(v):.2f}"
            for b, v in sorted(by_burnin.items())), flush=True)
        base.unpin_backward()                  # restore for next pin

    venv.close()
    os.makedirs("logs/battery", exist_ok=True)
    with open("logs/battery/battery.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print("battery appended -> logs/battery/battery.jsonl", flush=True)


if __name__ == "__main__":
    main()
