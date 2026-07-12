"""Film policy attempts from specific curriculum cells (human film review).

Pins the env's backward curriculum to one cell at a time and lets the normal
reset path load it — so episodes show exactly what training sees, spawn
burn-in included — while MAME records the whole session to an .avi.

    .venv/bin/python -m dkong_ai.film_cells --rom-dir ./roms \
        --model artifacts/ppo_dkong_run28_last \
        --cells 0:57,4:41,9:44 --eps 4 --avi run28_frontiers.avi

Convert after:  ffmpeg -i artifacts/recordings/<avi> -c:v libx264 -crf 23 \
                -pix_fmt yuv420p artifacts/recordings/<mp4>
"""
import argparse
import os

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from .mame_env import DonkeyKongEnv
from .dk_policy import DkFrameStackWrapper
from .eval import _load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest",
                    default="artifacts/backward_dense13/manifest.json")
    ap.add_argument("--cells", required=True,
                    help="comma-separated chain:pos pairs, e.g. 0:57,4:41,9:44")
    ap.add_argument("--eps", type=int, default=4, help="episodes per cell")
    ap.add_argument("--max-steps", type=int, default=450)
    ap.add_argument("--stack", type=int, default=2)
    ap.add_argument("--port", type=int, default=5100,
                    help="keep clear of a running training (it uses 5000+)")
    ap.add_argument("--avi", default="film_cells.avi",
                    help="written into artifacts/recordings/")
    args = ap.parse_args()

    recdir = os.path.abspath("artifacts/recordings")
    os.makedirs(recdir, exist_ok=True)
    base = DonkeyKongEnv(rom_dir=args.rom_dir, port=args.port, record=False,
                         backward_manifest=os.path.abspath(args.manifest),
                         extra_mame_args=["-snapshot_directory", recdir,
                                          "-aviwrite", args.avi])
    all_chains = base._bw_chains
    base.P_NO_BARRELS = 0.0
    venv = DkFrameStackWrapper(DummyVecEnv([lambda: base]), n_stack=args.stack)
    model, is_lstm = _load_model(args.model)
    print(f"model: {'RecurrentPPO' if is_lstm else 'PPO'} | avi -> "
          f"{os.path.join(recdir, args.avi)}", flush=True)

    for spec in args.cells.split(","):
        ci, pos = (int(x) for x in spec.split(":"))
        cell = all_chains[ci][pos]
        base.pin_backward_cell(ci, pos)
        print(f"=== chain {ci} pos {pos}: {os.path.basename(cell['sta'])} "
              f"(label h{cell['height']}) ===", flush=True)
        for ep in range(args.eps):
            obs = venv.reset()
            if base._start_type != "curriculum":
                print("  (unresponsive load — fell back to bottom start)",
                      flush=True)
            lstm_state = None
            ep_start = np.ones((1,), dtype=bool)
            done, steps, best_h, cleared = False, 0, 0, 0
            while not done and steps < args.max_steps:
                if is_lstm:
                    action, lstm_state = model.predict(
                        obs, state=lstm_state, episode_start=ep_start,
                        deterministic=False)
                    ep_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(obs, deterministic=False)
                obs, r, dones, infos = venv.step(action)
                done = bool(dones[0])
                steps += 1
                best_h = max(best_h, infos[0].get("max_height", 0))
                cleared = max(cleared, infos[0].get("cleared", 0))
            print(f"  ep{ep}: steps={steps} max_h={best_h} cleared={cleared}",
                  flush=True)

    venv.close()                       # clean quit finalizes the .avi
    print("done", flush=True)


if __name__ == "__main__":
    main()
