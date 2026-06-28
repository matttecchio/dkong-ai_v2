"""Diagnose the height plateau: where does the policy peak, and where does it die?

Logs, per episode: the (x,y) at peak height, and the (x,y) at each life loss.
Reveals whether it dies trying to climb past the wall or while farming low.

    python -m dkong_ai.diag --rom-dir ./roms --model artifacts/ppo_dkong_climb --port 5100
"""
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from .mame_env import DonkeyKongEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--model", default="artifacts/ppo_dkong_climb")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--port", type=int, default=5100)
    args = ap.parse_args()

    base = DonkeyKongEnv(rom_dir=args.rom_dir, port=args.port, record=False)
    venv = VecFrameStack(DummyVecEnv([lambda: base]), n_stack=4)
    model = PPO.load(args.model, device="cuda")

    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        peak_h, peak_xy = 0, (0, 0)
        deaths = []
        prev_lives = base._prev["lives"]
        # also tally how many steps spent at each height band
        bands = {}
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, r, dones, infos = venv.step(action)
            done = bool(dones[0])
            s = infos[0].get("state", {})
            y, x, lives = s.get("mario_y", 0), s.get("mario_x", 0), s.get("lives", 0)
            if y:
                h = 240 - y
                if h > peak_h:
                    peak_h, peak_xy = h, (x, y)
                bands[h // 20 * 20] = bands.get(h // 20 * 20, 0) + 1
            if lives < prev_lives:
                deaths.append((x, y))
            prev_lives = lives
        top_bands = sorted(bands.items(), key=lambda kv: -kv[1])[:4]
        print(f"ep {ep}: peak_h={peak_h} at (x={peak_xy[0]},y={peak_xy[1]}) | "
              f"deaths@(x,y)={deaths} | time-by-height-band={top_bands}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
