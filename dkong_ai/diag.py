"""Diagnose the height plateau: where does the policy peak, and where does it die?

Logs, per episode: the (x,y) at peak height, and the (x,y) at each life loss.
Reveals whether it dies trying to climb past the wall or while farming low.

    python -m dkong_ai.diag --rom-dir ./roms --model artifacts/ppo_dkong_run23 --port 5100
"""
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .mame_env import DonkeyKongEnv
from .dk_policy import DkFeaturesExtractor, DkFrameStackWrapper


def _load_model(path, device="auto"):
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(path, device=device), True
    except Exception:
        pass
    return PPO.load(path, device=device), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--model", default="artifacts/ppo_dkong")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--stack", type=int, default=2,
                    help="frame stack depth — must match training run (run 21+: 2)")
    ap.add_argument("--port", type=int, default=5100)
    args = ap.parse_args()

    base = DonkeyKongEnv(rom_dir=args.rom_dir, port=args.port, record=False)
    venv = DkFrameStackWrapper(DummyVecEnv([lambda: base]), n_stack=args.stack)
    model, is_lstm = _load_model(args.model)

    print(f"model type: {'RecurrentPPO (LSTM)' if is_lstm else 'PPO'}")
    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        peak_h, peak_xy = 0, (0, 0)
        deaths = []
        prev_lives = base._prev["lives"] if base._prev else 3
        bands = {}
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        while not done:
            if is_lstm:
                action, lstm_state = model.predict(
                    obs, state=lstm_state, episode_start=episode_start,
                    deterministic=False)
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=False)
            obs, r, dones, infos = venv.step(action)
            done = bool(dones[0])
            if done:
                episode_start = np.ones((1,), dtype=bool)
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
