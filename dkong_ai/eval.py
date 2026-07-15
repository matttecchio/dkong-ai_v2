"""Evaluate a trained policy and RECORD watchable .inp files.

Runs with record=True (intro resets, no save-state loads) so each session's .inp
plays back cleanly via scripts/playback.sh. Reports reward / max-height / score
per episode so you can pick the good one to watch.

    python -m dkong_ai.eval --rom-dir ./roms --model artifacts/ppo_dkong --episodes 5
"""
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .mame_env import DonkeyKongEnv
from .dk_policy import DkFeaturesExtractor, DkFrameStackWrapper


def _load_model(path, device="auto"):
    """Load PPO or RecurrentPPO depending on what was saved.

    device="auto" (review r14): CUDA when available (this box), CPU
    fallback elsewhere — same policy as train.py's --device auto."""
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
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--stack", type=int, default=2,
                    help="frame stack depth — must match the training run (run 21+: 2)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--port", type=int, default=5100,
                    help="keep clear of a running training (it uses 5000+)")
    ap.add_argument("--p-no-barrels", type=float, default=None)
    ap.add_argument("--p-curric", type=float, default=None)
    args = ap.parse_args()

    # record=True -> clean playable .inp; own port so it won't clash with training.
    base = DonkeyKongEnv(rom_dir=args.rom_dir, port=args.port, record=True)
    if args.p_no_barrels is not None:
        base.P_NO_BARRELS = args.p_no_barrels
    if args.p_curric is not None:
        base._p_curric = args.p_curric
    venv = DkFrameStackWrapper(DummyVecEnv([lambda: base]), n_stack=args.stack)
    model, is_lstm = _load_model(args.model)

    print(f"recording -> {base._inp_path or '(set on first reset)'}")
    print(f"model type: {'RecurrentPPO (LSTM)' if is_lstm else 'PPO'}")
    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        total_r, best_h, last = 0.0, 0, {}
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        while not done:
            if is_lstm:
                action, lstm_state = model.predict(
                    obs, state=lstm_state, episode_start=episode_start,
                    deterministic=args.deterministic)
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, dones, infos = venv.step(action)
            done = bool(dones[0])
            total_r += float(r[0])
            best_h = max(best_h, infos[0].get("max_height", 0))
            last = infos[0].get("state", last)
            if done:
                episode_start = np.ones((1,), dtype=bool)
        print(f"ep {ep}: reward={total_r:+.1f} max_height={best_h} "
              f"score={last.get('score')} cleared={infos[0].get('cleared')}")
    print(f"\n.inp recorded: {base._inp_path}")
    print(f"watch it:  ./scripts/playback.sh {base._inp_path}")
    venv.close()


if __name__ == "__main__":
    main()
