"""Behavioral cloning: supervised-train the SB3 CnnPolicy to imitate the expert
(from extract_bc's dataset), then save a model the RL run can --init-from.

⚠️ STALE — REFERENCE ONLY (run-5 era, 2026-06). Builds a single-image
Box(84,84,1) CnnPolicy: THREE architectures behind the active line (Dict
obs {image, ram-75} + MultiInputLstmPolicy + DkFeaturesExtractor). A model
saved here CANNOT be loaded by current train.py (--init-from raises on the
observation-space mismatch). Kept as the reference implementation of the
extract->supervise pipeline; if BC is revived, rebuild it against the
current policy class — or better, as the decaying BC-AUXILIARY-loss design
(ArcadeAI recon), which avoids init-BC's brittleness entirely.

    python -m dkong_ai.train_bc --data artifacts/bc_data.npz --out artifacts/ppo_dkong_bc
"""
import argparse
import sys

print("WARNING: train_bc.py is STALE (run-5 era, single-image CnnPolicy). "
      "Its output cannot be loaded by current train.py. See the module "
      "docstring.", file=sys.stderr)

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class _Stub(gym.Env):
    """Spaces-only env to construct the policy (no MAME)."""
    def __init__(self):
        self.observation_space = spaces.Box(0, 255, (84, 84, 1), np.uint8)
        self.action_space = spaces.Discrete(8)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, a):
        return self.observation_space.sample(), 0.0, True, False, {}


def stack4(frames):
    """Build (N,84,84,4) frame-stacks (oldest..newest), matching VecFrameStack."""
    n = len(frames)
    out = np.zeros((n, 84, 84, 4), np.uint8)
    for i in range(n):
        for k in range(4):
            out[i, :, :, k] = frames[max(0, i - (3 - k))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/bc_data.npz")
    ap.add_argument("--out", default="artifacts/ppo_dkong_bc")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    d = np.load(args.data)
    obs = stack4(d["frames"])                       # (N,84,84,4) uint8
    acts = d["actions"]                             # (N,)
    print(f"dataset: {len(obs)} samples")

    venv = VecFrameStack(DummyVecEnv([_Stub]), n_stack=4)
    model = PPO("CnnPolicy", venv, device="cuda", verbose=0)
    policy = model.policy
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    dev = model.device

    n = len(obs)
    acts_t = torch.as_tensor(acts, dtype=torch.long, device=dev)
    for ep in range(args.epochs):
        perm = np.random.permutation(n)
        tot, correct, nb = 0.0, 0, 0
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            o_t, _ = policy.obs_to_tensor(obs[idx])  # handles transpose+to-device
            a_t = acts_t[idx]
            dist = policy.get_distribution(o_t)
            logp = dist.log_prob(a_t)
            loss = -logp.mean()                      # == cross-entropy for categorical
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
            correct += int((dist.distribution.probs.argmax(1) == a_t).sum())
        print(f"epoch {ep:2d}: loss={tot/nb:.3f}  train_acc={correct/n:.3f}", flush=True)

    model.save(args.out)
    print(f"saved BC model -> {args.out}")


if __name__ == "__main__":
    main()
