"""Hybrid CNN + RAM-MLP feature extractor and frame-stack wrapper for DK.

The wall the pure-CNN agent couldn't crack: timing barrel jumps while traversing
left on the 2nd girder. Barrels approaching from behind on a blurry 84×84 frame
are hard to react to; explicit relative positions from RAM are not.

Architecture:
  - NatureCNN(n_stack×2, 84, 84) → 256-dim image features  (spatial/structural)
  - Linear(102→128→128)           → 128-dim RAM features    (barrel/fireball/hammer
                                     positions/velocities + crazy/blue type flags
                                     + difficulty regime)
  - Concat(256+128=384)          → PPO/RecurrentPPO policy/value heads

RAM_HIDDEN 64→128 (2026-07-10, run 28): interference reduction. Small shared
nets are why passed curriculum tiers eroded 43→5% while training drilled
adjacent cells — the run-27 walk-back stalled on a hollow tower, not a hard
frontier. Paired with LSTM 256→512 (--lstm-hidden).

DkFrameStackWrapper stacks the 'image' channel n times (matching SB3's
VecFrameStack behaviour) while passing the 'ram' vector from the latest frame
only — threat positions don't need history, the CNN does.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.vec_env import VecEnvWrapper

CNN_OUT    = 256
RAM_HIDDEN = 128


class DkFeaturesExtractor(BaseFeaturesExtractor):
    """Parallel CNN + RAM-MLP streams, concatenated for the policy/value heads."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=CNN_OUT + RAM_HIDDEN)
        self.cnn = NatureCNN(observation_space["image"], features_dim=CNN_OUT)
        ram_dim = int(np.prod(observation_space["ram"].shape))
        self.ram_mlp = nn.Sequential(
            nn.Linear(ram_dim, RAM_HIDDEN),
            nn.ReLU(),
            nn.Linear(RAM_HIDDEN, RAM_HIDDEN),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.cnn(obs["image"]), self.ram_mlp(obs["ram"])], dim=1)


class DkFrameStackWrapper(VecEnvWrapper):
    """Stacks the 'image' channel n_stack times; 'ram' passes through unmodified.

    Replaces SB3's VecFrameStack for Dict observation spaces. On episode reset
    the image buffer is filled with zeros (matching VecFrameStack convention)."""

    def __init__(self, venv, n_stack: int = 4):
        self.n_stack = n_stack
        img_sp = venv.observation_space["image"]
        h, w, c = img_sp.shape
        stacked_sp = spaces.Box(0, 255, (h, w, c * n_stack), dtype=np.uint8)
        obs_sp = spaces.Dict({
            "image": stacked_sp,
            "ram":   venv.observation_space["ram"],
        })
        super().__init__(venv, observation_space=obs_sp)
        self._shape = (h, w, c)
        self._bufs: list[deque] | None = None

    def _make_buf(self) -> deque:
        h, w, c = self._shape
        return deque([np.zeros((h, w, c), dtype=np.uint8)] * self.n_stack,
                     maxlen=self.n_stack)

    def reset(self):
        obs = self.venv.reset()
        self._bufs = [self._make_buf() for _ in range(self.venv.num_envs)]
        for i, buf in enumerate(self._bufs):
            buf.append(obs["image"][i])
        return self._build(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i, done in enumerate(dones):
            if done:
                self._bufs[i] = self._make_buf()
            self._bufs[i].append(obs["image"][i])
        return self._build(obs), rews, dones, infos

    def _build(self, obs: dict) -> dict:
        stacked = np.stack(
            [np.concatenate(list(b), axis=-1) for b in self._bufs]
        )
        return {"image": stacked, "ram": obs["ram"]}
