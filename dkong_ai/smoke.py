"""End-to-end smoke test of the Gymnasium env.

    python -m dkong_ai.smoke --rom-dir ./roms

Resets (coin+start+spawn), runs random actions, prints obs shape, reward stats,
and a few decoded states; finally runs SB3's check_env.
"""
import argparse

import numpy as np

from .mame_env import DonkeyKongEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    env = DonkeyKongEnv(rom_dir=args.rom_dir)
    obs, info = env.reset()
    print(f"reset OK | obs {obs.shape} {obs.dtype} | start state {info['state']}")

    rng = np.random.default_rng(0)
    rewards, deaths = [], 0
    last_state = info["state"]
    for i in range(args.steps):
        a = int(rng.integers(env.action_space.n))
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        if info["state"]["lives"] < last_state["lives"]:
            deaths += 1
        last_state = info["state"]
        if i % 60 == 0 or term:
            print(f"  step {i:3d} a={a} r={r:+.2f} term={term} state={info['state']}")
        if term:
            print(f"  -> terminated at step {i}")
            break

    print(f"\nsteps={len(rewards)} sum_r={sum(rewards):+.2f} "
          f"min={min(rewards):+.2f} max={max(rewards):+.2f} deaths_seen={deaths}")
    env.close()

    print("\n== check_env ==")
    from stable_baselines3.common.env_checker import check_env
    env2 = DonkeyKongEnv(rom_dir=args.rom_dir)
    check_env(env2, warn=True, skip_render_check=True)
    env2.close()
    print("check_env passed")


if __name__ == "__main__":
    main()
