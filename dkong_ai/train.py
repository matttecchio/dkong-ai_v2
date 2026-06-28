"""Train a PPO agent on Donkey Kong via the MAME bridge.

    python -m dkong_ai.train --rom-dir /path/to/roms --timesteps 2000000

Starts simple (single env, frame-stacked CNN policy). Scale to vectorized
parallel MAME instances (SubprocVecEnv on distinct ports) once the single-env
loop is confirmed working end-to-end.
"""
import argparse
import signal

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from .mame_env import DonkeyKongEnv
from .dk_policy import DkFeaturesExtractor, DkFrameStackWrapper


class ClimbMetricsCallback(BaseCallback):
    """Logs barrel-stage progress so we can SEE the agent climbing higher:
    mean/peak height reached and the stage-clear rate over recent episodes."""

    def __init__(self, window=100):
        super().__init__()
        self._heights, self._clears = [], []
        self.window = window
        self.best_height = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            h = info.get("max_height", 0)
            self._heights.append(h)
            self._clears.append(info.get("cleared", 0))
            self._heights = self._heights[-self.window:]
            self._clears = self._clears[-self.window:]
            self.best_height = max(self.best_height, h)
        if self._heights:
            self.logger.record("climb/height_mean", sum(self._heights) / len(self._heights))
            self.logger.record("climb/height_best", self.best_height)
            self.logger.record("climb/clear_rate", sum(self._clears) / len(self._clears))
        return True


def make_env(rom_dir, port, frameskip):
    def _thunk():
        # record=False -> fast save-state resets (no per-episode .inp; use eval.py
        # with recording for watchable playback of a trained policy).
        env = DonkeyKongEnv(rom_dir=rom_dir, port=port, frameskip=frameskip,
                            record=False)
        return Monitor(env, info_keywords=("max_height", "cleared"))
    return _thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--timesteps", type=int, default=10_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--base-port", type=int, default=5000)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--stack", type=int, default=4)
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--save", default="artifacts/ppo_dkong")
    ap.add_argument("--init-from", default=None,
                    help="warm-start policy weights from this saved model")
    ap.add_argument("--ent-coef", type=float, default=0.01,
                    help="PPO entropy coefficient (raise for more exploration)")
    ap.add_argument("--gamma", type=float, default=0.999,
                    help="discount factor (0.999 makes clear-reward visible at episode start)")
    args = ap.parse_args()

    # One MAME instance per env, each on its own socket port.
    thunks = [make_env(args.rom_dir, args.base_port + i, args.frameskip)
              for i in range(args.n_envs)]
    # Turn SIGTERM into a normal exception so the finally-block cleanup (which
    # shuts MAME down) runs on `kill <pid>`, not just on Ctrl-C / completion.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    venv = (SubprocVecEnv(thunks, start_method="spawn") if args.n_envs > 1
            else DummyVecEnv(thunks))
    venv = DkFrameStackWrapper(venv, n_stack=args.stack)

    # Namespace checkpoints by the run's save name so parallel/sequential runs
    # don't overwrite each other's checkpoints.
    import os
    run_name = os.path.basename(args.save)
    # Checkpoint every ~500k steps (60 files / 30M run ~= 1.2GB), not 50k (~12GB).
    ckpt = CheckpointCallback(save_freq=max(500_000 // args.n_envs, 1),
                              save_path=os.path.join("artifacts/checkpoints", run_name),
                              name_prefix=run_name)
    callbacks = [ckpt, ClimbMetricsCallback()]
    policy_kwargs = {
        "features_extractor_class":  DkFeaturesExtractor,
        "features_extractor_kwargs": {},
    }
    if args.init_from:
        print(f"warm-starting from {args.init_from}")
        model = PPO.load(args.init_from, env=venv, device="cuda",
                         tensorboard_log=args.logdir)
        model.verbose = 1
        model.ent_coef = args.ent_coef
        model.gamma = args.gamma
    else:
        model = PPO(
            "MultiInputPolicy", venv,
            policy_kwargs=policy_kwargs,
            n_steps=512, batch_size=256, n_epochs=4,
            learning_rate=2.5e-4, gamma=args.gamma, gae_lambda=0.95,
            clip_range=0.1, ent_coef=args.ent_coef,
            tensorboard_log=args.logdir, verbose=1, device="cuda",
        )
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=False,
                    callback=callbacks)
        model.save(args.save)
        print(f"saved -> {args.save}")
    except KeyboardInterrupt:
        print("interrupted — shutting down cleanly")
    finally:
        # Always close the vec env so each worker's env.close() runs (clean
        # ACT_QUIT -> MAME exits and finalizes its .inp). Without this, MAME
        # children orphan when the trainer exits. Save a recovery checkpoint
        # too in case we're unwinding from an error/interrupt.
        try:
            model.save(args.save + "_last")
        except Exception:
            pass
        venv.close()
        print("closed vec env (MAME instances shut down)")


if __name__ == "__main__":
    main()
