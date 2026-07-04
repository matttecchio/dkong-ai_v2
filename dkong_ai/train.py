"""Train a PPO/RecurrentPPO agent on Donkey Kong via the MAME bridge.

    python -m dkong_ai.train --rom-dir /path/to/roms --timesteps 2000000
    python -m dkong_ai.train --rom-dir /path/to/roms --lstm --stack 2  # LSTM run
"""
import argparse
import signal

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import get_schedule_fn

from .mame_env import DonkeyKongEnv
from .dk_policy import DkFeaturesExtractor, DkFrameStackWrapper


class ClimbMetricsCallback(BaseCallback):
    """Logs barrel-stage progress: mean/peak height, clear rate, score, and
    height segmented by episode start type (bottomup vs curriculum)."""

    def __init__(self, window=100):
        super().__init__()
        self._heights, self._clears, self._scores = [], [], []
        self._heights_bt: list[float] = []   # bottomup episodes only
        self._heights_cu: list[float] = []   # curriculum episodes only
        self._clears_bt: list[int] = []      # honest bottom-up clear signal
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
            state = info.get("state", {})
            score = state.get("score") or 0
            self._scores.append(score)
            self._heights = self._heights[-self.window:]
            self._clears  = self._clears[-self.window:]
            self._scores  = self._scores[-self.window:]
            self.best_height = max(self.best_height, h)
            if info.get("start_type") == "curriculum":
                self._heights_cu.append(h)
                self._heights_cu = self._heights_cu[-self.window:]
            else:
                self._heights_bt.append(h)
                self._heights_bt = self._heights_bt[-self.window:]
                self._clears_bt.append(info.get("cleared", 0))
                self._clears_bt = self._clears_bt[-self.window:]
        if self._heights:
            self.logger.record("climb/height_mean", sum(self._heights) / len(self._heights))
            self.logger.record("climb/height_best", self.best_height)
            self.logger.record("climb/clear_rate",  sum(self._clears) / len(self._clears))
            self.logger.record("climb/score_mean",  sum(self._scores) / len(self._scores))
        if self._heights_bt:
            self.logger.record("climb/height_mean_bottomup",
                               sum(self._heights_bt) / len(self._heights_bt))
        if self._heights_cu:
            self.logger.record("climb/height_mean_curric",
                               sum(self._heights_cu) / len(self._heights_cu))
        if self._clears_bt:
            self.logger.record("climb/clear_rate_bottomup",
                               sum(self._clears_bt) / len(self._clears_bt))
        return True


class BackwardCallback(BaseCallback):
    """Backward-algorithm walk-back (Go-Explore phase 2), per-chain.

    Episodes flagged start_type=="curriculum" begin from winner-chain states
    near the goal. Each chain advances INDEPENDENTLY: when the rolling clear
    rate of a chain's frontier tier (its deepest allowed cell, which gets 50%
    of that chain's draws) reaches `threshold`, that chain's starts move one
    cell deeper. One awkward cell stalls only its own chain; the walk-back
    flows down the easiest route first — one route to the bottom is enough.
    The start window is [n-1-level, n-1], so mastered tiers keep rehearsing."""

    def __init__(self, n_chains, window=64, threshold=0.5):
        super().__init__()
        self.n_chains = n_chains
        self.window = window
        self.chain_window = max(16, window // 4)
        self.threshold = threshold
        self.levels = [0] * n_chains
        self._results: list[int] = []
        self._frontier: list[list[int]] = [[] for _ in range(n_chains)]

    def _on_step(self) -> bool:
        pushed = False
        for info in self.locals.get("infos", []):
            if info.get("episode") is None:
                continue
            if info.get("start_type") != "curriculum":
                continue
            cleared = info.get("cleared", 0)
            self._results.append(cleared)
            bw = info.get("bw_start")
            if not bw:
                continue
            ci, pos, n, _h = bw
            if pos != max(0, n - 1 - self.levels[ci]):
                continue                      # rehearsal draw, not frontier
            f = self._frontier[ci]
            f.append(cleared)
            del f[:-self.chain_window]
            if (len(f) >= self.chain_window
                    and sum(f) / len(f) >= self.threshold):
                self.levels[ci] += 1
                self._frontier[ci] = []       # new tier, fresh jury
                pushed = True
                print(f"[backward] chain {ci} -> level {self.levels[ci]} "
                      f"(frontier clear rate {sum(f) / len(f):.2f})",
                      flush=True)
        if pushed:
            self.training_env.env_method("set_backward_levels", self.levels)
        if self._results:
            self.logger.record("climb/backward_clear_rate",
                               sum(self._results) / len(self._results))
            self._results = self._results[-self.window:]
        rates = [sum(f) / len(f) for f in self._frontier if f]
        if rates:
            self.logger.record("climb/backward_clear_frontier",
                               sum(rates) / len(rates))
        self.logger.record("climb/backward_level",
                           sum(self.levels) / len(self.levels))
        self.logger.record("climb/backward_level_max", max(self.levels))
        return True


def make_env(rom_dir, port, frameskip, backward_manifest=None):
    def _thunk():
        # record=False -> fast save-state resets (no per-episode .inp; use eval.py
        # with recording for watchable playback of a trained policy).
        env = DonkeyKongEnv(rom_dir=rom_dir, port=port, frameskip=frameskip,
                            record=False, backward_manifest=backward_manifest)
        return Monitor(env, info_keywords=("max_height", "cleared", "start_type"))
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
    ap.add_argument("--lr", type=float, default=2.5e-4,
                    help="PPO learning rate (default 2.5e-4; reduce for fine-tuning stable policy)")
    ap.add_argument("--gamma", type=float, default=0.999,
                    help="discount factor (0.999 makes clear-reward visible at episode start)")
    ap.add_argument("--p-no-barrels", type=float, default=None,
                    help="fraction of episodes with barrels disabled (default: env class value 0.15)")
    ap.add_argument("--p-curric", type=float, default=None,
                    help="fraction of episodes using curriculum start states (default: env class value 0.15)")
    ap.add_argument("--lstm", action="store_true",
                    help="use RecurrentPPO (LSTM) instead of PPO")
    ap.add_argument("--lstm-hidden", type=int, default=256,
                    help="LSTM hidden size (default 256)")
    ap.add_argument("--transfer-features-from", default=None,
                    help="copy features_extractor weights (CNN+RAM MLP) from this "
                         "checkpoint into a fresh model; rest is randomly initialised")
    ap.add_argument("--n-epochs", type=int, default=4,
                    help="PPO epochs per rollout (default 4; reduce to 3 to lower clip_fraction)")
    ap.add_argument("--backward-dir", default=None,
                    help="Go-Explore phase-2: dir with manifest.json + winner-chain "
                         ".sta states (from export_chains); curriculum episodes then "
                         "start from chain cells and walk back as clear rate rises")
    ap.add_argument("--bw-window", type=int, default=64,
                    help="curriculum episodes per walk-back decision")
    ap.add_argument("--bw-threshold", type=float, default=0.5,
                    help="clear rate needed to walk the start back one cell")
    args = ap.parse_args()

    if args.p_no_barrels is not None:
        DonkeyKongEnv.P_NO_BARRELS = args.p_no_barrels
    if args.p_curric is not None:
        DonkeyKongEnv._p_curric = args.p_curric

    bw_manifest = None
    if args.backward_dir:
        import os as _os
        bw_manifest = _os.path.abspath(
            _os.path.join(args.backward_dir, "manifest.json"))
        print(f"backward curriculum: {bw_manifest}")

    # One MAME instance per env, each on its own socket port.
    thunks = [make_env(args.rom_dir, args.base_port + i, args.frameskip,
                       bw_manifest)
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
    if args.backward_dir:
        import json as _json
        with open(bw_manifest) as _f:
            n_chains = len(_json.load(_f)["chains"])
        callbacks.append(BackwardCallback(n_chains, window=args.bw_window,
                                          threshold=args.bw_threshold))
    if args.lstm:
        from sb3_contrib import RecurrentPPO
        AlgoClass   = RecurrentPPO
        policy_name = "MultiInputLstmPolicy"
        policy_kwargs = {
            "features_extractor_class":  DkFeaturesExtractor,
            "features_extractor_kwargs": {},
            "lstm_hidden_size": args.lstm_hidden,
            "n_lstm_layers": 1,
            "shared_lstm": True,
            "enable_critic_lstm": False,
        }
    else:
        AlgoClass   = PPO
        policy_name = "MultiInputPolicy"
        policy_kwargs = {
            "features_extractor_class":  DkFeaturesExtractor,
            "features_extractor_kwargs": {},
        }

    if args.init_from:
        print(f"warm-starting from {args.init_from}")
        try:
            model = AlgoClass.load(args.init_from, env=venv, device="cuda",
                                   tensorboard_log=args.logdir)
            model.verbose = 1
            model.ent_coef = args.ent_coef
            model.gamma = args.gamma
            model.learning_rate = args.lr
            model.lr_schedule = get_schedule_fn(args.lr)
            model.n_epochs = args.n_epochs
        except ValueError as e:
            if "Observation spaces do not match" not in str(e):
                raise
            # Obs space changed. Partial load: copy matching-shape layers only.
            print(f"obs space mismatch — partial load (CNN+heads preserved, RAM MLP reinit)")
            model = AlgoClass(
                policy_name, venv,
                policy_kwargs=policy_kwargs,
                n_steps=512, batch_size=256, n_epochs=args.n_epochs,
                learning_rate=args.lr, gamma=args.gamma, gae_lambda=0.95,
                clip_range=0.1, ent_coef=args.ent_coef,
                tensorboard_log=args.logdir, verbose=1, device="cuda",
            )
            old = AlgoClass.load(args.init_from, device="cuda")
            old_sd = old.policy.state_dict()
            new_sd = model.policy.state_dict()
            filtered = {k: v for k, v in old_sd.items()
                        if k in new_sd and v.shape == new_sd[k].shape}
            skipped = [k for k in old_sd if k not in filtered]
            model.policy.load_state_dict(filtered, strict=False)
            print(f"  partial load: {len(filtered)}/{len(old_sd)} layers copied; "
                  f"skipped (shape mismatch): {skipped}")
    else:
        model = AlgoClass(
            policy_name, venv,
            policy_kwargs=policy_kwargs,
            n_steps=512, batch_size=256, n_epochs=args.n_epochs,
            learning_rate=args.lr, gamma=args.gamma, gae_lambda=0.95,
            clip_range=0.1, ent_coef=args.ent_coef,
            tensorboard_log=args.logdir, verbose=1, device="cuda",
        )
        if args.transfer_features_from:
            # Copy features_extractor weights (CNN + RAM MLP) from any saved
            # model into the fresh model. Everything else (LSTM, heads) stays
            # randomly initialised. Works across PPO→RecurrentPPO changes and
            # from RecurrentPPO→RecurrentPPO (tries RecurrentPPO first).
            print(f"transferring features_extractor weights from {args.transfer_features_from}")
            try:
                src = AlgoClass.load(args.transfer_features_from, device="cuda")
            except Exception:
                src = PPO.load(args.transfer_features_from, device="cuda")
            src_sd = src.policy.state_dict()
            tgt_sd = model.policy.state_dict()
            transferred = {k: v for k, v in src_sd.items()
                           if k.startswith("features_extractor.")
                           and k in tgt_sd and v.shape == tgt_sd[k].shape}
            model.policy.load_state_dict(transferred, strict=False)
            print(f"  transferred {len(transferred)} features_extractor layers")
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
