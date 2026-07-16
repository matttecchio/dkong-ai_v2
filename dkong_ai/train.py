"""Train a PPO/RecurrentPPO agent on Donkey Kong via the MAME bridge.

    python -m dkong_ai.train --rom-dir /path/to/roms --timesteps 2000000
    python -m dkong_ai.train --rom-dir /path/to/roms --lstm --stack 2  # LSTM run
"""
import argparse
import json
import os
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
        self._glitch: list[int] = []         # episodes ended by the ladder guard
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
            self._glitch.append(int(info.get("glitch_kill", 0)))
            self._glitch = self._glitch[-self.window:]
            self.best_height = max(self.best_height, h)
            if info.get("start_type") == "curriculum":
                self._heights_cu.append(h)
                self._heights_cu = self._heights_cu[-self.window:]
            elif info.get("no_barrels"):
                # Barrel-free bottom episodes are trivial climbs; counting
                # them here is how clear_rate_bottomup faked 0.04-0.14 in
                # run 27g while live-barrel evals measured 0/425.
                pass
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
        if self._glitch:
            # Fraction of recent episodes ended by the broken-ladder guard.
            # Should DECAY as the policy unlearns the x=99 exploit.
            self.logger.record("climb/glitch_kill_rate",
                               sum(self._glitch) / len(self._glitch))
        return True


class SILCallback(BaseCallback):
    """Self-imitation on the policy's OWN successes (Oh et al. 2018, adapted).

    PPO is on-policy: a clear from a 2% frontier cell gets three epochs of
    gradient and is discarded — the project's scarcest resource, thrown away.
    This callback keeps a buffer of successful episodes' (obs, exec_action)
    sequences and periodically adds an imitation loss on them, replaying wins
    until absorbed. Unlike the run-5 init-BC (brittle, expert-distribution
    mismatch), these are the policy's own on-distribution trajectories.

    Success = honest clear, or +40px gain from a curriculum spawn (matches
    the env's success-recording criteria). Imitates `exec_action` (what the
    env RAN — forced approach/burn-in steps override the agent's choice).
    LSTM-correct: each episode is evaluated as one sequence with
    episode_start=[1,0,...] and zero initial state — exactly the semantics
    of an episode boundary."""

    def __init__(self, coef=0.1, buffer_eps=40, eps_per_update=2,
                 max_ep_len=600):
        super().__init__()
        self.coef = coef
        self.buffer_eps = buffer_eps
        self.eps_per_update = eps_per_update
        self.max_ep_len = max_ep_len
        self._acc: list[list] = []      # per-env (obs, act) accumulator
        # STRATIFIED buffers (2026-07-14): a single FIFO let routine tower
        # rehearsal wins (arriving ~constantly) evict a rare floor crossing
        # within minutes — the flywheel never compounded for exactly the
        # successes that need it most. Classes get reserved space and equal
        # sampling weight.
        self._bufs: dict[str, list] = {"floor": [], "clear_bottomup": [],
                                       "clear": []}
        self._last_obs = None
        self._updates = 0

    def _on_training_start(self) -> None:
        self._acc = [[] for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        import numpy as _np
        obs = self.locals.get("obs_tensor")     # PRE-step obs (torch, on-dev)
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones")
        if obs is None or dones is None:
            return True
        for i in range(len(dones)):
            info = infos[i] if i < len(infos) else {}
            if len(self._acc[i]) < self.max_ep_len:
                self._acc[i].append((
                    {k: v[i].detach().cpu().numpy().copy()
                     for k, v in obs.items()},
                    int(info.get("exec_action", 0))))
            if dones[i]:
                ep = info.get("episode")
                start_h = max(0, 240 - info.get("start_y", 240))
                gained = info.get("max_height", 0) - start_h
                honest = (not info.get("no_barrels")
                          and not info.get("glitch_kill"))
                cls = None
                if honest and ep is not None:
                    if info.get("cleared"):
                        cls = ("clear_bottomup"
                               if info.get("start_type") == "bottomup"
                               else "clear")
                    elif (info.get("start_type") == "curriculum"
                          and start_h < 45
                          and (gained >= 40 or info.get("max_height", 0) >= 68)):
                        cls = "floor"       # crossing + wait-spot ladder wins
                                            # + waterfall passages (h>=68 from
                                            # low starts pass at gain ~30)
                                            # (<45 covers the x53 wait-spot
                                            # starts at h37-38; was <30 which
                                            # silently excluded them)
                    elif (info.get("start_type") == "curriculum"
                          and gained >= 40):
                        cls = "clear"       # generic curric progress: pool
                                            # with routine tower successes
                if cls and 4 < len(self._acc[i]):
                    per_class = max(4, self.buffer_eps // len(self._bufs))
                    self._bufs[cls].append(self._acc[i])
                    self._bufs[cls] = self._bufs[cls][-per_class:]
                self._acc[i] = []
        return True

    def _on_rollout_end(self) -> None:
        import numpy as _np
        import torch as _th
        for cls, buf in self._bufs.items():
            self.logger.record(f"sil/buffer_{cls}", len(buf))
        nonempty = [b for b in self._bufs.values() if b]
        if not nonempty:
            return
        policy = self.model.policy
        # _on_rollout_end runs straight after collection, where SB3 leaves
        # the policy in eval mode — cudnn refuses RNN backward there
        # ("cudnn RNN backward can only be called in training mode",
        # run-29 launch crash). PPO's own train() flips it back anyway.
        policy.set_training_mode(True)
        rng = _np.random.default_rng()
        losses = []
        # try/finally restores eval mode even if an update raises
        # (review r11 #6; PPO self-heals, but don't rely on it)
        try:
            for _ in range(min(self.eps_per_update, sum(len(b) for b in nonempty))):
                # Equal weight per CLASS, then uniform within: a floor crossing
                # gets replayed as often as the whole tower-clear pool does.
                buf = nonempty[int(rng.integers(len(nonempty)))]
                if not buf:
                    continue          # r15: theoretical clear-during-update
                ep = buf[int(rng.integers(len(buf)))]
                T = len(ep)
                obs = {k: _th.as_tensor(_np.stack([t[0][k] for t in ep]),
                                        device=policy.device)
                       for k in ep[0][0]}
                acts = _th.as_tensor(_np.array([t[1] for t in ep]),
                                     device=policy.device)
                starts = _th.zeros(T, device=policy.device)
                starts[0] = 1.0
                shp = (policy.lstm_actor.num_layers, 1,
                       policy.lstm_actor.hidden_size)
                zeros = (_th.zeros(shp, device=policy.device),
                         _th.zeros(shp, device=policy.device))
                from sb3_contrib.common.recurrent.type_aliases import RNNStates
                lstm_states = RNNStates(zeros, zeros)
                _, log_prob, _ = policy.evaluate_actions(
                    obs, acts, lstm_states, starts)
                loss = -self.coef * log_prob.mean()
                policy.optimizer.zero_grad()
                loss.backward()
                _th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                policy.optimizer.step()
                losses.append(float(loss.item()))
        finally:
            policy.set_training_mode(False)
        self._updates += 1
        self.logger.record("sil/updates", self._updates)
        self.logger.record("sil/loss", sum(losses) / len(losses))


class BackwardCallback(BaseCallback):
    """Backward-algorithm walk-back (Go-Explore phase 2), per-chain.

    Episodes flagged start_type=="curriculum" begin from winner-chain states
    near the goal. Each chain advances INDEPENDENTLY: when the rolling clear
    rate of a chain's frontier tier (its deepest allowed cell, which gets 50%
    of that chain's draws) reaches `threshold`, that chain's starts move one
    cell deeper. One awkward cell stalls only its own chain; the walk-back
    flows down the easiest route first — one route to the bottom is enough.
    The start window is [n-1-level, n-1], so mastered tiers keep rehearsing."""

    PROGRESS_PX = 40   # success bar for progress-gated (floor) chains

    def __init__(self, n_chains, window=64, threshold=0.5, gates=None):
        super().__init__()
        self.n_chains = n_chains
        self.window = window
        # Per-chain gate type: "clear" (reach Pauline) or "progress" (gain
        # PROGRESS_PX from spawn). Progress gates make LOW cells usable as
        # curriculum: a clear-gate from h20 is absurd — the walk-back could
        # never advance — but "+40px" is exactly the traffic-crossing +
        # first-ladder skill the floor policy lacks (run 28d: honest floor
        # = 0 after the stub-glitch guard fix; the poverty trap won't
        # dissolve from bottom-up episodes alone).
        self.gates = gates or ["clear"] * n_chains
        self.chain_window = max(16, window // 4)
        # Rehearsal drives the consolidation governor; at window=64 its
        # easy/hard tier mix swings the rate +/-0.1 by draw luck alone and
        # the governor flaps (freeze/resume with no promotion in between,
        # observed run 27l). 4x window steadies it without touching the
        # frontier jury size.
        self.rehearsal_window = window * 4
        self.threshold = threshold
        self.levels = [0] * n_chains
        self._results: list[int] = []
        self._frontier: list[list[int]] = [[] for _ in range(n_chains)]
        self._rehearsal: list[int] = []   # non-frontier curriculum episodes:
                                          # consolidation of PROMOTED tiers
        # Consolidation with hysteresis: while the rehearsal clear rate sags
        # below CONSOL_ON, promotions FREEZE (frontiers keep drilling, tiers
        # keep rehearsing) until it recovers past CONSOL_OFF. Run 27k showed
        # why: rehearsal slid 0.92->0.64 over 3M steps as the tower grew
        # faster than it hardened.
        # Calibration (run 27m): with ~37 promoted tiers the rehearsal rate
        # sits at a STATIONARY ~0.70 — DK's stochasticity ceiling, not decay.
        # Thresholds must sit BELOW that equilibrium or the governor flaps
        # (freeze<->resume with no promotions between, observed at 0.65/0.75).
        # Fire only on real decay.
        # Recalibrated 0.60/0.68 -> 0.40/0.48 (run 27s): once the tower
        # contains the hard h160-178 band, each promotion enters rehearsal at
        # its ~0.3 gate rate and the pooled equilibrium drops to ~0.47-0.57 —
        # 0.60/0.68 froze promotions for 12M+ steps on composition, not decay
        # (per-cell CSV audit showed every tier RISING while frozen). The
        # equilibrium moves with tower difficulty; thresholds must track it.
        self.CONSOL_ON, self.CONSOL_OFF = 0.40, 0.48
        self._consolidating = False
        self.levels_path: str | None = None   # set by main(): persistence

    def _on_training_start(self) -> None:
        # Resume walk-back levels across restarts — every restart used to
        # re-burn hours of frontier grind. Delete the file to start over.
        if self.levels_path and os.path.exists(self.levels_path):
            with open(self.levels_path) as f:
                saved = json.load(f).get("levels", [])
            if len(saved) == self.n_chains:
                self.levels = [int(x) for x in saved]
                self.training_env.env_method("set_backward_levels", self.levels)
                print(f"[backward] resumed levels {self.levels}", flush=True)
            else:
                print(f"[backward] levels file chain count mismatch "
                      f"({len(saved)} != {self.n_chains}) — starting at 0",
                      flush=True)

    def _save_levels(self):
        if self.levels_path:
            with open(self.levels_path, "w") as f:
                json.dump({"levels": self.levels}, f)

    def _on_step(self) -> bool:
        pushed = False
        for info in self.locals.get("infos", []):
            if info.get("episode") is None:
                continue
            if info.get("start_type") != "curriculum":
                continue
            if info.get("no_barrels"):
                # Barrel-frozen episodes are far easier; letting them into the
                # gate would advance levels the live-barrel policy hasn't
                # earned. (Current runs use --p-no-barrels 0.0, so this is a
                # guard for configs that re-enable freeze episodes.)
                continue
            bw = info.get("bw_start")
            if not bw:
                continue
            ci, pos, n, _h = bw
            if self.gates[ci] == "progress":
                start_h = max(0, 240 - info.get("start_y", 240))
                success = int(info.get("max_height", 0) - start_h
                              >= self.PROGRESS_PX)
            else:
                success = info.get("cleared", 0)
            self._results.append(success)
            if pos != max(0, n - 1 - self.levels[ci]):
                # Rehearsal draw from an already-promoted tier. Its rolling
                # clear rate is the consolidation signal: rising = the tower
                # keeps hardening behind the frontier; sagging = pause the
                # walk-back and train in place before advancing further.
                self._rehearsal.append(success)
                self._rehearsal = self._rehearsal[-self.rehearsal_window:]
                continue
            f = self._frontier[ci]
            f.append(success)
            del f[:-self.chain_window]
            if (not self._consolidating
                    and len(f) >= self.chain_window
                    and sum(f) / len(f) >= self.threshold):
                self.levels[ci] += 1
                self._frontier[ci] = []       # new tier, fresh jury
                pushed = True
                print(f"[backward] chain {ci} -> level {self.levels[ci]} "
                      f"(frontier clear rate {sum(f) / len(f):.2f})",
                      flush=True)
        if pushed:
            self.training_env.env_method("set_backward_levels", self.levels)
            self._save_levels()
        if len(self._rehearsal) >= self.rehearsal_window:
            rate = sum(self._rehearsal) / len(self._rehearsal)
            if not self._consolidating and rate < self.CONSOL_ON:
                self._consolidating = True
                print(f"[backward] CONSOLIDATING: rehearsal {rate:.2f} < "
                      f"{self.CONSOL_ON} — promotions frozen until it "
                      f"recovers past {self.CONSOL_OFF}", flush=True)
            elif self._consolidating and rate > self.CONSOL_OFF:
                self._consolidating = False
                print(f"[backward] consolidation done: rehearsal {rate:.2f} "
                      f"> {self.CONSOL_OFF} — promotions resume", flush=True)
        self.logger.record("climb/backward_consolidating",
                           int(self._consolidating))
        self._results = self._results[-self.window:]
        if self._results:
            self.logger.record("climb/backward_clear_rate",
                               sum(self._results) / len(self._results))
        rates = [sum(f) / len(f) for f in self._frontier if f]
        if rates:
            self.logger.record("climb/backward_clear_frontier",
                               sum(rates) / len(rates))
        if self._rehearsal:
            self.logger.record("climb/backward_clear_rehearsal",
                               sum(self._rehearsal) / len(self._rehearsal))
        self.logger.record("climb/backward_level",
                           sum(self.levels) / len(self.levels))
        self.logger.record("climb/backward_level_max", max(self.levels))
        return True


def make_env(rom_dir, port, frameskip, backward_manifest=None,
             p_no_barrels=None, p_curric=None, gamma=None):
    def _thunk():
        # record=False -> fast save-state resets (no per-episode .inp; use eval.py
        # with recording for watchable playback of a trained policy).
        env = DonkeyKongEnv(rom_dir=rom_dir, port=port, frameskip=frameskip,
                            record=False, backward_manifest=backward_manifest)
        # Set per-instance INSIDE the thunk: it runs in the worker process.
        # Mutating DonkeyKongEnv class attrs in main() looks equivalent but is
        # silently undone by SubprocVecEnv's spawn start method (workers
        # re-import the module, reverting to class defaults) — that bug ran
        # the whole 27 series at 15% curriculum / 15% barrel-free instead of
        # the CLI values, and the barrel-free bottom climbs faked
        # clear_rate_bottomup 0.04-0.14.
        if p_no_barrels is not None:
            env.P_NO_BARRELS = p_no_barrels
        if p_curric is not None:
            env._p_curric = p_curric
        if gamma is not None:
            # PBRS policy-invariance requires the SAME gamma in shaping and
            # RL update (review r9): bind the env's shaping gamma to the
            # training gamma. Instance attr inside the thunk — the spawn
            # gotcha above applies to this exactly as to _p_curric.
            env.PBRS_GAMMA = gamma
        # Per-episode CSV: ground truth for auditing the aggregate metrics.
        # clear_rate_bottomup rose while every controlled bottom-start eval
        # scored 0/425 — these rows are how we catch a phantom clear in the
        # act (start_y/start_screen say where the episode REALLY began).
        os.makedirs("logs/episodes", exist_ok=True)
        return Monitor(env, filename=f"logs/episodes/dk_{port}",
                       info_keywords=("max_height", "cleared", "start_type",
                                      "start_y", "start_screen", "end_screen",
                                      "bw_pos", "bw_chain", "no_barrels",
                                      "glitch_kill", "burnin", "approach_len",
                                      "bw_fallback_chain",
                                      "difficulty_start", "difficulty_end"))
    return _thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--timesteps", type=int, default=10_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--base-port", type=int, default=5000)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--stack", type=int, default=2,
                    help="frame stack depth (run 21+ canonical: 2 — matches "
                         "eval.py and scripts/current_launch.sh)")
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
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="torch device (auto = cuda if available). Hardcoded "
                         "cuda used to make recovery on a GPU-less host fail "
                         "confusingly.")
    ap.add_argument("--sil-coef", type=float, default=0.1,
                    help="self-imitation aux-loss coefficient on the policy's "
                         "own successful episodes (0 disables; LSTM runs only)")
    args = ap.parse_args()

    import torch
    dev = (("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else args.device)

    # NOTE: do NOT set these as DonkeyKongEnv class attributes here — spawn
    # workers re-import the module and revert them. They travel to the
    # workers as make_env parameters instead.
    bw_manifest = None
    if args.backward_dir:
        import os as _os
        bw_manifest = _os.path.abspath(
            _os.path.join(args.backward_dir, "manifest.json"))
        print(f"backward curriculum: {bw_manifest}")

    # Run metadata beside the checkpoints: which code, dials, and curriculum
    # produced this run (post-hoc archaeology has reconstructed these from
    # session memory too many times — e.g. "which manifest was 28c on?").
    import hashlib
    import subprocess as _sp
    import time as _time
    try:
        _sha = _sp.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        _sha = "unknown"
    _mhash = "none"
    if bw_manifest and os.path.exists(bw_manifest):
        _mhash = hashlib.sha256(open(bw_manifest, "rb").read()).hexdigest()
    meta = {"ts": _time.strftime("%Y-%m-%d %H:%M:%S"), "git_sha": _sha,
            "manifest_sha256": _mhash, "device": dev, "args": vars(args)}
    with open(args.save + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[meta] git {_sha[:10]} manifest {_mhash[:10]} device {dev} "
          f"-> {args.save}_meta.json", flush=True)

    # Refuse to start on occupied bridge ports: the bridge is the socket
    # SERVER, so a second trainer's envs would silently connect to the FIRST
    # trainer's MAME instances and corrupt both runs' rollouts (2026-07-05:
    # an overlapped restart did exactly that for ~7 minutes).
    import socket as _socket
    for i in range(args.n_envs):
        with _socket.socket() as _s:
            if _s.connect_ex(("127.0.0.1", args.base_port + i)) == 0:
                raise SystemExit(
                    f"port {args.base_port + i} already has a listener — "
                    f"another trainer is (still) running; refusing to start")

    # One MAME instance per env, each on its own socket port.
    thunks = [make_env(args.rom_dir, args.base_port + i, args.frameskip,
                       bw_manifest, p_no_barrels=args.p_no_barrels,
                       p_curric=args.p_curric, gamma=args.gamma)
              for i in range(args.n_envs)]
    # Turn SIGTERM into a normal exception so the finally-block cleanup (which
    # shuts MAME down) runs on `kill <pid>`, not just on Ctrl-C / completion.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    venv = (SubprocVecEnv(thunks, start_method="spawn") if args.n_envs > 1
            else DummyVecEnv(thunks))
    venv = DkFrameStackWrapper(venv, n_stack=args.stack)

    # Namespace checkpoints by the run's save name so parallel/sequential runs
    # don't overwrite each other's checkpoints. (No local `import os` here:
    # it made `os` main()-local and blew up every EARLIER os.* reference —
    # the run-29 startup crash.)
    run_name = os.path.basename(args.save)
    # Checkpoint every ~500k steps (60 files / 30M run ~= 1.2GB), not 50k (~12GB).
    ckpt = CheckpointCallback(save_freq=max(500_000 // args.n_envs, 1),
                              save_path=os.path.join("artifacts/checkpoints", run_name),
                              name_prefix=run_name)
    callbacks = [ckpt, ClimbMetricsCallback()]
    if args.sil_coef > 0 and args.lstm:
        callbacks.append(SILCallback(coef=args.sil_coef))
        print(f"[sil] self-imitation on successes, coef={args.sil_coef}")
    if args.backward_dir:
        import json as _json
        with open(bw_manifest) as _f:
            _chains = _json.load(_f)["chains"]
        n_chains = len(_chains)
        if n_chains == 0:
            # Mirror the env's behavior (it disables the curriculum with a
            # warning); an empty levels list would crash the callback's
            # mean/max logging on the first step.
            print("WARNING: backward manifest has no chains — "
                  "BackwardCallback disabled")
        else:
            gates = [ch.get("gate", "clear") for ch in _chains]
            if any(g == "progress" for g in gates):
                print(f"[backward] progress-gated chains: "
                      f"{[i for i, g in enumerate(gates) if g == 'progress']}")
            bw_cb = BackwardCallback(n_chains, window=args.bw_window,
                                     threshold=args.bw_threshold, gates=gates)
            # Persist walk-back levels next to the manifest so restarts
            # resume the descent instead of re-earning it.
            bw_cb.levels_path = os.path.join(args.backward_dir,
                                             "levels.json")
            callbacks.append(bw_cb)
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
            model = AlgoClass.load(args.init_from, env=venv, device=dev,
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
                tensorboard_log=args.logdir, verbose=1, device=dev,
            )
            old = AlgoClass.load(args.init_from, device=dev)
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
            tensorboard_log=args.logdir, verbose=1, device=dev,
        )
        if args.transfer_features_from:
            # Copy features_extractor weights (CNN + RAM MLP) from any saved
            # model into the fresh model, shape-matched layer by layer.
            # Everything else (LSTM, heads) stays randomly initialised.
            print(f"transferring features_extractor weights from {args.transfer_features_from}")
            # Read the checkpoint's raw state dict WITHOUT constructing the
            # source model: AlgoClass.load() rebuilds the policy from the
            # CURRENT DkFeaturesExtractor/constants, so after any capacity
            # change the old weights no longer fit their own skeleton and
            # load() raises (hit at run 28: RAM_HIDDEN 64->128 made
            # run27_last unloadable). The zip's params need no skeleton.
            from stable_baselines3.common.save_util import load_from_zip_file
            _, _params, _ = load_from_zip_file(
                args.transfer_features_from, device=dev)
            src_sd = _params["policy"]
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
