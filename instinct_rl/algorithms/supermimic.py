"""SuperMimic: motion tracking by backpropagating through a learned world model.

The algorithm replaces PPO's update step. It keeps PPO's interaction loop (collect with the policy,
store into a replay buffer) and then, instead of computing advantages from real returns, fits a
dynamics model on the replay and optimises the policy on *imagined* rollouts produced by that model.
A critic bootstraps the return past the imagination horizon, which is what makes long horizons stable.

Structure (paper: Liao et al., arXiv 2610.28480, Tables A.1 and A.5):

1. **Collect** ``num_steps_per_env`` steps with the deterministic policy plus Gaussian action noise,
   storing ``(state, context, action, reward, done, episode_step, ref_index)`` in a
   :class:`~instinct_rl.storage.replay_storage.ReplayStorage`.
2. **Fit the world model** on ``wm_batches_per_update`` windows of ``wm_horizon`` steps drawn from the
   replay (its own optimiser, gradient clipping and (optionally) gradient accumulation).
3. **Imagine** ``num_imagined`` rollouts of ``horizon`` steps with the world model, applying the
   policy at every step, rewards and terminations coming from the environment's imagination
   interface so the imagined MDP matches the simulated one exactly.
4. **Fit the critic** on the imagined states with ``lambda``-returns from the imagined rewards.
5. **Fit the policy** by maximising the imagined return, backpropagating *through* the world model,
   with the critic's bootstrapped value at the horizon.

Everything the algorithm needs from the task arrives through the
:class:`~instinct_rl.env.imagination.ImaginationInterface` protocol, so this class contains no task,
robot or reward specifics.
"""

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from instinct_rl.modules.world_model import WorldModel
from instinct_rl.storage.replay_storage import ReplayStorage, TensorSpec
from instinct_rl.utils.returns import compute_gae, normalize_advantages


class SuperMimic:
    """Model-based motion-tracking algorithm.

    Args:
        actor_critic: The policy (and critic) module. ``act``/``act_inference`` are used for actions,
            ``evaluate`` for values, ``get_actions_log_prob`` is optional.
        imagination: Object implementing ``ImaginationInterface``.
        world_model: Instance of :class:`~instinct_rl.modules.world_model.WorldModel` (or compatible).
        state_specs: ``{name: TensorSpec}`` describing one world-model state tensor. The concatenation
            of these names, in order, must equal the world model's ``state_segments``.
        context_specs: ``{name: TensorSpec}`` for the context vector (may be empty).
        extra_replay_specs: Additional tensors to store per step (e.g. ``ref_index`` as two int64s).
        replay_capacity: Steps kept per environment.
        horizon: Imagination horizon ``H``.
        wm_horizon: World-model fit horizon ``k``.
        gamma, lam: Discount factors.
        use_critic: If False, the policy bootstraps with zero (the critic ablation in the paper).
        collection_noise_std: Std of the Gaussian action noise used during collection.
        wm_batches_per_update, wm_batch_size: World-model fitting effort per iteration.
        num_imagined: Number of imagined rollouts per policy update.
        critic_epochs, critic_mini_batches: Critic fitting effort per iteration.
        updates_per_iteration: Policy/world-model/critic cycles per runner iteration.
        wm_warmup_iterations: Iterations of world-model-only training before the first policy update.
        policy_grad_value_clip: Element-wise clip applied to policy gradients after the global norm
            clip (the paper uses 0.1).
        learning_rate: Kept for the runner's logger; the effective rates live in the learners.
        save_replay_in_checkpoint: Include the replay buffers in ``state_dict()``.
    """

    #: Observation group and term name carrying the reference frame index.
    ref_index_group = "ref_index"
    ref_index_key = "ref_index"

    def __init__(
        self,
        actor_critic,
        imagination,
        world_model=None,
        state_specs: dict[str, TensorSpec] | None = None,
        context_specs: dict[str, TensorSpec] | None = None,
        extra_replay_specs: dict[str, TensorSpec] | None = None,
        replay_capacity: int = 1000,
        horizon: int = 32,
        wm_horizon: int = 8,
        gamma: float = 0.99,
        lam: float = 0.95,
        use_critic: bool = True,
        collection_noise_std: float = 0.1,
        wm_batches_per_update: int = 10,
        wm_batch_size: int = 2048,
        num_imagined: int = 4096,
        critic_epochs: int = 5,
        critic_mini_batches: int = 4,
        updates_per_iteration: int = 2,
        wm_warmup_iterations: int = 0,
        world_model_learning_rate: float = 2e-4,
        policy_learning_rate: float = 2e-4,
        critic_learning_rate: float = 2e-4,
        world_model_max_grad_norm: float = 1.0,
        policy_max_grad_norm: float = 1.0,
        policy_grad_value_clip: float | None = 0.1,
        policy_chunk_size: int = 1024,
        world_model_hidden_sizes: Sequence[int] = (2048, 2048, 2048),
        world_model_predict_delta: Sequence[str] | str = "none",
        critic_max_grad_norm: float = 1.0,
        learning_rate: float | None = None,
        save_replay_in_checkpoint: bool = True,
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"SuperMimic got unknown arguments: {sorted(kwargs)}")

        self.device = torch.device(device)
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.imagination = imagination
        # ``world_model`` may be built later: its action size is only known once the storage is
        # initialised, and its state layout comes from the imagination.
        self.world_model = None if world_model is None else world_model.to(self.device)
        self._world_model_cfg = dict(
            state_segments=imagination.state_format,
            hidden_sizes=world_model_hidden_sizes,
            predict_delta=world_model_predict_delta,
        )

        # The imagination owns the layout (it is built from the env's own observation groups), so when
        # specs are not given they are read from it rather than restated in config.
        if state_specs is None:
            # flattened, because ``process_env_step`` stores each component as (num_envs, prod(shape)):
            # the replay holds vectors, not the observation group's per-term shapes.
            state_specs = {
                name: TensorSpec((int(np.prod(shape)),)) for name, shape in imagination.state_format.items()
            }
        self.state_specs = dict(state_specs)
        self.context_specs = dict(context_specs or {})
        self.state_names = list(self.state_specs)
        self.context_names = list(self.context_specs)
        self.extra_replay_specs = dict(extra_replay_specs or {})

        self.horizon = int(horizon)
        self.wm_horizon = int(wm_horizon)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.use_critic = bool(use_critic)
        self.collection_noise_std = float(collection_noise_std)
        self.wm_batches_per_update = int(wm_batches_per_update)
        self.wm_batch_size = int(wm_batch_size)
        self.num_imagined = int(num_imagined)
        self.critic_epochs = int(critic_epochs)
        self.critic_mini_batches = int(critic_mini_batches)
        self.updates_per_iteration = int(updates_per_iteration)
        self.wm_warmup_iterations = int(wm_warmup_iterations)
        self.policy_grad_value_clip = policy_grad_value_clip
        self.policy_chunk_size = int(policy_chunk_size)
        self.save_replay_in_checkpoint = bool(save_replay_in_checkpoint)
        # The runner logs `alg.learning_rate`; report the policy rate as the headline number.
        self.learning_rate = float(policy_learning_rate if learning_rate is None else learning_rate)

        self.replay_capacity = int(replay_capacity)
        self.replay: ReplayStorage | None = None
        self.num_envs = 0
        self.num_actions = 0

        # State of the last imagined rollout, held so the policy pass can replay the same
        # trajectories with gradients. Declared here so a chunked update that runs before any
        # imagination fails loudly instead of reading a stale attribute.
        self._imagined_start_state: torch.Tensor | None = None
        self._imagined_start_context: torch.Tensor | None = None
        self._imagined_start_ref: torch.Tensor | None = None
        self._imagined_start_history: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        self._imagined_local_ref: torch.Tensor | None = None
        self._imagined_row_index: torch.Tensor | None = None
        self._bounds: tuple[torch.Tensor, torch.Tensor] | None = None

        self._world_model_learning_rate = float(world_model_learning_rate)
        self.world_model_optimizer = (
            None if self.world_model is None
            else torch.optim.Adam(self.world_model.parameters(), lr=self._world_model_learning_rate)
        )
        self.policy_optimizer = torch.optim.Adam(self.actor_critic.actor.parameters(), lr=policy_learning_rate)
        critic_params = list(self.actor_critic.critic.parameters())
        if not critic_params:
            raise ValueError("SuperMimic needs a critic network in the actor_critic (even if use_critic is False)")
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_learning_rate)

        self.world_model_max_grad_norm = world_model_max_grad_norm
        self.policy_max_grad_norm = policy_max_grad_norm
        self.critic_max_grad_norm = critic_max_grad_norm

        self.current_learning_iteration = 0
        self._noise_generator = torch.Generator(device="cpu").manual_seed(0)
        self._last_action = None

    # ------------------------------------------------------------------ runner interface

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        specs: dict[str, TensorSpec] = {}
        specs.update(self.state_specs)
        specs.update(self.context_specs)
        specs.update(self.extra_replay_specs)
        specs["action"] = TensorSpec((self.num_actions,))
        specs["reward"] = TensorSpec((int(num_rewards),))
        specs["done"] = TensorSpec((1,))
        specs["episode_step"] = TensorSpec((), torch.int64)
        # ``ref_index`` is not a policy input; it rides along so a rollout seeded from a replayed state
        # can start from the matching point in the clip.
        if self.ref_index_group in obs_format:
            shape = tuple(obs_format[self.ref_index_group][self.ref_index_key])
            self.extra_replay_specs.setdefault(self.ref_index_key, TensorSpec(shape, torch.float32))
            specs[self.ref_index_key] = self.extra_replay_specs[self.ref_index_key]
        self.replay = ReplayStorage(
            num_envs=self.num_envs,
            capacity_per_env=self.replay_capacity,
            specs=specs,
            episode_key="episode_step",
            device="cpu",
        )
        # on the policy device: compared against ``done``, which the runner has already moved there
        self._episode_step = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        if self.world_model is None:
            self.world_model = WorldModel(
                action_size=self.num_actions,
                context_size=sum(int(np.prod(s)) for s in self.context_specs.values()),
                **self._world_model_cfg,
            ).to(self.device)
            self.world_model_optimizer = torch.optim.Adam(
                self.world_model.parameters(), lr=self._world_model_learning_rate
            )

    def train_mode(self):
        self.actor_critic.train()
        self.world_model.train()

    def test_mode(self):
        self.actor_critic.eval()
        self.world_model.eval()

    def set_normalizers(self, normalizers):
        """Runner hook: keep the normalisers where the imagination can reach them."""
        self.normalizers = dict(normalizers or {})

    def _normalize(self, group: str, obs: torch.Tensor) -> torch.Tensor:
        """Apply the same normaliser the runner applies to the real observation group."""
        normalizer = getattr(self, "normalizers", {}).get(group)
        return obs if normalizer is None else normalizer(obs)

    def act(self, obs, extra_obs):
        """Deterministic action plus exploration noise."""
        with torch.no_grad():
            action = self.actor_critic.act_inference(obs)
            if self.collection_noise_std > 0.0:
                noise = torch.randn(
                    action.shape, generator=self._noise_generator, dtype=action.dtype
                ).to(action.device)
                action = action + self.collection_noise_std * noise
        self._last_action = action.detach()
        self._last_obs_pack = extra_obs
        return action

    def process_env_step(self, rewards, dones, infos, next_obs, next_extra_obs):
        """Write the transition into the replay buffer."""
        pack = next_extra_obs if next_extra_obs is not None else {}
        step: dict[str, torch.Tensor] = {}

        # The env ships the world-model state as one concatenated observation group; the layout lives
        # in the imagination, which slices it into the named components the replay stores.
        if any(name not in pack for name in self.state_names):
            pack = {**pack, **self.imagination.split_obs_pack(pack)}

        for name in self.state_names:
            if name not in pack:
                raise KeyError(f"imagination state component {name!r} missing from the observation groups")
            step[name] = pack[name].reshape(self.num_envs, -1).to(self.device).detach() if torch.is_tensor(pack[name]) \
                else torch.as_tensor(pack[name], device=self.device)
        for name in self.context_names:
            if name not in pack:
                raise KeyError(f"imagination context component {name!r} missing from the observation groups")
            step[name] = pack[name].reshape(self.num_envs, -1).to(self.device).detach()
        for name in self.extra_replay_specs:
            if name not in pack:
                raise KeyError(f"replay component {name!r} missing from the observation groups")
            step[name] = pack[name].reshape(self.num_envs, -1).to(self.device).detach()

        step["action"] = self._last_action.reshape(self.num_envs, -1)
        # The env may return (num_envs,) or (num_envs, num_rewards); normalise to 2-D so the stored
        # spec matches either shape.
        reward = rewards.to(self.device)
        step["reward"] = reward.reshape(self.num_envs, -1)
        done = dones.to(self.device)
        step["done"] = done.reshape(self.num_envs, -1)[:, :1].float()
        self._episode_step = (self._episode_step + 1) * (1 - step["done"].squeeze(-1).long())
        step["episode_step"] = self._episode_step.clone()
        self.replay.add(step)

        self.actor_critic.reset(dones)

    def compute_returns(self, last_extra_obs):
        """No-op: SuperMimic never computes returns from the real rollout."""
        return None

    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        losses = defaultdict(float)
        stats = defaultdict(float)

        if self.replay is None or not self.replay.ready(max(self.wm_horizon + 1, 2)):
            return dict(losses), dict(stats)

        warmup = current_learning_iteration < self.wm_warmup_iterations
        cycles = 1 if warmup else self.updates_per_iteration

        for _ in range(cycles):
            for key, value in self._fit_world_model().items():
                losses[key] += value
            stats["world_model_trained"] = 1.0
            if warmup:
                continue
            imagined = self._imagine()
            # An untrained dynamics model compounds error over H steps; a divergent rollout must not be
            # fitted on, or one bad update poisons the critic and then the simulator.
            if not self._imagined_is_finite(imagined):
                stats["imagined_skipped"] += 1.0
                continue
            for key, value in self._fit_critic(imagined).items():
                losses[key] += value
            for key, value in self._fit_policy(imagined).items():
                losses[key] += value

        divisor = float(cycles)
        return ({k: v / divisor for k, v in losses.items()}, dict(stats))

    def _state_bounds(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Flattened per-dimension min/max of the state, in state-vector order."""
        if self.replay is None or not self.replay.ready(2):
            return None
        rng = self.replay.observed_range(self.state_names)
        lo = torch.cat([rng[n][0].reshape(-1) for n in self.state_names]).to(self.device)
        hi = torch.cat([rng[n][1].reshape(-1) for n in self.state_names]).to(self.device)
        return lo, hi

    def _clamp_state(self, state: torch.Tensor, bounds) -> torch.Tensor:
        """Keep an imagined state inside the region the model was fitted on."""
        if bounds is None:
            return state
        lo, hi = bounds
        return torch.clamp(state, lo, hi)

    @staticmethod
    def _imagined_is_finite(imagined: dict[str, torch.Tensor]) -> bool:
        bad = [
            name
            for name, v in imagined.items()
            if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()
        ]
        if bad:
            # Name the culprit once rather than silently skipping every update: a permanently
            # divergent rollout must be visible, not absorbed.
            if not getattr(SuperMimic, "_warned_nan", False):
                step = next(
                    (
                        int(t)
                        for name in bad
                        for t in (~torch.isfinite(imagined[name])).nonzero()[0][:1]
                    ),
                    0,
                )
                print(f"[SuperMimic] imagined rollout diverged: non-finite in {bad} (first at index {step})")
                SuperMimic._warned_nan = True
            return False
        return True

    # ------------------------------------------------------------------ stages

    def _state_from_pack(self, pack: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([pack[name].reshape(pack[name].shape[0], -1) for name in self.state_names], dim=-1)

    def _context_from_pack(self, pack: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if not self.context_names:
            return None
        return torch.cat([pack[name].reshape(pack[name].shape[0], -1) for name in self.context_names], dim=-1)

    def _fit_world_model(self) -> dict[str, float]:
        """Fit the dynamics model on short windows from the replay."""
        self.world_model.train()
        keys = self.state_names + self.context_names
        total = defaultdict(float)
        for _ in range(self.wm_batches_per_update):
            windows, mask = self.replay.sample_windows(
                self.wm_batch_size, self.wm_horizon + 1, keys + ["action"]
            )
            # the replay lives on the CPU to keep GPU memory for activations
            windows = {k: v.to(self.device) for k, v in windows.items()}
            mask = mask.to(self.device)
            # (B, L, E, ...) -> treat every env as an independent sample: fold L and E
            def flatten(name, offset):
                t = windows[name][:, offset]  # (B, E, ...)
                return t.reshape(-1, *t.shape[2:])

            # states at t..t+k, actions at t..t+k-1, stacked over the window
            state_seq = torch.cat(
                [torch.cat([flatten(n, i) for n in self.state_names], dim=-1).unsqueeze(1)
                 for i in range(self.wm_horizon + 1)],
                dim=1,
            )  # (B*E, k+1, state)
            action_seq = torch.cat(
                [flatten("action", i).unsqueeze(1) for i in range(self.wm_horizon)],
                dim=1,
            )  # (B*E, k, action)
            if self.context_names:
                context = torch.cat([flatten(n, 0) for n in self.context_names], dim=-1)
            else:
                context = None
            flat_mask = mask.reshape(-1).to(torch.bool)

            if not bool(flat_mask.any()):
                continue
            losses = self.world_model.rollout_loss(
                state_seq[:, 0], action_seq, context, state_seq[:, 1:], mask=None
            )
            loss = losses["loss"]
            self.world_model_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.world_model_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.world_model.parameters(), self.world_model_max_grad_norm)
            self.world_model_optimizer.step()
            total["world_model_loss"] += float(loss.detach())
        return {k: v / max(self.wm_batches_per_update, 1) for k, v in total.items()}

    @torch.no_grad()
    def _sample_imagined_starts(self):
        """Draw start states, with the history window the policy observation needs, from the replay."""
        history = max(int(getattr(self.imagination, "history_length", 0)), 1)
        needs_history = bool(getattr(self.imagination, "history_terms", ()))
        keys = self.state_names + self.context_names
        if needs_history:
            # The stored action at a row is the action that *produced* that row's state, which is
            # exactly the ``last_action`` observation of that frame.
            keys = keys + ["action"]
        batch = self.replay.sample_starts(self.num_imagined, keys, history=history)
        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        state = {name: batch[name][:, -1] for name in self.state_names}
        context = {name: batch[name][:, -1] for name in self.context_names}

        history_state = history_action = None
        if needs_history:
            history_state = torch.cat(
                [batch[name].reshape(self.num_imagined, history, -1) for name in self.state_names], dim=-1
            )
            history_action = batch["action"].reshape(self.num_imagined, history, -1)
        return state, context, batch["history_mask"], batch.get("env_index"), history_state, history_action

    def _imagine(self) -> dict[str, torch.Tensor]:
        """Roll the world model forward under the current policy.

        Returns batched tensors with time on dim 0: states, actions, rewards, termination
        probabilities and the policy-observation history the policy consumed.
        """
        state, context, _, env_index, history_state, history_action = self._sample_imagined_starts()
        self._bounds = self._state_bounds()
        state_vec = torch.cat([state[name] for name in self.state_names], dim=-1)
        context_vec = (
            torch.cat([context[name] for name in self.context_names], dim=-1) if self.context_names else None
        )

        # The reference snapshot is taken from the environment's *current* clip position and the rollout
        # starts at column 0 of it. Seeding each rollout from its own replayed frame index would need one
        # window per sample, which does not fit in memory at the paper's batch sizes; the resulting bias
        # only affects which clip position a rollout starts at, not its dynamics or reward.
        ref_index = None
        if hasattr(self.imagination, "refresh_reference_window"):
            self.imagination.refresh_reference_window(self.horizon)
            # one window row per *environment*; imagined starts are sampled across envs, so select the
            # row each start belongs to before the rollout indexes it by frame.
            if env_index is None:
                raise RuntimeError("imagined starts did not report their env index")
            self.imagination.select_reference_rows(env_index)
            # rows of the *selected* window, so a later batch subset can pick its own slice
            self._imagined_row_index = torch.arange(state_vec.shape[0], device=self.device)
            ref_index = torch.zeros(state_vec.shape[0], dtype=torch.long, device=self.device)

        self.world_model.eval()
        self.actor_critic.eval()

        # remembered so the policy step can start from the same held states
        self._imagined_start_state = state_vec
        self._imagined_start_context = context_vec
        self._imagined_start_ref = ref_index
        self._imagined_start_history = (history_state, history_action)
        self._imagined_local_ref = (
            None if ref_index is None else torch.zeros(ref_index.shape[0], dtype=torch.long, device=self.device)
        )

        # The action that brought the robot to the start frame is the last column of the action window;
        # the environment's ``prev_action`` at reset (zeros) is the fallback when no history is kept.
        prev_action = (
            history_action[:, -1]
            if history_action is not None
            else torch.zeros(state_vec.shape[0], self.num_actions, device=self.device)
        )
        obs = self._normalize(
            "policy",
            self.imagination.policy_obs(
                state_vec, context_vec, prev_action, ref_index, history_state, history_action
            ),
        )
        obs_hist = []
        state_hist, action_hist, reward_hist, done_hist, value_hist = [], [], [], [], []

        for _ in range(self.horizon):
            obs_hist.append(obs)
            action = self.actor_critic.act_inference(obs)
            reward = self.imagination.reward(state_vec, action, prev_action, ref_index)
            done = self.imagination.termination_prob(state_vec, ref_index)
            value = (
                self.actor_critic.evaluate(
                    self._normalize("critic", self.imagination.critic_obs(state_vec, context_vec, ref_index))
                )
                if self.use_critic
                else torch.zeros(state_vec.shape[0], 1, device=self.device)
            )

            state_hist.append(state_vec)
            action_hist.append(action)
            reward_hist.append(reward)
            done_hist.append(done)
            value_hist.append(value)

            with torch.no_grad():
                next_state_vec = self._clamp_state(
                    self.world_model(state_vec, action, context_vec), self._bounds
                )
            ref_index = self.imagination.advance(ref_index)
            if history_state is not None:
                history_state, history_action = self.imagination.advance_history(
                    history_state, history_action, next_state_vec, action
                )
            obs = self._normalize(
                "policy",
                self.imagination.policy_obs(
                    next_state_vec, context_vec, action, ref_index, history_state, history_action
                ),
            )
            obs = self.imagination.add_obs_noise(obs)

            state_vec = next_state_vec
            prev_action = action

        return {
            "obs": torch.stack(obs_hist),
            "state": torch.stack(state_hist),
            "action": torch.stack(action_hist),
            "reward": torch.stack(reward_hist),
            "done": torch.stack(done_hist),
            "value": torch.stack(value_hist),
        }

    def _fit_critic(self, imagined: dict[str, torch.Tensor]) -> dict[str, float]:
        """Regress the critic onto lambda-returns of the imagined rollouts."""
        with torch.no_grad():
            # ``reward`` is per-term (already weighted); the return is over their sum, which is the
            # scalar the environment's reward manager also produces.
            # values are (T, B, 1); the reduced reward is (T, B), so match shapes explicitly
            returns, _ = compute_gae(
                imagined["reward"].sum(dim=-1, keepdim=True),
                imagined["value"],
                imagined["done"],
                torch.zeros_like(imagined["value"][-1]),
                self.gamma,
                self.lam,
            )
            targets = returns.detach()

        self.actor_critic.train()
        self.world_model.eval()
        batch = imagined["value"].shape[1]
        mini = max(batch // self.critic_mini_batches, 1)
        total = 0.0
        for _ in range(self.critic_epochs):
            perm = torch.randperm(batch, device=self.device)
            for start in range(0, batch, mini):
                idx = perm[start : start + mini]
                # The critic input is rebuilt from the imagined states through the interface, so it
                # sees the same features the runner's critic group would provide.
                context_sel = (
                    self._imagined_start_context[idx] if self._imagined_start_context is not None else None
                )
                ref_sel = self._imagined_local_ref[idx] if self._imagined_local_ref is not None else None
                pred = torch.stack(
                    [
                        self.actor_critic.evaluate(
                            self._normalize("critic", self.imagination.critic_obs(imagined["state"][t][idx], context_sel, ref_sel))
                        )
                        for t in range(imagined["state"].shape[0])
                    ]
                )
                loss = ((pred - targets[:, idx]) ** 2).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.critic_max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.critic_max_grad_norm)
                self.critic_optimizer.step()
                total += float(loss.detach())
        n = self.critic_epochs * max(len(range(0, batch, mini)), 1)
        return {"critic_loss": total / n}

    def _fit_policy(self, imagined: dict[str, torch.Tensor]) -> dict[str, float]:
        """Maximise the imagined return, differentiating through the world model.

        This is the SuperMimic gradient: the policy sees the consequences of its actions *through* the
        dynamics model, so a single differentiable rollout replaces PPO's importance-ratio surrogate.
        The started states, context and per-step rewards come from the caller's imagination interface,
        so the imagined objective is the same one the simulation would produce.

        The pass is chunked over the batch to bound activation memory (the paper's ``H``-step graph is
        large); each chunk is one optimiser step on the mean return.
        """
        self.actor_critic.train()
        self.world_model.train()

        batch = imagined["value"].shape[1]
        chunk = max(min(batch, self.policy_chunk_size), 1)
        total = 0.0
        steps = 0
        for start in range(0, batch, chunk):
            idx = torch.arange(start, min(start + chunk, batch), device=self.device)
            objective = self._imagined_return(idx)
            loss = -objective
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.policy_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.policy_max_grad_norm)
            if self.policy_grad_value_clip is not None:
                nn.utils.clip_grad_value_(self.actor_critic.actor.parameters(), self.policy_grad_value_clip)
            self.policy_optimizer.step()
            total += float(objective.detach())
            steps += 1
        return {"policy_return": total / max(steps, 1)}

    def _imagined_return(self, idx: torch.Tensor) -> torch.Tensor:
        """Differentiable H-step return for a batch subset, starting from held states."""
        state_vec = self._imagined_start_state[idx]
        context_vec = self._imagined_start_context
        ref_index = self._imagined_start_ref
        if ref_index is not None and hasattr(self.imagination, "select_reference_rows"):
            # This pass works on a batch subset, so the reference rows are selected to match it and the
            # frame index is rebuilt at the subset's width. Starting from column 0 is exact here: the
            # window was sampled at the current time for each sample's own environment.
            self.imagination.subset_reference_rows(idx)
            ref_index = torch.zeros(idx.numel(), dtype=torch.long, device=self.device)
        prev_action = (
            self._imagined_start_history[1][idx][:, -1]
            if self._imagined_start_history[1] is not None
            else torch.zeros(state_vec.shape[0], self.num_actions, device=self.device)
        )
        history_state, history_action = self._imagined_start_history
        if history_state is not None:
            history_state = history_state[idx]
            history_action = history_action[idx]
        obs = self._normalize(
            "policy",
            self.imagination.policy_obs(
                state_vec, context_vec, prev_action, ref_index, history_state, history_action
            ),
        )
        total = torch.zeros(state_vec.shape[0], device=self.device)
        for _ in range(self.horizon):
            action = self.actor_critic.act_inference(obs)
            reward = self.imagination.reward(state_vec, action, prev_action, ref_index).sum(dim=-1)
            total = total + reward
            state_vec = self._clamp_state(self.world_model(state_vec, action, context_vec), self._bounds)
            ref_index = self.imagination.advance(ref_index)
            if history_state is not None:
                history_state, history_action = self.imagination.advance_history(
                    history_state, history_action, state_vec, action
                )
            obs = self._normalize(
                "policy",
                self.imagination.policy_obs(
                    state_vec, context_vec, action, ref_index, history_state, history_action
                ),
            )
            prev_action = action
        if self.use_critic:
            bootstrap = self.actor_critic.evaluate(
                self._normalize("critic", self.imagination.critic_obs(state_vec, context_vec, ref_index))
            ).squeeze(-1)
            total = total + bootstrap
        return total.mean()

    # ------------------------------------------------------------------ bookkeeping

    def distributed_data_parallel(self):
        return None

    def state_dict(self) -> dict:
        state = {
            "world_model": self.world_model.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "episode_step": self._episode_step,
            "iteration": self.current_learning_iteration,
        }
        if self.save_replay_in_checkpoint and self.replay is not None:
            state["replay_state_dict"] = self.replay.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        required = {"world_model", "world_model_optimizer", "policy_optimizer", "critic_optimizer"}
        missing = required - set(state)
        if missing:
            raise KeyError(f"checkpoint is missing SuperMimic entries: {sorted(missing)}")
        self.world_model.load_state_dict(state["world_model"])
        self.world_model_optimizer.load_state_dict(state["world_model_optimizer"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "episode_step" in state and self.replay is not None:
            saved = state["episode_step"].to(self._episode_step.dtype)
            if saved.shape == self._episode_step.shape:
                self._episode_step = saved
        self.current_learning_iteration = int(state.get("iteration", 0))
        if "replay_state_dict" in state and self.replay is not None:
            self.replay.load_state_dict(state["replay_state_dict"])
