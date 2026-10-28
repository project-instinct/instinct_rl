"""SuperMimic: motion tracking by backpropagating through a learned world model.

The algorithm replaces PPO's update step. It keeps PPO's interaction loop (collect with the policy,
store into a replay buffer) and then, instead of computing advantages from real returns, fits a
dynamics model on the replay and optimises the policy on *imagined* rollouts produced by that model.
A critic bootstraps the return past the imagination horizon, which is what makes long horizons stable.

Structure (paper: Liao et al., arXiv 2610.28480, Tables A.1 and A.5):

1. **Collect** ``num_steps_per_env`` steps with the deterministic policy plus Gaussian action noise,
   storing ``(state, context, action, reward, done, episode_step)`` plus the imagination's seed groups
   (e.g. which reference each state was tracking) in a
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
from instinct_rl.utils.returns import compute_gae


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

    #: Observation group carrying the simulation context.
    context_group = "sim_context"

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
        world_model_predict_delta: Sequence[str] | str = "all",
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
            state_specs = {name: TensorSpec((int(np.prod(shape)),)) for name, shape in imagination.state_format.items()}
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
        self._imagined_start: dict | None = None
        self._imagined_has_reference = False
        self._bounds: tuple[torch.Tensor, torch.Tensor] | None = None

        self._world_model_learning_rate = float(world_model_learning_rate)
        self.world_model_optimizer = (
            None
            if self.world_model is None
            else torch.optim.Adam(self.world_model.parameters(), lr=self._world_model_learning_rate)
        )
        # The critic is the ``critic`` head plus a critic-only encoder if the model has one; the policy is
        # every other parameter. Selecting ``actor`` alone would leave an actor-side encoder untrained.
        critic_params = list(self.actor_critic.critic.parameters())
        critic_encoders = getattr(self.actor_critic, "critic_encoders", None)
        if critic_encoders is not None and critic_encoders is not getattr(self.actor_critic, "encoders", None):
            critic_params += list(critic_encoders.parameters())
        if not critic_params:
            raise ValueError("SuperMimic needs a critic network in the actor_critic (even if use_critic is False)")
        critic_ids = {id(p) for p in critic_params}
        self.policy_params = [p for p in self.actor_critic.parameters() if id(p) not in critic_ids]
        self.critic_params = critic_params
        # paper Table A.5: Adam with beta = (0.9, 0.95) for the policy
        self.policy_optimizer = torch.optim.Adam(self.policy_params, lr=policy_learning_rate, betas=(0.9, 0.95))
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
        # The simulation context is carried per transition like state: the dynamics model is
        # conditioned on it, so a rollout seeded from a replayed frame must use that frame's world.
        # Derived from ``obs_format`` when the config did not name one, before any spec is built.
        if self.context_group in obs_format and not self.context_specs:
            for name, shape in obs_format[self.context_group].items():
                self.context_specs[name] = TensorSpec((int(np.prod(shape)),))
            self.context_names = list(self.context_specs)

        specs: dict[str, TensorSpec] = {}
        specs.update(self.state_specs)
        specs.update(self.context_specs)
        specs.update(self.extra_replay_specs)
        specs["action"] = TensorSpec((self.num_actions,))
        specs["reward"] = TensorSpec((int(num_rewards),))
        specs["done"] = TensorSpec((1,))
        specs["episode_step"] = TensorSpec((), torch.int64)
        # Groups the imagination needs to seed a rollout from a replayed state (e.g. which reference the
        # state was tracking). Stored per transition under the group name, flattened.
        self.seed_groups = tuple(getattr(self.imagination, "SEED_GROUPS", ()))
        for group in self.seed_groups:
            if group not in obs_format:
                raise KeyError(f"the imagination needs observation group {group!r} to seed rollouts")
            size = sum(int(np.prod(shape)) for shape in obs_format[group].values())
            self.extra_replay_specs[group] = TensorSpec((size,))
            specs[group] = self.extra_replay_specs[group]
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
                context_size=sum(int(np.prod(spec.shape)) for spec in self.context_specs.values()),
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
        """Apply the runner's normaliser for this group, without updating its statistics.

        The statistics describe the real observations the deployed policy sees. Calling the normaliser
        in training mode would also fold every imagined observation into them, so imagined (and possibly
        off-distribution) data would shift how real observations are scaled.
        """
        normalizer = getattr(self, "normalizers", {}).get(group)
        return obs if normalizer is None else normalizer.normalize(obs)

    def act(self, obs, extra_obs):
        """Deterministic action plus exploration noise."""
        with torch.no_grad():
            action = self.actor_critic.act_inference(obs)
            if self.collection_noise_std > 0.0:
                noise = torch.randn(action.shape, generator=self._noise_generator, dtype=action.dtype).to(action.device)
                action = action + self.collection_noise_std * noise
        self._last_action = action.detach()
        self._last_obs_pack = extra_obs
        return action

    def process_env_step(self, rewards, dones, infos, next_obs, next_extra_obs):
        """Write the transition into the replay buffer."""
        pack = next_extra_obs if next_extra_obs is not None else {}
        step: dict[str, torch.Tensor] = {}

        # The env ships the world-model state and simulation context as one concatenated tensor per
        # group; the layouts live in the imagination, which slices them into the named components the
        # replay stores.
        if any(name not in pack for name in self.state_names + self.context_names):
            pack = {**pack, **self.imagination.split_obs_pack(pack)}

        for name in self.state_names:
            if name not in pack:
                raise KeyError(f"imagination state component {name!r} missing from the observation groups")
            step[name] = (
                pack[name].reshape(self.num_envs, -1).to(self.device).detach()
                if torch.is_tensor(pack[name])
                else torch.as_tensor(pack[name], device=self.device)
            )
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
                    (int(t) for name in bad for t in (~torch.isfinite(imagined[name])).nonzero()[0][:1]),
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
            # Draw one environment per start. Without ``env_ids`` the sampler returns every
            # environment for each start, and folding those dimensions makes the world-model batch
            # ``wm_batch_size * num_envs`` -- a silently different batch on every machine, and at
            # 4096 envs large enough to exhaust the GPU. The replay holds envs independently, so one
            # random env per start is a valid sample of the same data.
            env_sel = torch.randint(0, self.num_envs, (self.wm_batch_size,), generator=self._noise_generator)
            windows, mask = self.replay.sample_windows(
                self.wm_batch_size, self.wm_horizon + 1, keys + ["action"], env_ids=env_sel
            )
            # the replay lives on the CPU to keep GPU memory for activations
            windows = {k: v.to(self.device) for k, v in windows.items()}
            mask = mask.to(self.device)

            def flatten(name, offset):
                # ``(B, ...)``: one environment per sample, so the batch axis is already right.
                return windows[name][:, offset]

            # states at t..t+k, actions at t..t+k-1, stacked over the window
            state_seq = torch.cat(
                [
                    torch.cat([flatten(n, i) for n in self.state_names], dim=-1).unsqueeze(1)
                    for i in range(self.wm_horizon + 1)
                ],
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
            # Broadcast the per-window validity to every step: a window that crosses an episode reset
            # describes a discontinuity, not dynamics, so it must not be fitted on.
            losses = self.world_model.rollout_loss(
                state_seq[:, 0],
                action_seq,
                context,
                state_seq[:, 1:],
                mask=flat_mask.unsqueeze(-1).expand(-1, self.wm_horizon),
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
    def _sample_imagined_starts(self) -> dict:
        """Draw start states from the replay, with what the imagination needs to continue from them.

        Returns ``state`` and ``context`` vectors of the newest frame, its ``env_index``, the trailing
        ``history_state`` / ``history_action`` windows when the policy keeps observation history, and
        ``seeds``: the newest frame of every seed group the imagination declared.
        """
        history = max(int(getattr(self.imagination, "history_length", 0)), 1)
        needs_history = bool(getattr(self.imagination, "history_terms", ()))
        keys = self.state_names + self.context_names + list(self.seed_groups)
        if needs_history:
            # The stored action at a row is the action that *produced* that row's state, which is
            # exactly the ``last_action`` observation of that frame.
            keys = keys + ["action"]
        batch = self.replay.sample_starts(self.num_imagined, keys, history=history)
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        def newest(names):
            return torch.cat([batch[n][:, -1] for n in names], dim=-1) if names else None

        start = {
            "state": newest(self.state_names),
            "context": newest(self.context_names),
            "env_index": batch["env_index"],
            "seeds": {g: batch[g][:, -1] for g in self.seed_groups},
            "history_state": None,
            "history_action": None,
        }
        if needs_history:
            start["history_state"] = torch.cat(
                [batch[n].reshape(self.num_imagined, history, -1) for n in self.state_names], dim=-1
            )
            start["history_action"] = batch["action"].reshape(self.num_imagined, history, -1)
        return start

    def _policy_observation(self, state, context, prev_action, ref_index, history_state, history_action):
        """Imagined policy input: the noise-free observation, plus sensor noise, then normalised.

        Paper Sec. 3.3: the model emits the noise-free observation and the sensor noise is re-added
        before the policy acts. Noise goes on the raw observation, where the environment applies it.
        """
        obs = self.imagination.policy_obs(state, context, prev_action, ref_index, history_state, history_action)
        return self._normalize("policy", self.imagination.add_obs_noise(obs))

    def _value(self, state, context, ref_index, prev_action) -> torch.Tensor:
        """Critic value ``(B, 1)``; zero for the paper's critic ablation."""
        if not self.use_critic:
            return torch.zeros(state.shape[0], 1, device=self.device)
        obs = self.imagination.critic_obs(state, context, ref_index, prev_action)
        return self.actor_critic.evaluate(self._normalize("critic", obs))

    def _rows(self, idx: torch.Tensor):
        """The held rollout starts restricted to the batch rows ``idx``."""
        start = self._imagined_start
        if self._imagined_has_reference:
            self.imagination.subset_reference_rows(idx)

        def take(t):
            return None if t is None else t[idx]

        return (
            start["state"][idx],
            take(start["context"]),
            take(start["history_state"]),
            take(start["history_action"]),
        )

    def _first_prev_action(self, history_action, batch: int) -> torch.Tensor:
        """The action that produced the start state: the newest entry of the action window, or the
        environment's value at reset (zeros) when no history is kept."""
        if history_action is not None:
            return history_action[:, -1]
        return torch.zeros(batch, self.num_actions, device=self.device)

    @torch.no_grad()
    def _imagine(self) -> dict[str, torch.Tensor]:
        """Roll the world model forward under the current policy, without gradients.

        This is the data the critic is fit on. The policy gradient is taken in a separate differentiable
        pass (:meth:`_imagined_return`) from the same held starts.

        Timing follows the environment (and paper Eq. 2): action ``a_t`` is taken in ``s_t``, and the
        reward ``r_{t+1}`` and termination ``d_{t+1}`` are evaluated on ``s_{t+1}`` against the
        reference one step later.

        Returns:
            ``state`` and ``prev_action`` with ``H + 1`` steps (``s_0 .. s_H`` and the action that
            produced each), ``reward``/``done`` ``(H, B, 1)`` for ``r_1 .. r_H`` / ``d_1 .. d_H``, and
            ``value`` ``(H + 1, B, 1)`` for ``V(s_0) .. V(s_H)``.
        """
        start = self._sample_imagined_starts()
        self._bounds = self._state_bounds()
        batch = start["state"].shape[0]

        ref_index = None
        if hasattr(self.imagination, "begin_rollout"):
            ref_index = self.imagination.begin_rollout(start["seeds"], start["env_index"], self.horizon)
        self._imagined_start = start
        self._imagined_has_reference = ref_index is not None

        self.world_model.eval()
        self.actor_critic.eval()

        state, context = start["state"], start["context"]
        history_state, history_action = start["history_state"], start["history_action"]
        prev_action = self._first_prev_action(history_action, batch)
        obs = self._policy_observation(state, context, prev_action, ref_index, history_state, history_action)

        states, prev_actions, rewards, dones = [state], [prev_action], [], []
        for _ in range(self.horizon):
            action = self.actor_critic.act_inference(obs)
            next_state = self._clamp_state(self.world_model(state, action, context), self._bounds)
            ref_index = self.imagination.advance(ref_index)
            rewards.append(self.imagination.reward(next_state, action, prev_action, ref_index).sum(-1, keepdim=True))
            dones.append(self.imagination.termination_prob(next_state, ref_index))
            if history_state is not None:
                history_state, history_action = self.imagination.advance_history(
                    history_state, history_action, next_state, action
                )
            obs = self._policy_observation(next_state, context, action, ref_index, history_state, history_action)
            state, prev_action = next_state, action
            states.append(state)
            prev_actions.append(prev_action)

        values = [
            self._value(states[t], context, self._ref_at(t, batch), prev_actions[t]) for t in range(self.horizon + 1)
        ]
        return {
            "state": torch.stack(states),
            "prev_action": torch.stack(prev_actions),
            "reward": torch.stack(rewards),
            "done": torch.stack(dones),
            "value": torch.stack(values),
        }

    def _ref_at(self, t: int, batch: int) -> torch.Tensor | None:
        """Reference column of imagined step ``t`` for ``batch`` rollouts started at column 0."""
        if not self._imagined_has_reference:
            return None
        return torch.full((batch,), t, dtype=torch.long, device=self.device)

    def _fit_critic(self, imagined: dict[str, torch.Tensor]) -> dict[str, float]:
        """Regress the critic onto the lambda-returns of the imagined rollouts (paper Eq. 3).

        The recursion starts from ``R_H = V(s_H)``, the critic's own estimate beyond the horizon, and
        the targets are computed once per rollout and held fixed while the critic is fit.
        """
        with torch.no_grad():
            targets, _ = compute_gae(
                imagined["reward"],
                imagined["value"][:-1],
                imagined["done"],
                imagined["value"][-1],
                self.gamma,
                self.lam,
            )

        self.actor_critic.train()
        batch = targets.shape[1]
        mini = max(batch // self.critic_mini_batches, 1)
        total, count = 0.0, 0
        for _ in range(self.critic_epochs):
            perm = torch.randperm(batch, device=self.device)
            for begin in range(0, batch, mini):
                idx = perm[begin : begin + mini]
                _, context, _, _ = self._rows(idx)
                pred = torch.stack(
                    [
                        self._value(
                            imagined["state"][t][idx],
                            context,
                            self._ref_at(t, idx.numel()),
                            imagined["prev_action"][t][idx],
                        )
                        for t in range(self.horizon)
                    ]
                )
                loss = ((pred - targets[:, idx]) ** 2).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.critic_max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critic_params, self.critic_max_grad_norm)
                self.critic_optimizer.step()
                total += float(loss.detach())
                count += 1
        return {"critic_loss": total / max(count, 1)}

    def _fit_policy(self, imagined: dict[str, torch.Tensor]) -> dict[str, float]:
        """Maximise the imagined return, differentiating through the world model.

        This is the SuperMimic gradient: the policy sees the consequences of its actions *through* the
        dynamics model, so a single differentiable rollout replaces PPO's importance-ratio surrogate.

        The pass is chunked over the batch to bound activation memory; each chunk is one optimiser
        step on its mean return.
        """
        self.actor_critic.train()
        self.world_model.eval()

        batch = imagined["value"].shape[1]
        chunk = max(min(batch, self.policy_chunk_size), 1)
        total = 0.0
        reward_total = 0.0
        steps = 0
        for begin in range(0, batch, chunk):
            idx = torch.arange(begin, min(begin + chunk, batch), device=self.device)
            objective, reward_sum = self._imagined_return(idx)
            loss = -objective
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.policy_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.policy_params, self.policy_max_grad_norm)
            if self.policy_grad_value_clip is not None:
                nn.utils.clip_grad_value_(self.policy_params, self.policy_grad_value_clip)
            self.policy_optimizer.step()
            total += float(objective.detach())
            reward_total += float(reward_sum)
            steps += 1
        # ``policy_return`` includes the critic's bootstrap, so it also moves as the critic learns;
        # ``imagined_reward`` is the H-step reward part alone, which only the policy changes.
        return {"policy_return": total / max(steps, 1), "imagined_reward": reward_total / max(steps, 1)}

    def _imagined_return(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable imagined objective for the batch rows ``idx`` (paper Eq. 2).

        Returns the objective and, detached, its H-step reward part without the bootstrap.

        ``J = sum_t gamma^t c_t r_{t+1} + gamma^H c_H V(s_H)``, with the soft survival weight
        ``c_0 = 1``, ``c_{t+1} = c_t (1 - d_{t+1})``, so returns after a likely termination are
        discounted away while the gradient through ``d`` is kept.
        """
        state, context, history_state, history_action = self._rows(idx)
        batch = state.shape[0]
        ref_index = self._ref_at(0, batch)
        prev_action = self._first_prev_action(history_action, batch)
        obs = self._policy_observation(state, context, prev_action, ref_index, history_state, history_action)

        total = torch.zeros(batch, device=self.device)
        survival = torch.ones(batch, device=self.device)
        discount = 1.0
        for _ in range(self.horizon):
            action = self.actor_critic.act_inference(obs)
            next_state = self._clamp_state(self.world_model(state, action, context), self._bounds)
            ref_index = self.imagination.advance(ref_index)
            reward = self.imagination.reward(next_state, action, prev_action, ref_index).sum(dim=-1)
            total = total + discount * survival * reward
            survival = survival * (1.0 - self.imagination.termination_prob(next_state, ref_index).squeeze(-1))
            discount *= self.gamma
            if history_state is not None:
                history_state, history_action = self.imagination.advance_history(
                    history_state, history_action, next_state, action
                )
            obs = self._policy_observation(next_state, context, action, ref_index, history_state, history_action)
            state, prev_action = next_state, action
        reward_sum = total.detach().mean()
        total = total + discount * survival * self._value(state, context, ref_index, prev_action).squeeze(-1)
        return total.mean(), reward_sum

    # ------------------------------------------------------------------ bookkeeping

    def distributed_data_parallel(self):
        return None

    def state_dict(self) -> dict:
        state = {
            # the policy and critic, under the key PPO uses, so play/export scripts find them
            "model_state_dict": self.actor_critic.state_dict(),
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
        required = {"model_state_dict", "world_model", "world_model_optimizer", "policy_optimizer", "critic_optimizer"}
        missing = required - set(state)
        if missing:
            raise KeyError(f"checkpoint is missing SuperMimic entries: {sorted(missing)}")
        self.actor_critic.load_state_dict(state["model_state_dict"])
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
