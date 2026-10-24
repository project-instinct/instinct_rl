"""SuperMimic: collect -> fit a world model -> BPTT policy + critic update in imagination. There is no PPO.

Adapted from the SuperMimic reference implementation (https://github.com/zita-ch/supermimic_vibed, MIT License,
Copyright (c) 2026 ETH Zurich, Chong Zhang; see licenses/dependencies/supermimic_vibed_license.txt) (``train.py``).

The algorithm runs on the `OnPolicyRunner` like every other algorithm (`init_storage` receives the env, from which
the task model is built, and the normalizers of the runner): every learning iteration the runner steps the env
``num_steps_per_env`` times with the policy plus exploration noise (`act`) and calls `update` once, so the
update-to-data ratio is ``1 / num_steps_per_env`` updates per env step. The transitions go to a replay storage, on which a world model (floating-base state dynamics + contacts) and a
failure head are fitted with k-step windows. The policy is then trained by back-propagating the discounted return
through H-step imaginations of the world model, with a critic fitted by GAE(lambda) on the imagined transitions.
`update` does nothing until the storage holds ``warmup_steps`` env steps, and trains only the world model until
learning iteration ``world_model_only_iterations``.

What the algorithm reads from the env is an observation group (``world_model_obs_group``, default "world_model"), as
`WasabiPPO` reads its ``actor_state_key``. It is defined by the env (no noise, no history, no normalizer), and its
components are found by their names in ``obs_format``:

    root_pos_component ("root_pos_w", 3)        root position in the world frame
    root_quat_component ("root_quat_w", 4)      root orientation quaternion (w, x, y, z), world frame
    root_lin_vel_component ("root_lin_vel_w", 3)  world linear velocity of the root link-frame origin
    root_ang_vel_component ("root_ang_vel_w", 3)  world angular velocity of the root
    joint_pos_component ("joint_pos", J)        absolute joint positions
    joint_vel_component ("joint_vel", J)        joint velocities
    last_action_component ("last_action", J)    the processed action of the action manager: in an observation before
                                                a step it is the previous action, in a terminal observation after the
                                                step it is the action that was applied in the step
    contact_component ("contact", C)            0/1 indicators of the undesired-contact bodies
    context_component ("context", X)            the episode-constant context (domain randomization parameters)
    command_components (("motion_frame", "motion_phase", "motion_origin"))
                                                the time-varying reference command as floats; a component of shape
                                                (1,) is a scalar for the task model

The state right after a step must be read BEFORE the env resets the finished envs, so the env also provides
``infos["terminal_observations"][world_model_obs_group]`` (see `VecEnv`). The episode step is tracked from ``dones``,
a real termination is ``dones & ~infos["time_outs"]``.

Everything that depends on the task (the analytic rewards / terminations / observations of the imagination) is behind
`instinct_rl.env.TaskModel`, built from ``task_model_class_name``. The policy IS the actor critic of the task, built
by `instinct_rl.modules.build_actor_critic`, and acts on the ``policy`` observation group, which in imagination is
assembled analytically (`TaskModel.policy_observation`) from the imagined state.

This class is the native setting of SuperMimic: proprioceptive dynamics only, the policy observation has no image. A
policy that sees an image plugs into the hooks of this class with `VisualSuperMimic` (``visual_latent.py``), like
`WasabiPPO` extends `PPO`.
"""

from __future__ import annotations

import importlib
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint

import instinct_rl.env as instinct_env
import instinct_rl.modules as instinct_modules
from instinct_rl.modules import ActorCritic
from instinct_rl.modules.world_model import (
    flatten_state,
    set_amp_dtype,
    unflatten_state,
)
from instinct_rl.storage import ReplayStorage
from instinct_rl.utils.rotations import quat_to_mat
from instinct_rl.utils.utils import get_obs_slice

NUM_STATE_COMPONENTS = 6  # the state tuple (p, R, v, w, q, qd)


def error_floor(num_joints: int) -> torch.Tensor:
    """Per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)."""
    return torch.cat(
        (
            torch.full((3,), 3e-3),
            torch.full((3,), 5e-3),
            torch.full((3,), 2e-2),
            torch.full((3,), 5e-2),
            torch.full((num_joints,), 3e-3),
            torch.full((num_joints,), 5e-2),
        )
    )


def stack_states(states: list[tuple]) -> tuple:
    """Stack a list of H state tuples (B, ...) along a new leading time dimension and merge it with the batch: the state
    tuple of (H * B, ...)."""
    return tuple(torch.stack([state[index] for state in states]).flatten(0, 1) for index in range(NUM_STATE_COMPONENTS))


def stack_commands(commands: list[dict], keys: tuple) -> dict:
    """The same as `stack_states` for a list of H commands."""
    return {key: torch.stack([command[key] for command in commands]).flatten(0, 1) for key in keys}


class SuperMimic:
    actor_critic: ActorCritic

    z_dim = 0
    """Size of the visual latent that is part of the world-model state. 0 = proprioceptive dynamics only. A plug-in that
    adds a latent state sets it before `init_storage` of this class runs."""

    def __init__(
        self,
        actor_critic,
        # ---- collection ----
        action_noise=0.1,
        action_clamp=20.0,
        warmup_steps=64,
        # ---- the observation group of the env, see the module docstring ----
        world_model_obs_group="world_model",
        root_pos_component="root_pos_w",
        root_quat_component="root_quat_w",
        root_lin_vel_component="root_lin_vel_w",
        root_ang_vel_component="root_ang_vel_w",
        joint_pos_component="joint_pos",
        joint_vel_component="joint_vel",
        last_action_component="last_action",
        contact_component="contact",
        context_component="context",
        command_components=("motion_frame", "motion_phase", "motion_origin"),
        # ---- replay storage ----
        buffer_steps=1000,
        proprio_history_length=8,
        # ---- world model ----
        world_model_window=8,
        world_model_batches=10,
        world_model_batch_size=2048,
        world_model_failure_fraction=0.25,
        world_model_failure_horizon=16,
        world_model_holdout_envs=0.1,
        world_model_only_iterations=500,
        contact_loss_coef=0.1,
        world_model_max_grad_norm=1.0,
        # ---- failure head ----
        use_failure_head=True,
        failure_head_warmup_iterations=300,
        failure_windows=512,
        failure_clip=0.95,
        failure_positive_weight=1.0,
        failure_head_max_grad_norm=1.0,
        # ---- policy / critic ----
        imagination_horizon=32,
        policy_batch_size=1024,
        gamma=0.99,
        lam=0.95,
        use_critic=True,
        critic_epochs=5,
        critic_minibatches=4,
        critic_max_grad_norm=1.0,
        max_grad_norm=1.0,
        max_grad_value=0.1,
        # ---- optimizers and learning rate schedule ----
        learning_rate=2e-4,
        lr_decay_iters=250_000,
        lr_final_ratio=0.03,
        optimizer_class_name="AdamW",
        optimizer_kwargs={"betas": (0.9, 0.95), "weight_decay": 0.0},
        world_model_optimizer_class_name="Adam",
        world_model_optimizer_kwargs={},
        failure_head_optimizer_class_name="Adam",
        failure_head_optimizer_kwargs={},
        critic_optimizer_class_name="Adam",
        critic_optimizer_kwargs={},
        # ---- performance ----
        amp="bf16",
        allow_tf32=True,
        compile=True,
        compile_policy=True,
        policy_autocast=False,
        checkpoint_imagination=False,
        # ---- diagnostics: held-out calibration of the failure head, imagination vs reality, plug-in diagnostics ----
        diagnostics_interval=50,
        termination_check_interval=250,
        log_reward_gap=True,
        # ---- networks, built by ``instinct_rl.modules.build_world_model_module`` and the task model ----
        world_model_class_name="FloatingBaseWorldModel",
        world_model_kwargs={},
        failure_head_class_name="FailureHead",
        failure_head_kwargs={},
        value_head_class_name="ValueHead",
        value_head_kwargs={},
        task_model_class_name="",
        task_model_kwargs={},
        device="cpu",
        **kwargs,
    ):
        """

        Args:
            action_noise: Std of the Gaussian exploration noise added to the actions of the policy during collection.
            action_clamp: The actions are clamped to [-action_clamp, action_clamp], in collection and in imagination.
            warmup_steps: Env steps that the replay storage must hold before `update` trains anything. It must be at
                least ``imagination_horizon`` and ``world_model_window``.
            *_component: Names of the components of the observation group ``world_model_obs_group``, see the module
                docstring. ``command_components`` are the keys of the command of the task model.
            buffer_steps: Replay storage length in vectorized env steps, (T, N) layout.
            proprio_history_length: Proprio observation history length of the policy.
            world_model_window: Window length k of the multi-step world-model fit.
            world_model_batches: World-model gradient steps per update.
            world_model_batch_size: Windows per world-model batch.
            world_model_failure_fraction: Fraction of each world-model batch drawn from windows that start within
                ``world_model_failure_horizon`` steps before a real termination; the dynamics and the failure head see
                them, the failure head importance-weights them.
            world_model_failure_horizon: Steps before a real termination in which a window counts as a failure window.
            world_model_holdout_envs: Fraction of envs never used for world-model / head training, only for held-out
                calibration diagnostics.
            world_model_only_iterations: World-model-only learning iterations at the start, no policy update.
            contact_loss_coef: Weight of the contact (BCE) loss relative to the state error of the world model.
            use_failure_head: Use the learned failure ("continue") head in imagination, ``d = 1 - (1 - d_analytic) (1 -
                p_fail)``; the head is trained either way and the gradient through ``p_fail`` is kept.
            failure_head_warmup_iterations: Iterations after the head was created before it affects the policy; it is
                trained from the start.
            failure_windows: Windows (of ``world_model_batch_size``) that train the head, the feature computation runs
                the FK.
            failure_clip: Upper clip of ``p_fail`` in imagination.
            failure_positive_weight: BCE positive weight; 1 because the failure oversampling is importance-weighted to
                keep the head calibrated.
            imagination_horizon: Imagination horizon H of the BPTT policy update.
            policy_batch_size: Imagination starts per policy update.
            gamma: Discount factor.
            lam: GAE lambda of the critic targets.
            use_critic: False = ablation without critic, i.e. no bootstrap value term in the return and no critic fit.
            critic_epochs: Critic fitting epochs per policy update.
            critic_minibatches: Critic minibatches per epoch.
            max_grad_norm: Gradient norm clip of the policy. The step is skipped if the norm is not finite.
            max_grad_value: Gradient value clip of the policy (same name as in `PPO`). Unlike `PPO` (value clip before
                the norm clip, default None), it is applied after the norm clip, as in the reference implementation.
            learning_rate: Initial learning rate of all the optimizers.
            lr_decay_iters: Iterations of the cosine decay from ``learning_rate`` to ``learning_rate * lr_final_ratio``,
                constant afterwards.
            lr_final_ratio: Final learning rate as a fraction of ``learning_rate``.
            optimizer_class_name: Class name in `torch.optim` of the optimizer of the policy; the same for the world
                model, the failure head and the critic, with the ``*_optimizer_kwargs`` added to the learning rate.
            amp: Autocast dtype of the MLP matmuls, "bf16" or "none".
            allow_tf32: Allow TF32 in matmuls and convolutions.
            compile: Use ``torch.compile`` on the imagination / world-model functions.
            compile_policy: ``torch.compile`` the imagination step, the reward / termination and the critic-feature
                functions, i.e. the level-batched FK.
            policy_autocast: bf16 autocast of the policy network in imagination only; collection / play stay fp32.
            checkpoint_imagination: Gradient checkpointing of every imagination step (recompute in backward): less
                memory, more compute.
            diagnostics_interval: Every this many learning iterations `update` also returns the held-out diagnostics as
                stats (the runner logs them as ``Train/...``).
            termination_check_interval: The same for the (more expensive) imagination vs reality check of the
                terminations.
            log_reward_gap: Put the gap between the reward of the task model and the reward of the simulator into
                ``infos["step"]`` (the runner logs it as ``Step/analytic_reward_abs_gap``).
            world_model_class_name: Class name of the world model, `module:Class` for a class outside of
                `instinct_rl.modules`. The same for the other networks. The ``*_kwargs`` are added to the arguments the
                dimensions of the task model determine.
            task_model_class_name: Class name of the `TaskModel`, `module:Class` or a name in `instinct_rl.env`.
        """
        if kwargs:
            print("\033[43;33mWarning: SuperMimic init got extra kwargs:", kwargs.keys(), "\033[0m")
        assert warmup_steps >= max(imagination_horizon, world_model_window)
        self.device = device

        self.action_noise = action_noise
        self.action_clamp = action_clamp
        self.warmup_steps = warmup_steps
        self.world_model_obs_group = world_model_obs_group
        self.root_pos_component = root_pos_component
        self.root_quat_component = root_quat_component
        self.root_lin_vel_component = root_lin_vel_component
        self.root_ang_vel_component = root_ang_vel_component
        self.joint_pos_component = joint_pos_component
        self.joint_vel_component = joint_vel_component
        self.last_action_component = last_action_component
        self.contact_component = contact_component
        self.context_component = context_component
        self.command_components = tuple(command_components)
        self.buffer_steps = buffer_steps
        self.proprio_history_length = proprio_history_length
        self.world_model_window = world_model_window
        self.world_model_batches = world_model_batches
        self.world_model_batch_size = world_model_batch_size
        self.world_model_failure_fraction = world_model_failure_fraction
        self.world_model_failure_horizon = world_model_failure_horizon
        self.world_model_holdout_envs = world_model_holdout_envs
        self.world_model_only_iterations = world_model_only_iterations
        self.contact_loss_coef = contact_loss_coef
        self.world_model_max_grad_norm = world_model_max_grad_norm
        self.use_failure_head = use_failure_head
        self.failure_head_warmup_iterations = failure_head_warmup_iterations
        self.failure_windows = failure_windows
        self.failure_clip = failure_clip
        self.failure_positive_weight = failure_positive_weight
        self.failure_head_max_grad_norm = failure_head_max_grad_norm
        self.imagination_horizon = imagination_horizon
        self.policy_batch_size = policy_batch_size
        self.gamma = gamma
        self.lam = lam
        self.use_critic = use_critic
        self.critic_epochs = critic_epochs
        self.critic_minibatches = critic_minibatches
        self.critic_max_grad_norm = critic_max_grad_norm
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value
        self.initial_learning_rate = learning_rate
        self.lr_decay_iters = lr_decay_iters
        self.lr_final_ratio = lr_final_ratio
        self.world_model_optimizer_class_name = world_model_optimizer_class_name
        self.world_model_optimizer_kwargs = world_model_optimizer_kwargs
        self.failure_head_optimizer_class_name = failure_head_optimizer_class_name
        self.failure_head_optimizer_kwargs = failure_head_optimizer_kwargs
        self.critic_optimizer_class_name = critic_optimizer_class_name
        self.critic_optimizer_kwargs = critic_optimizer_kwargs
        self.amp = amp
        self.allow_tf32 = allow_tf32
        self.compile = compile
        self.compile_policy = compile_policy
        self.policy_autocast = policy_autocast
        self.checkpoint_imagination = checkpoint_imagination
        self.diagnostics_interval = diagnostics_interval
        self.termination_check_interval = termination_check_interval
        self.log_reward_gap = log_reward_gap
        self.world_model_class_name = world_model_class_name
        self.world_model_kwargs = world_model_kwargs
        self.failure_head_class_name = failure_head_class_name
        self.failure_head_kwargs = failure_head_kwargs
        self.value_head_class_name = value_head_class_name
        self.value_head_kwargs = value_head_kwargs
        self.task_model_class_name = task_model_class_name
        self.task_model_kwargs = task_model_kwargs

        # SuperMimic components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = getattr(optim, optimizer_class_name)(
            self.actor_critic.parameters(), lr=learning_rate, **optimizer_kwargs
        )
        self.storage = None  # initialized later
        # the normalizer of the runner for the ``policy`` group (set by `init_storage`): the imagination normalizes the
        # imagined observations with it
        self.policy_normalizer = None

        # algorithm status
        self.current_learning_iteration = 0
        self.failure_head_start_iteration = 0  # iteration at which the head was created
        self.start_world_model_obs = None  # the world-model observation at the start of the transition being collected
        self.context_normalization_ready = False
        self.episode_step = None

    def init_storage(
        self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1, env=None, normalizers=None
    ):
        """Build everything that only training needs: the task model, the world model, the critic and the failure head
        with their optimizers, the replay storage and the (compiled) functions of the imagination.

        Args:
            num_transitions_per_env: Not used: this is the number of env steps per learning iteration, which sets the
                update-to-data ratio. The replay storage has ``buffer_steps`` steps.
            env: The `VecEnv` of the runner. The task model reads the constants of the task from it (only here).
            normalizers: The observation normalizers of the runner; the one of the ``policy`` group normalizes the
                imagined observations.
        """
        assert env is not None and "policy" in normalizers, "SuperMimic needs the env and a `policy` normalizer"
        device = self.device
        self.policy_normalizer = normalizers["policy"]
        set_amp_dtype(torch.bfloat16 if self.amp == "bf16" else None)
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        self.num_envs = num_envs
        self.policy_obs_format = obs_format["policy"]
        world_model_obs_format = obs_format[self.world_model_obs_group]
        self.component_slices = {name: get_obs_slice(world_model_obs_format, name) for name in world_model_obs_format}

        self.task_model = self.build_task_model(env, obs_format)
        task_model = self.task_model
        if not bool(task_model.max_command_length() > self.imagination_horizon):
            raise ValueError(
                f"No reference clip is longer than the imagination horizon ({self.imagination_horizon} frames)."
            )
        assert task_model.num_contacts == math.prod(world_model_obs_format[self.contact_component])
        assert task_model.context_dim == math.prod(world_model_obs_format[self.context_component])
        self.context = torch.zeros(num_envs, task_model.context_dim, device=device)  # raw context of every env
        self.context_normalized = task_model.normalize_context(self.context)
        self.episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.world_model = instinct_modules.build_world_model_module(
            self.world_model_class_name,
            num_joints=task_model.num_joints,
            context_dim=task_model.context_dim,
            scan_dim=task_model.scan_dim,
            num_contacts=task_model.num_contacts,
            z_dim=self.z_dim,
            **self.world_model_kwargs,
        ).to(device)
        self.critic = instinct_modules.build_world_model_module(
            self.value_head_class_name, in_dim=task_model.critic_dim, **self.value_head_kwargs
        ).to(device)
        self.failure_head = instinct_modules.build_world_model_module(
            self.failure_head_class_name, in_dim=task_model.critic_dim, **self.failure_head_kwargs
        ).to(device)
        self.world_model_parameters = self.collect_world_model_parameters()
        self.world_model_optimizer = getattr(optim, self.world_model_optimizer_class_name)(
            self.world_model_parameters, lr=self.learning_rate, **self.world_model_optimizer_kwargs
        )
        self.critic_optimizer = getattr(optim, self.critic_optimizer_class_name)(
            self.critic.parameters(), lr=self.learning_rate, **self.critic_optimizer_kwargs
        )
        self.failure_head_optimizer = getattr(optim, self.failure_head_optimizer_class_name)(
            self.failure_head.parameters(), lr=self.learning_rate, **self.failure_head_optimizer_kwargs
        )
        self.error_floor = error_floor(task_model.num_joints).to(device)

        self.storage = ReplayStorage(
            self.buffer_steps,
            num_envs,
            self.storage_fields(num_actions),
            device,
            self.world_model_holdout_envs,
        )
        # the fields of the storage which the samplers read
        self.command_keys = self.command_components
        self.next_command_keys = tuple(key + "1" for key in self.command_components)
        self.window_keys = ("s", "s1", "a", "contact", "terminated") + self.next_command_keys
        self.start_keys = ("s", "a_prev") + self.command_keys
        self.build_functions()

    def build_task_model(self, env, obs_format):
        """Build the `TaskModel` of the task from ``task_model_class_name``."""
        if ":" in self.task_model_class_name:
            module_name, class_name = self.task_model_class_name.split(":")
            task_model_class = getattr(importlib.import_module(module_name), class_name)
        else:
            task_model_class = getattr(instinct_env, self.task_model_class_name)
        return task_model_class(env=env, obs_format=obs_format, device=self.device, **self.task_model_kwargs)

    def storage_fields(self, num_actions: int) -> dict[str, tuple]:
        """The fields of the replay storage as {name: (shape, dtype)}. A plug-in adds the fields of its own, which
        `transition_fields` then has to fill."""
        task_model = self.task_model
        state_dim = 13 + 2 * task_model.num_joints
        field_specs = {
            "s": (state_dim, torch.float32),
            "s1": (state_dim, torch.float32),
            "a": (num_actions, torch.float32),
            "a_prev": (num_actions, torch.float32),
            "contact": (task_model.num_contacts, torch.float32),  # indicator of the undesired-contact bodies at s1
            "r": ((), torch.float32),
            "terminated": ((), torch.bool),  # the transition ended in a REAL termination (not a timeout / dataset end)
            "ep_step": ((), torch.int32),
            "done": ((), torch.bool),
        }
        for name in self.command_components:
            _, shape = self.component_slices[name]
            command_shape = () if shape == (1,) else tuple(shape)
            field_specs[name] = (command_shape, torch.float32)  # the command at s
            field_specs[name + "1"] = (command_shape, torch.float32)  # ... at s1 (pre-reset)
        return field_specs

    def collect_world_model_parameters(self) -> list:
        """The parameters that the optimizer of the world model trains. A plug-in adds the modules it trains with it."""
        return [*self.world_model.parameters()]

    def build_functions(self):
        """The (compiled) functions of the world-model fit and of the imagination."""
        task_model, world_model = self.task_model, self.world_model
        window = self.world_model_window
        world_model_step, policy_action = self.world_model_step, self.policy_action

        def world_model_rollout(states, actions, context_normalized, initial_latent=None):
            """k-step rollout; with the initial latent also the latents (B, k, Z), else None."""
            state, latent = unflatten_state(states[:, 0]), initial_latent
            predicted_states, contact_logits, latents = [], [], []
            for step in range(window):
                state, logits, latent = world_model_step(
                    state, actions[:, step], context_normalized, task_model.scan(state), latent
                )
                predicted_states.append(state)
                contact_logits.append(logits)
                if latent is not None:
                    latents.append(latent)
            predicted_states = tuple(
                torch.stack([state[index] for state in predicted_states], 1) for index in range(NUM_STATE_COMPONENTS)
            )  # (B, k, ...)
            return predicted_states, torch.stack(contact_logits, 1), torch.stack(latents, 1) if latents else None

        def world_model_error(predicted_states, contact_logits, next_states, contact, mask):
            """The state error in units of the (floored) std of the state increments and the contact BCE, summed over
            the steps of the window and averaged over the windows. Also the state error at the last step (B,)."""
            std = torch.maximum(world_model.out_norm.std(), self.error_floor)
            state_error = ((world_model.state_error(predicted_states, unflatten_state(next_states)) / std) ** 2).mean(
                -1
            )  # (B, k)
            contact_bce = F.binary_cross_entropy_with_logits(contact_logits, contact, reduction="none").sum(-1)
            num_windows = next_states.shape[0]
            return (
                (state_error * mask).sum() / num_windows,
                (contact_bce * mask).sum() / num_windows,
                state_error[:, -1],
            )

        def reward_and_termination(state, command, action, previous_action, context, contact_logits):
            reward, reward_terms = task_model.reward(
                state, command, action, previous_action, context, torch.sigmoid(contact_logits)
            )
            return reward, task_model.soft_termination(state, command), reward_terms

        def critic_input(state, command, context_normalized):
            return task_model.critic_features(state, command, context_normalized)

        def imagination_step(state, proprio_history, command, latent, context, context_normalized, scan):
            """One imagination step: policy observation in the env's format, actor, world-model step, terrain scan and
            proprio history of the next state. ``latent`` is the visual latent state (None without one): an input of the
            policy and of the step, which returns the next latent as well (else None)."""
            action = policy_action(state, proprio_history, command, latent)
            action = torch.clamp(action, -self.action_clamp, self.action_clamp)
            next_state, contact_logits, next_latent = world_model_step(state, action, context_normalized, scan, latent)
            proprio = task_model.proprio_observation(next_state, action, context)
            proprio = proprio + (2 * torch.rand_like(proprio) - 1) * task_model.proprio_noise
            next_proprio_history = torch.cat((proprio_history[:, 1:], proprio.unsqueeze(1)), 1)
            return (
                action,
                next_state,
                contact_logits,
                task_model.scan(next_state),
                next_proprio_history,
                next_latent,
            )

        if self.compile:
            world_model_rollout = torch.compile(world_model_rollout)
            world_model_error = torch.compile(world_model_error)
        if self.compile_policy:
            # level-batched FK keeps the graph small enough to compile (the former unrolled 30-body loop did not finish)
            reward_and_termination = torch.compile(reward_and_termination, dynamic=True)
            critic_input = torch.compile(critic_input, dynamic=True)
            imagination_step = torch.compile(imagination_step)
        self.world_model_rollout, self.world_model_error = world_model_rollout, world_model_error
        self.reward_and_termination, self.critic_input = reward_and_termination, critic_input
        self.imagination_step = imagination_step

    def world_model_step(self, state, action, context_normalized, scan, latent):
        """One step of the world model (traced by the compiled functions).

        Args:
            latent: The visual latent state (B, Z); None without one.

        Returns:
            The next state tuple, the contact logits at it and the next latent (None without one).
        """
        next_state, contact_logits = self.world_model.step(state, action, context_normalized, scan)
        return next_state, contact_logits, None

    def normalize_policy_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize a policy observation with the statistics of the runner's normalizer. NOTE: calling the normalizer
        would also update its statistics (it is in train mode during training), here they must stay fixed."""
        return (obs - self.policy_normalizer.mean) / (self.policy_normalizer.std + self.policy_normalizer.eps)

    def policy_action(self, state, proprio_history, command, latent):
        """Action of the policy in imagination (traced by the compiled functions): the observation is assembled in
        EXACTLY the env's ``policy`` group format from the imagined state, the normalized observation goes into the
        actor critic. Gradients flow into the policy and (through the state) into the world model.

        Args:
            latent: The visual latent state (B, Z); None without one.
        """
        obs = self.task_model.policy_observation(proprio_history, state, command)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.policy_autocast):  # training imagination only
            return self.actor_critic.act_inference(self.normalize_policy_observation(obs)).float()

    def rollout(self, batch: dict, context_normalized: torch.Tensor):
        """k-step world-model rollout of a window batch: predicted states, contact logits and the predicted latents
        (from ``batch["initial_latent"]`` at the window start; None without a visual latent)."""
        return self.world_model_rollout(batch["s"], batch["a"], context_normalized, batch.get("initial_latent"))

    """
    Collection
    """

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    @torch.no_grad()
    def act(self, obs, extra_obs):
        """The deterministic policy (the mean of the action distribution) plus Gaussian exploration noise.

        Args:
            obs: the (normalized) policy observation.
            extra_obs: all observation groups of the step, not used (the policy acts on ``obs``).
        """
        assert not torch.is_inference_mode_enabled(), (
            "The imagination differentiates through buffers that the rollout updates (the normalizers of the world"
            " model and of the policy): set `inference_mode_rollout` to False in the runner config."
        )
        self.actor_critic.act(obs)  # only updates the action distribution, whose std the runner logs
        actions = self.actor_critic.action_mean
        actions = actions + self.action_noise * torch.randn_like(actions)
        return torch.clamp(actions, -self.action_clamp, self.action_clamp)

    def component(self, world_model_obs: torch.Tensor, name: str) -> torch.Tensor:
        """The component ``name`` of a flat world-model observation, shaped (num_envs, *shape); a component of shape
        (1,) is a scalar (num_envs,)."""
        obs_slice, shape = self.component_slices[name]
        value = world_model_obs[..., obs_slice]
        return value.reshape(self.num_envs) if shape == (1,) else value.reshape(self.num_envs, *shape)

    def read_world_model_obs(self, world_model_obs: torch.Tensor) -> dict:
        """Split a flat world-model observation into the state tuple ``(p, R, v, w, q, qd)``, the command (dict), the
        last action and the contacts."""
        state = (
            self.component(world_model_obs, self.root_pos_component),
            quat_to_mat(self.component(world_model_obs, self.root_quat_component)),
            self.component(world_model_obs, self.root_lin_vel_component),
            self.component(world_model_obs, self.root_ang_vel_component),
            self.component(world_model_obs, self.joint_pos_component),
            self.component(world_model_obs, self.joint_vel_component),
        )
        return dict(
            world_model_obs=world_model_obs,
            state=state,
            command={name: self.component(world_model_obs, name) for name in self.command_components},
            last_action=self.component(world_model_obs, self.last_action_component),
            contact=self.component(world_model_obs, self.contact_component),
        )

    @torch.no_grad()
    def process_env_step(self, rewards, dones, infos, next_obs, next_extra_obs):
        """Store the transition of the step. Its start is the world-model observation of the previous step (after the
        reset of the finished envs), its end is the terminal observation of this step (before the reset).

        The first call of a run has no start observation: it only takes the observation and the context of the envs.
        """
        next_world_model_obs = next_extra_obs[self.world_model_obs_group]
        self.update_context(next_world_model_obs)
        if self.start_world_model_obs is not None:
            self.add_transition(rewards, dones, infos)
        self.start_world_model_obs = next_world_model_obs
        self.episode_step = (self.episode_step + 1) * (~dones.bool())
        self.actor_critic.reset(dones)

    def update_context(self, world_model_obs: torch.Tensor):
        """Overwrite the context table from the observation (it is constant during an episode, and the DR of the env
        keeps it constant during a run). The statistics of its normalization are fixed on the first observation."""
        self.context.copy_(self.component(world_model_obs, self.context_component))
        if not self.context_normalization_ready:
            self.task_model.fit_context_normalization(self.context)
            self.context_normalization_ready = True

    def add_transition(self, rewards, dones, infos):
        task_model, world_model = self.task_model, self.world_model
        start = self.read_world_model_obs(self.start_world_model_obs)
        end = self.read_world_model_obs(infos["terminal_observations"][self.world_model_obs_group])
        done = dones.bool()
        end["reward"] = rewards.reshape(self.num_envs, -1).sum(-1)
        end["done"] = done
        end["terminated"] = done & ~infos["time_outs"].bool()  # a real termination, not a timeout / dataset end
        start["episode_step"] = self.episode_step
        self.storage.add_transitions(self.transition_fields(start, end))
        # statistics of the input / output of the world model
        context_normalized = task_model.normalize_context(self.context)
        world_model.in_norm.update(
            world_model.inputs(start["state"], end["last_action"], context_normalized, task_model.scan(start["state"]))
        )
        world_model.out_norm.update(world_model.targets(start["state"], end["state"]))
        if self.log_reward_gap:
            analytic_reward = task_model.reward(
                end["state"], end["command"], end["last_action"], start["last_action"], self.context, end["contact"]
            )[0]
            infos["step"]["analytic_reward_abs_gap"] = (analytic_reward - end["reward"]).abs().mean() / task_model.dt

    def transition_fields(self, start: dict, end: dict) -> dict:
        """The values of one transition for every field of `storage_fields` ({name: tensor of (num_envs, ...)}).
        A plug-in adds its fields.

        Args:
            start: The data before the step (`read_world_model_obs` of its observation) and ``episode_step``.
            end: The data after the step, before the reset: the same as ``start``, with ``last_action`` the action that
                was applied, and ``reward`` (of the simulator), ``done`` and ``terminated``.
        """
        return {
            "s": flatten_state(start["state"]),
            "s1": flatten_state(end["state"]),
            "a": end["last_action"],
            "a_prev": start["last_action"],
            "contact": end["contact"],
            "r": end["reward"],
            "ep_step": start["episode_step"].int(),
            "done": end["done"],
            "terminated": end["terminated"],
            **start["command"],
            **{key + "1": value for key, value in end["command"].items()},
        }

    def compute_returns(self, last_extra_obs):
        """Nothing to compute: the policy is trained by back-propagation through the imagination, not on returns of the
        collected rollout."""
        pass

    """
    Update
    """

    def update(self, current_learning_iteration):
        """One learning iteration: ``world_model_batches`` world-model steps (the world model and the failure head),
        then (after ``world_model_only_iterations``) one policy step (the policy and the critic). Does nothing while
        the replay storage holds less than ``warmup_steps`` env steps.

        Returns:
            mean_losses: {name: tensor}, the losses of the steps, averaged over the steps they were computed in.
            average_stats: {name: tensor}, the statistics of the steps and, every ``diagnostics_interval`` iterations,
                the diagnostics.
        """
        self.current_learning_iteration = current_learning_iteration
        if self.storage.size < self.warmup_steps:
            return dict(), dict()
        self.set_learning_rate(self.learning_rate_at(current_learning_iteration))
        self.context_normalized = self.task_model.normalize_context(self.context)
        losses_per_step, stats_per_step = defaultdict(list), defaultdict(list)
        for _ in range(self.world_model_batches):
            losses, stats = self.update_world_model()
            self.append_values(losses_per_step, losses)
            self.append_values(stats_per_step, stats)
        if current_learning_iteration >= self.world_model_only_iterations:
            losses, stats = self.update_policy(current_learning_iteration)
            self.append_values(losses_per_step, losses)
            self.append_values(stats_per_step, stats)
        mean_losses = {key: torch.stack(values).mean() for key, values in losses_per_step.items()}
        average_stats = {key: torch.stack(values).mean() for key, values in stats_per_step.items()}
        if current_learning_iteration % self.diagnostics_interval == 0:
            termination_check = current_learning_iteration % self.termination_check_interval == 0
            for key, value in self.compute_diagnostics(termination_check).items():
                average_stats[key] = torch.tensor(value, device=self.device)
        return mean_losses, average_stats

    @staticmethod
    def append_values(values_per_step: dict, values: dict):
        for key, value in values.items():
            values_per_step[key].append(value.detach())

    def learning_rate_at(self, current_learning_iteration: int) -> float:
        """Cosine decay from the initial to the final learning rate, constant afterwards."""
        if current_learning_iteration >= self.lr_decay_iters:
            return self.initial_learning_rate * self.lr_final_ratio
        progress = current_learning_iteration / self.lr_decay_iters
        return self.initial_learning_rate * (
            self.lr_final_ratio + (1 - self.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        )

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate
        for optimizer in (
            self.optimizer,
            self.world_model_optimizer,
            self.critic_optimizer,
            self.failure_head_optimizer,
        ):
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

    def total_loss(self, losses: dict) -> torch.Tensor:
        loss = 0.0
        for key, value in losses.items():
            loss += getattr(self, key + "_coef", 1.0) * value
        return loss

    def gradient_step(
        self,
        optimizer: optim.Optimizer,
        parameters: list,
        loss: torch.Tensor,
        max_grad_norm: float,
        max_grad_value: float | None = None,
        skip_nonfinite_grad: bool = False,
    ) -> torch.Tensor:
        """Perform a gradient step of the optimizer towards the loss.
        NOTE: SuperMimic runs in a single process, there is no gradient reduction as in `PPO.gradient_step`.

        Args:
            max_grad_norm: The gradient norm is clipped to it.
            max_grad_value: If given, the gradient values are clipped to it after the norm.
            skip_nonfinite_grad: Do not step if the gradient norm is not finite.

        Returns:
            The gradient norm before the clip.
        """
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        if max_grad_value is not None:
            nn.utils.clip_grad_value_(parameters, max_grad_value)
        if not skip_nonfinite_grad or torch.isfinite(grad_norm):
            optimizer.step()
        return grad_norm

    """
    World model and failure head
    """

    def update_world_model(self):
        """One gradient step of the world model and one of the failure head on a batch of k-step windows.

        Returns:
            The losses and the stats of both.
        """
        batch = self.storage.sample_windows(
            self.world_model_batch_size,
            self.world_model_window,
            self.window_keys,
            self.storage.train_envs,
            self.world_model_failure_fraction,
            self.world_model_failure_horizon,
        )
        losses, inter_vars, stats = self.compute_world_model_losses(batch)
        self.gradient_step(
            self.world_model_optimizer,
            self.world_model_parameters,
            self.total_loss(losses),
            self.world_model_max_grad_norm,
        )
        # failure ("continue") head on TRUE and (detached) WM-PREDICTED states of the windows, importance-weighted
        failure_losses, _, failure_stats = self.compute_failure_losses(batch, inter_vars["predicted_states"])
        self.gradient_step(
            self.failure_head_optimizer,
            self.failure_head_parameters,
            self.total_loss(failure_losses),
            self.failure_head_max_grad_norm,
        )
        return {**losses, **failure_losses}, {**stats, **failure_stats}

    @property
    def failure_head_parameters(self) -> list:
        return [*self.failure_head.parameters()]

    def compute_world_model_losses(self, batch: dict):
        """Rollout and losses of the world model on a window batch (the plug-ins add to it).

        Args:
            batch: The windows of `ReplayStorage.sample_windows`.

        Returns:
            The losses (``state_error_loss``, ``contact_loss``), the intermediate variables (the predicted states of the
            windows ``predicted_states`` (tuple of (B, k, ...)), the ``contact_logits`` and the ``predicted_latents``)
            and the stats (the state error at the last step of the windows).
        """
        predicted_states, contact_logits, predicted_latents = self.rollout(batch, self.context_normalized[batch["env"]])
        state_error_loss, contact_loss, last_step_error = self.world_model_error(
            predicted_states, contact_logits, batch["s1"], batch["contact"], batch["mask"]
        )
        last_step_mask = batch["mask"][:, -1]
        losses = dict(state_error_loss=state_error_loss, contact_loss=contact_loss)
        inter_vars = dict(
            predicted_states=predicted_states,
            contact_logits=contact_logits,
            predicted_latents=predicted_latents,
        )
        stats = dict(
            last_step_state_error=(last_step_error.detach() * last_step_mask).sum() / last_step_mask.sum().clamp(min=1)
        )
        return losses, inter_vars, stats

    def compute_failure_losses(self, batch: dict, predicted_states: tuple):
        """BCE of the failure head against the real terminations of the windows, on the true and the predicted states,
        weighted by the validity of the step and the importance weight of the window."""
        window = batch["mask"].shape[1]
        num_windows = min(self.failure_windows, batch["mask"].shape[0])

        def head_windows(value):
            return value[:num_windows].flatten(0, 1)

        with torch.no_grad():
            context_normalized = self.context_normalized[batch["env"][:num_windows]]
            context_normalized = context_normalized.unsqueeze(1).expand(num_windows, window, -1).flatten(0, 1)
            commands = {key: head_windows(batch[key + "1"]) for key in self.command_keys}
            true_features = self.critic_input(unflatten_state(head_windows(batch["s1"])), commands, context_normalized)
            predicted_features = self.critic_input(
                tuple(head_windows(state) for state in predicted_states), commands, context_normalized
            )
            self.failure_head.norm.update(true_features)
        labels = head_windows(batch["terminated"]).float()
        weights = (batch["mask"][:num_windows] * batch["weight"][:num_windows, None]).flatten()
        positive_weights = torch.where(labels > 0.5, self.failure_positive_weight, 1.0)

        def head_loss(features):
            bce = F.binary_cross_entropy_with_logits(self.failure_head(features), labels, reduction="none")
            return (bce * positive_weights * weights).sum() / weights.sum().clamp(min=1e-6)

        losses = dict(
            failure_loss_true_states=head_loss(true_features),
            failure_loss_predicted_states=head_loss(predicted_features),
        )
        stats = dict(
            failure_positive_ratio=(labels * batch["mask"][:num_windows].flatten()).sum()
            / batch["mask"][:num_windows].sum()
        )
        return losses, dict(), stats

    """
    Policy and critic
    """

    def sample_valid_starts(self):
        """Uniform over all buffered states whose imagination stays inside the reference (``imagination_valid``)."""
        eligible = self.storage.eligible_starts(
            lambda command: self.task_model.imagination_valid(command, self.imagination_horizon), self.command_keys
        )
        if len(eligible) == 0:
            raise RuntimeError("No buffered state has a full imagination horizon left inside its reference clip.")
        pick = eligible[torch.randint(0, len(eligible), (self.policy_batch_size,), device=self.device)]
        return self.storage.starts_at(
            pick[:, 0], pick[:, 1], self.proprio_history_length, self.start_keys, ("s", "a_prev")
        )

    def start_latent(self, starts: dict):
        """The visual latent state at the imagination start states (None without one)."""
        return None

    def imagine(self, starts: dict, histories: dict) -> dict:
        """H-step imagination of the policy from buffer start states / histories (world model + visual latent state, if
        any). Gradients flow as in `update_policy` (call under ``no_grad`` for diagnostics)."""
        task_model = self.task_model
        horizon = self.imagination_horizon
        state = unflatten_state(starts["s"])
        command = {key: starts[key] for key in self.command_keys}
        previous_action, env_indices = starts["a_prev"], starts["env"]
        context, context_normalized = self.context[env_indices], self.context_normalized[env_indices]
        latent = self.start_latent(starts)  # at the start state it comes from the REAL buffered image
        with torch.no_grad():
            proprio_history = task_model.proprio_observation(
                unflatten_state(histories["s"]), histories["a_prev"], self.context[histories["env"]].unsqueeze(1)
            )
            # one noise sample per distinct historical observation (backfilled slots duplicate it, as in collection)
            noise = (2 * torch.rand_like(proprio_history) - 1) * task_model.proprio_noise
            proprio_history = proprio_history + noise.gather(1, histories["slot"].unsqueeze(-1).expand_as(noise))
        start_state, start_command, scan = state, command, task_model.scan(state)
        states, actions, previous_actions, commands, contact_logits = [], [], [], [], []
        for _ in range(horizon):
            previous_actions.append(previous_action)
            # ---- policy observation in EXACTLY the env's policy-group format ----
            step_args = (state, proprio_history, command, latent, context, context_normalized, scan)
            if self.checkpoint_imagination and torch.is_grad_enabled():
                # recompute the step in the backward pass instead of keeping its activations (same RNG state is
                # replayed)
                action, state, logits, scan, proprio_history, latent = torch.utils.checkpoint.checkpoint(
                    self.imagination_step, *step_args, use_reentrant=False
                )
            else:
                action, state, logits, scan, proprio_history, latent = self.imagination_step(*step_args)
            previous_action = action
            command = task_model.advance_command(command)
            states.append(state)
            actions.append(action)
            commands.append(command)
            contact_logits.append(logits)
        return dict(
            start_state=start_state,
            start_command=start_command,
            final_state=state,
            final_command=command,
            context=context,
            context_normalized=context_normalized,
            states=states,
            actions=actions,
            previous_actions=previous_actions,
            commands=commands,
            contact_logits=contact_logits,
        )

    def update_policy(self, current_learning_iteration: int):
        """One policy step: H-step imagination from buffer states, BPTT of the discounted return, then the GAE critic
        fit.

        Returns:
            The losses and the stats of both.
        """
        starts, histories = self.sample_valid_starts()
        frozen_modules = (self.world_model, self.critic, self.failure_head)
        for module in frozen_modules:
            module.requires_grad_(False)
        imagination = self.imagine(starts, histories)
        losses, inter_vars, stats = self.compute_policy_losses(imagination, current_learning_iteration)
        stats["grad_norm"] = self.gradient_step(
            self.optimizer,
            [*self.actor_critic.parameters()],
            self.total_loss(losses),
            self.max_grad_norm,
            self.max_grad_value,
            skip_nonfinite_grad=True,
        )
        for module in frozen_modules:
            module.requires_grad_(True)
        if self.use_critic:
            critic_losses, critic_stats = self.update_critic(imagination, inter_vars["rewards"], inter_vars["dones"])
            losses, stats = {**losses, **critic_losses}, {**stats, **critic_stats}
        return losses, stats

    def compute_policy_losses(self, imagination: dict, current_learning_iteration: int):
        """The loss of the policy: the negative discounted return of the imagination (the rewards and terminations of
        the task model, the failure head and the critic as the bootstrap value at the end).

        Returns:
            The loss (``policy_loss``), the detached ``rewards`` and ``dones`` (H, B) of the imagined transitions and
            the stats: the imagined reward per second, the survival at the end, the weighted reward terms.
        """
        task_model = self.task_model
        horizon, batch_size = self.imagination_horizon, self.policy_batch_size
        states = stack_states(imagination["states"])  # (H * B, ...)
        commands = stack_commands(imagination["commands"], self.command_keys)

        def repeat_over_horizon(value):
            return value.unsqueeze(0).expand(horizon, *value.shape).flatten(0, 1)

        rewards, dones, reward_terms = self.reward_and_termination(
            states,
            commands,
            torch.stack(imagination["actions"]).flatten(0, 1),
            torch.stack(imagination["previous_actions"]).flatten(0, 1),
            repeat_over_horizon(imagination["context"]),
            torch.stack(imagination["contact_logits"]).flatten(0, 1),
        )
        failure_probability_mean = torch.zeros((), device=self.device)
        if self.use_failure_head and current_learning_iteration >= (
            self.failure_head_start_iteration + self.failure_head_warmup_iterations
        ):
            # learned failure ("continue") head, combined with the analytic soft termination; the gradient is kept
            features = self.critic_input(states, commands, repeat_over_horizon(imagination["context_normalized"]))
            failure_probability = torch.sigmoid(self.failure_head(features))
            failure_probability = failure_probability.clamp(max=self.failure_clip)
            dones = 1 - (1 - dones) * (1 - failure_probability)
            failure_probability_mean = failure_probability.detach().mean()
        rewards, dones = rewards.view(horizon, batch_size), dones.view(horizon, batch_size)
        alive_after = torch.cumprod(1 - dones, 0)  # c_{t+1}
        alive_before = torch.cat((torch.ones(1, batch_size, device=self.device), alive_after[:-1]), 0)  # c_t
        discounts = self.gamma ** torch.arange(horizon, device=self.device, dtype=torch.float32).unsqueeze(-1)
        discounted_return = (discounts * alive_before * rewards).sum(0)
        survival = alive_after[-1]
        reward_sum = (alive_before * rewards).detach().sum(0)
        if self.use_critic:
            final_features = self.critic_input(
                imagination["final_state"], imagination["final_command"], imagination["context_normalized"]
            )
            discounted_return = discounted_return + (self.gamma**horizon) * survival * self.critic(final_features)
        losses = dict(policy_loss=-discounted_return.mean())
        inter_vars = dict(rewards=rewards.detach(), dones=dones.detach())
        stats = dict(
            imagination_reward=(reward_sum / horizon / task_model.dt).mean().detach(),
            survival=survival.mean().detach(),
            failure_probability=failure_probability_mean,
        )
        for name, term in reward_terms.items():
            stats["reward_term/" + name] = term.detach().view(horizon, batch_size).mean()
        return losses, inter_vars, stats

    def update_critic(self, imagination: dict, rewards: torch.Tensor, dones: torch.Tensor):
        """Fit the critic to the GAE(lambda) targets of the imagined transitions, ``critic_epochs`` epochs of
        ``critic_minibatches`` minibatches.

        Args:
            rewards: The imagined rewards (H, B), detached.
            dones: The imagined termination probabilities (H, B), detached.
        """
        features, targets = self.compute_critic_targets(imagination, rewards, dones)
        num_samples = features.shape[0]
        minibatch_size = num_samples // self.critic_minibatches
        losses_per_step = defaultdict(list)
        for _ in range(self.critic_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for index in range(self.critic_minibatches):
                batch_indices = permutation[index * minibatch_size : (index + 1) * minibatch_size]
                losses, _, _ = self.compute_critic_losses(features[batch_indices], targets[batch_indices])
                self.gradient_step(
                    self.critic_optimizer,
                    [*self.critic.parameters()],
                    self.total_loss(losses),
                    self.critic_max_grad_norm,
                )
                self.append_values(losses_per_step, losses)
        return {key: torch.stack(values).mean() for key, values in losses_per_step.items()}, dict()

    def compute_critic_targets(self, imagination: dict, rewards: torch.Tensor, dones: torch.Tensor):
        """GAE(lambda) targets of the imagined transitions (H, B) and the critic features of their start states."""
        horizon, batch_size = self.imagination_horizon, self.policy_batch_size
        with torch.no_grad():
            states = tuple(
                torch.cat(
                    (
                        imagination["start_state"][index].unsqueeze(0),
                        torch.stack([state[index].detach() for state in imagination["states"]]),
                    ),
                    0,
                ).flatten(0, 1)
                for index in range(NUM_STATE_COMPONENTS)
            )
            commands = {
                key: torch.cat(
                    (
                        imagination["start_command"][key].unsqueeze(0),
                        torch.stack([command[key] for command in imagination["commands"]]),
                    ),
                    0,
                ).flatten(0, 1)
                for key in self.command_keys
            }
            context_normalized = imagination["context_normalized"]
            context_normalized = context_normalized.unsqueeze(0).expand(horizon + 1, *context_normalized.shape)
            context_normalized = context_normalized.flatten(0, 1)
            features = self.critic_input(states, commands, context_normalized).view(horizon + 1, batch_size, -1)
        self.critic.norm.update(features)
        with torch.no_grad():
            values = self.critic(features)
            target = values[-1]
            targets = []
            for step in range(horizon - 1, -1, -1):
                target = rewards[step] + self.gamma * (1 - dones[step]) * (
                    (1 - self.lam) * values[step + 1] + self.lam * target
                )
                targets.append(target)
            targets = torch.stack(targets[::-1])  # (H, B)
        return features[:-1].reshape(-1, features.shape[-1]), targets.reshape(-1)

    def compute_critic_losses(self, features: torch.Tensor, targets: torch.Tensor):
        return dict(value_loss=((self.critic(features) - targets) ** 2).mean()), dict(), dict()

    """
    Diagnostics
    """

    @torch.no_grad()
    def compute_diagnostics(self, termination_check: bool) -> dict:
        """Held-out diagnostics on real windows of the storage (the plug-ins add to it).

        Args:
            termination_check: Also run the (more expensive) imagination vs reality check of the terminations.

        Returns:
            {name: float}
        """
        diagnostics = self.fail_calibration()
        if termination_check:
            diagnostics.update(self.termination_check())
        return diagnostics

    def fail_calibration(self, num_windows: int = 2048) -> dict:
        """Failure head vs actual REAL terminations on held-out windows (envs never used for training), on true states
        and on open-loop WM-predicted states: mean predicted vs actual rate, reliability bins, AUROC, per command
        group."""
        window = self.world_model_window
        batch = self.storage.sample_windows(num_windows, window, self.window_keys, self.storage.holdout_envs)
        env_indices = batch["env"]
        context_normalized = (
            self.context_normalized[env_indices].unsqueeze(1).expand(num_windows, window, -1).flatten(0, 1)
        )
        commands = {key: batch[key + "1"].flatten(0, 1) for key in self.command_keys}
        predicted_states, _, _ = self.rollout(batch, self.context_normalized[env_indices])
        valid = batch["mask"].flatten() > 0.5
        labels = batch["terminated"].flatten()[valid].float()
        diagnostics = {}
        command_group = self.task_model.command_group(commands)[valid]
        for name, state in (
            ("true_states", unflatten_state(batch["s1"].flatten(0, 1))),
            ("wm_states", tuple(x.flatten(0, 1) for x in predicted_states)),
        ):
            features = self.critic_input(state, commands, context_normalized)
            probability = torch.sigmoid(self.failure_head(features))[valid]
            diagnostics[f"fail/{name}/pred_rate"] = float(probability.mean())
            diagnostics[f"fail/{name}/true_rate"] = float(labels.mean())
            for low, high in ((0.0, 0.02), (0.02, 0.1), (0.1, 0.5), (0.5, 1.01)):
                in_bin = (probability >= low) & (probability < high)
                if in_bin.any():
                    diagnostics[f"fail/{name}/bin_{low}_{high}/pred"] = float(probability[in_bin].mean())
                    diagnostics[f"fail/{name}/bin_{low}_{high}/actual"] = float(labels[in_bin].mean())
            if 0 < labels.sum() < len(labels):
                order = probability.argsort()
                ranks = torch.empty_like(probability)
                ranks[order] = torch.arange(1, len(probability) + 1, device=self.device, dtype=probability.dtype)
                num_positives = labels.sum()
                diagnostics[f"fail/{name}/auroc"] = float(
                    (ranks[labels > 0.5].sum() - num_positives * (num_positives + 1) / 2)
                    / (num_positives * (len(probability) - num_positives))
                )
            if name == "true_states":
                for group in command_group.unique().tolist():
                    in_group = command_group == group
                    diagnostics[f"fail/motion_{group}/pred_rate"] = float(probability[in_group].mean())
                    diagnostics[f"fail/motion_{group}/true_rate"] = float(labels[in_group].mean())
        return diagnostics

    def termination_check(self, num_starts: int = 256) -> dict:
        """Imagination vs reality from the same held-out real start states: fraction of the starts that terminate
        within the horizon in imagination (analytic hard termination; and 1 - survival with the failure head) vs in
        the buffer."""
        task_model = self.task_model
        horizon = self.imagination_horizon
        eligible = self.storage.eligible_starts(
            lambda command: task_model.imagination_valid(command, horizon),
            self.command_keys,
            self.storage.holdout_envs,
            max_age=self.storage.size - horizon - 1,
        )
        if len(eligible) < 8:
            return {}
        pick = eligible[torch.randint(0, len(eligible), (min(num_starts, len(eligible)),), device=self.device)]
        starts, histories = self.storage.starts_at(
            pick[:, 0], pick[:, 1], self.proprio_history_length, self.start_keys, ("s", "a_prev")
        )
        imagination = self.imagine(starts, histories)
        batch_size = len(pick)
        states = stack_states(imagination["states"])
        commands = stack_commands(imagination["commands"], self.command_keys)
        context_normalized = imagination["context_normalized"].unsqueeze(0).expand(horizon, batch_size, -1)
        context_normalized = context_normalized.flatten(0, 1)
        hard_termination = task_model.hard_termination(states, commands).view(horizon, batch_size).any(0)
        soft_termination = task_model.soft_termination(states, commands).view(horizon, batch_size)
        failure_probability = torch.sigmoid(self.failure_head(self.critic_input(states, commands, context_normalized)))
        failure_probability = failure_probability.view(horizon, batch_size).clamp(max=self.failure_clip)
        survival_with_head = ((1 - soft_termination) * (1 - failure_probability)).prod(0)
        # real outcome: a REAL termination within the next H transitions of the same episode
        rows = self.storage._row(pick[:, 0].unsqueeze(-1) + torch.arange(horizon, device=self.device))
        env_indices = pick[:, 1].unsqueeze(-1).expand_as(rows)
        done = self.storage.data["done"][rows, env_indices]
        terminated = self.storage.data["terminated"][rows, env_indices]
        same_episode = torch.cumsum(done.int(), 1) - done.int() == 0
        real = (terminated & same_episode).any(1)
        diagnostics = {
            "termcheck/imagined_hard": float(hard_termination.float().mean()),
            "termcheck/imagined_soft_with_head": float((1 - survival_with_head).mean()),
            "termcheck/real": float(real.float().mean()),
        }
        command_group = task_model.command_group({key: starts[key] for key in self.command_keys})
        for group in command_group.unique().tolist():
            in_group = command_group == group
            diagnostics[f"termcheck/motion_{group}/imagined_hard"] = float(hard_termination[in_group].float().mean())
            diagnostics[f"termcheck/motion_{group}/real"] = float(real[in_group].float().mean())
        return diagnostics

    """
    Checkpoint
    """

    def state_dict(self):
        """Everything of the training in one state dict. The policy has the keys of `PPO.state_dict`, so the checkpoint
        is directly playable."""
        return {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "world_model": self.world_model.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "failure_head": self.failure_head.state_dict(),
            "failure_head_optimizer": self.failure_head_optimizer.state_dict(),
            "failure_head_start_iteration": self.failure_head_start_iteration,
            "task_model": self.task_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.actor_critic.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.world_model.load_state_dict(state_dict["world_model"])
        self.world_model_optimizer.load_state_dict(state_dict["world_model_optimizer"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.failure_head.load_state_dict(state_dict["failure_head"])
        self.failure_head_optimizer.load_state_dict(state_dict["failure_head_optimizer"])
        self.failure_head_start_iteration = state_dict["failure_head_start_iteration"]
        self.task_model.load_state_dict(state_dict["task_model"])
        self.context_normalization_ready = True  # the statistics of the context are the ones of the run

    def distributed_data_parallel(self):
        raise NotImplementedError("SuperMimic trains in a single process (no distributed data parallel).")
