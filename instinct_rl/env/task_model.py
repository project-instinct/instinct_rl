from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class TaskModel(ABC):
    """Abstract class for the differentiable model of a task, used by the world-model algorithms (see `SuperMimic`).

    The algorithm trains the policy on trajectories imagined by a learned world model of the robot. To evaluate the
    policy on an imagined state, it needs the observations, rewards and terminations of the task as torch functions of
    that state. This class is the contract of these functions. It does NOT read the simulator: the simulator data
    reaches the algorithm as observation groups of the `VecEnv` (see `SuperMimic` for the group and its components).

    The algorithm builds the task model from its config as
    `task_model_class(env=env, obs_format=obs_format, device=device, **task_model_kwargs)`, where `env` is the `VecEnv`
    of the runner. The constants of the task (reward weights, termination thresholds, actuator gains, the reference
    motions, the terrain) are not observations: the constructor reads them once from the env.

    Conventions:

    - All tensors are on `device`, batched in the first dimensions.
    - `S` is the floating-base state tuple `(p, R, v, w, q, qd)`: root position (..., 3), root rotation matrix
      (..., 3, 3), world linear velocity of the root link-frame origin (..., 3), world angular velocity (..., 3), joint
      positions (..., num_joints) and joint velocities (..., num_joints). All positions are world positions.
    - `cmd` is the dict {name: tensor} of the reference command the robot is tracking. The keys are the
      `command_components` of the algorithm; a component of shape (1,) is given as a scalar. It is the time-varying task
      input of the policy, e.g. the index of the reference frame, its fractional phase and the origin of the reference
      in the world frame. It is a float tensor, as it is in the observation.
    - The context `xi` is the constant vector per episode that describes the physical parameters of an env that the
      world model cannot observe (e.g. domain randomization of masses, gains and friction).
    - The functions are called inside `torch.compile`d functions (some with dynamic shapes), so they must be traceable:
      no host syncs, no data-dependent shapes.
    """

    num_joints: int
    """Number of actuated joints. The actions of the policy have the same size."""

    num_contacts: int
    """Number of bodies whose contact indicator is predicted by the world model."""

    context_dim: int
    """Size of the context xi."""

    scan_dim: int
    """Size of the exteroceptive terrain height scan (`scan`)."""

    critic_dim: int
    """Size of the privileged critic features (`critic_features`)."""

    dt: float
    """Duration of one env step [s]. The rewards of `reward` already contain this factor."""

    proprio_noise: torch.Tensor
    """Half widths of the uniform noise added to the proprioceptive observation (`proprio_observation`). Shape
    (proprio_dim,)."""

    device: torch.device
    """Device to use."""

    """
    Context and checkpointing.
    """

    @abstractmethod
    def fit_context_normalization(self, context: torch.Tensor):
        """Fix the statistics of `normalize_context` from the contexts of all the envs (called once, with the first
        observation of a new run).

        Args:
            context: Raw context of every env. Shape (num_envs, context_dim)
        """
        raise NotImplementedError

    @abstractmethod
    def normalize_context(self, xi: torch.Tensor) -> torch.Tensor:
        """Normalize the context with the fixed statistics.

        Args:
            xi: Raw context. Shape (..., context_dim)

        Returns:
            The normalized context with the same shape.
        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        """Return the state of the task model which has to be checkpointed (the statistics of `normalize_context`)."""
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        """Restore what `state_dict` returned."""
        raise NotImplementedError

    """
    Differentiable task model.
    """

    @abstractmethod
    def scan(self, S: tuple) -> torch.Tensor:
        """Return the noise-free terrain height scan at the root pose of `S`. Shape (..., scan_dim)."""
        raise NotImplementedError

    @abstractmethod
    def proprio_observation(self, S: tuple, a_prev: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Return the noise-free proprioceptive observation of the policy, one time step (not the history). Shape
        (..., proprio_dim), the same size as `proprio_noise`.

        Args:
            a_prev: The last action, the one that led to `S`. Shape (..., num_joints)
            xi: Raw context. Shape (..., context_dim)
        """
        raise NotImplementedError

    @abstractmethod
    def policy_observation(
        self,
        o_hist: torch.Tensor,
        S: tuple,
        cmd: dict[str, torch.Tensor],
        noisy_goal: bool = True,
    ) -> torch.Tensor:
        """Assemble the flat observation of the policy in EXACTLY the format of the env (the order and the flattening of
        the terms of the `policy` observation group).

        Args:
            o_hist: History of `proprio_observation` (with noise), oldest first. Shape (B, hist, proprio_dim)
            cmd: Command at `S`.
            noisy_goal: Add the observation noise of the goal terms (the observation of the command).
        """
        raise NotImplementedError

    @abstractmethod
    def reward(
        self,
        S: tuple,
        cmd: dict[str, torch.Tensor],
        a: torch.Tensor,
        a_prev: torch.Tensor,
        xi: torch.Tensor,
        contact_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the per-step reward (weights and `dt` applied) of the task for the transition ending in `S`.

        Args:
            a: The action applied in the transition. Shape (..., num_joints)
            a_prev: The action before it. Shape (..., num_joints)
            contact_prob: Probability (or indicator) of the contacts at `S`. Shape (..., num_contacts)

        Returns:
            The reward (...) and the dict {name: (...)} of its weighted terms.
        """
        raise NotImplementedError

    @abstractmethod
    def soft_termination(self, S: tuple, cmd: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the differentiable probability (...) that the termination conditions of the task end the episode at
        `S`."""
        raise NotImplementedError

    @abstractmethod
    def hard_termination(self, S: tuple, cmd: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return whether (bool, ...) the termination conditions of the task end the episode at `S`."""
        raise NotImplementedError

    @abstractmethod
    def critic_features(self, S: tuple, cmd: dict[str, torch.Tensor], xi_n: torch.Tensor) -> torch.Tensor:
        """Return the privileged features (..., critic_dim) of the critic and the failure head: the state, its errors to
        the reference, the normalized context, the goal and the terrain scan (all without observation noise).

        Args:
            xi_n: Normalized context (see `normalize_context`). Shape (..., context_dim)
        """
        raise NotImplementedError

    @abstractmethod
    def advance_command(self, cmd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the command one env step later."""
        raise NotImplementedError

    @abstractmethod
    def imagination_valid(self, cmd: dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
        """Return whether (bool, ...) the command can be followed for `horizon` more steps, i.e. an imagination of this
        length from a state with `cmd` stays inside the reference."""
        raise NotImplementedError

    @abstractmethod
    def command_group(self, cmd: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the index (long, ...) of the group of commands (e.g. the reference motion) that `cmd` belongs to. Used
        for diagnostics only."""
        raise NotImplementedError

    @abstractmethod
    def max_command_length(self) -> int:
        """Return the number of steps of the longest reference. The algorithm checks that it is longer than the
        imagination horizon."""
        raise NotImplementedError


class VisualTaskModel(TaskModel):
    """Abstract class for the task model of a task whose policy also observes an image (see `VisualSuperMimic`).

    The policy observation contains one image term (a processed image such as a depth map), which the world model does
    not render: the latent of the image encoder of the policy is a part of its state. The image only enters
    `policy_observation`.
    """

    @abstractmethod
    def policy_observation(
        self,
        o_hist: torch.Tensor,
        S: tuple,
        cmd: dict[str, torch.Tensor],
        image: torch.Tensor,
        noisy_goal: bool = True,
    ) -> torch.Tensor:
        """Assemble the flat observation of the policy in EXACTLY the format of the env, including the image term.

        Args:
            o_hist: History of `proprio_observation` (with noise), oldest first. Shape (B, hist, proprio_dim)
            cmd: Command at `S`.
            image: Processed image. Shape (B, *image_shape). The algorithm cuts the image term out of the observation
                and feeds the visual latent to the actor instead, so the content of `image` does not matter there.
            noisy_goal: Add the observation noise of the goal terms (the observation of the command).
        """
        raise NotImplementedError
