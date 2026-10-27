"""Protocol for tasks that can be rolled forward in a differentiable model.

A model-based algorithm needs to imagine what the simulator would do without calling it. Implementing
this protocol is how a task contributes that ability: the algorithm only ever touches the methods
below, so it stays independent of the robot, the reward terms and the physics backend.

Implementations live in the task package (InstinctLab side) and are handed to the algorithm through
the algorithm config's ``inputs``, which the runner resolves.

Tasks whose rollouts depend on per-transition data beyond state and context (for example which
reference motion a state was tracking) declare ``SEED_GROUPS``: observation groups the algorithm
stores with every transition. They also implement ``begin_rollout(seeds, env_index, horizon)``,
which receives those groups for the sampled start states and returns the initial ``ref_index``, and
``subset_reference_rows(idx)``, which narrows the rollout's reference to a batch subset. Everything is plain torch and differentiable where the algorithm needs
gradients (``reward``, ``termination_prob``, ``advance``, ``policy_obs``); ``critic_obs`` is typically
not differentiated through.

Layout conventions match the rest of Instinct-RL: component names and shapes are ``OrderedDict[str,
tuple]``, the same format as ``obs_format``, so a task can hand the algorithm the very layout it uses
for observations.
"""

from typing import OrderedDict, Protocol, runtime_checkable

import torch


@runtime_checkable
class ImaginationInterface(Protocol):
    """Differentiable view of a task's MDP."""

    #: Layout of one world-model state frame: ``{component: shape}``.
    state_format: "OrderedDict[str, tuple]"
    #: Layout of the per-environment context vector: ``{component: shape}`` (may be empty).
    context_format: "OrderedDict[str, tuple]"
    #: How many frames of history ``policy_obs`` needs (0 or 1 = current frame only).
    history_length: int
    #: Terms of the policy observation that carry ``history_length`` frames.
    history_terms: tuple[str, ...]

    def policy_obs(
        self,
        state: torch.Tensor,
        context: torch.Tensor | None,
        prev_action: torch.Tensor | None,
        ref_index: torch.Tensor | None,
        history_state: torch.Tensor | None = None,
        history_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble the policy observation from an imagined state.

        Returns an *unnormalised* observation with the same layout as the env's ``policy`` group; the
        algorithm applies the runner's normaliser.

        Args:
            state: ``(B, state_size)`` current imagined state.
            context: ``(B, context_size)`` or None.
            prev_action: ``(B, num_actions)`` action applied to reach ``state``.
            ref_index: Column offset into the task's reference window, or None.
            history_state: ``(B, history_length, state_size)`` trailing state frames, oldest first,
                current last. Required when :attr:`history_terms` is non-empty.
            history_action: ``(B, history_length, num_actions)`` trailing actions on the same frames,
                where entry ``j`` is the action that produced ``history_state[:, j]``.
        """

    def add_obs_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply the observation noise the task uses during training."""

    def critic_obs(
        self,
        state: torch.Tensor,
        context: torch.Tensor | None,
        ref_index: torch.Tensor | None,
        prev_action: torch.Tensor | None,
    ) -> torch.Tensor:
        """Assemble the critic observation (privileged information is allowed here).

        ``prev_action`` is the action that produced ``state``, for critics that observe it.
        """

    def reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        ref_index: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-step reward, shape ``(B, num_rewards)``, differentiable w.r.t. state and action."""

    def termination_prob(self, state: torch.Tensor, ref_index: torch.Tensor | None) -> torch.Tensor:
        """Termination probability, shape ``(B, 1)``, in ``[0, 1]`` and differentiable.

        A soft version of the simulator's termination is required: hard 0/1 flags would cut the
        gradient and bias the imagined return.
        """

    def advance(self, ref_index: torch.Tensor | None) -> torch.Tensor | None:
        """Return the reference index one control step later (``None`` for tasks without references)."""

    def valid_start(self, ref_index: torch.Tensor | None, horizon: int) -> torch.Tensor:
        """Boolean ``(B,)``: can a rollout of ``horizon`` steps start here without leaving the clip?"""
