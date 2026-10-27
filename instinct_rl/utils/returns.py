"""Generalised advantage estimation as plain functions.

Instinct-RL's ``RolloutStorage.compute_returns`` computes GAE inline for on-policy rollouts. Model-based
algorithms need the same arithmetic over *imagined* rollouts, with a different tensor layout and a
termination probability instead of hard done flags, so the math lives here and both call sites use it.
"""

from typing import Literal

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over a horizon.

    Args:
        rewards: ``(T, B, R)`` or ``(T, B)`` rewards.
        values: ``(T, B, R)`` or ``(T, B)`` value estimates for the same steps.
        dones: ``(T, B, 1)`` or ``(T, B)`` in ``{0, 1}``, or continuous termination probabilities.
            Values are clamped to ``[0, 1]`` so both conventions work.
        last_values: ``(B, R)`` or ``(B,)`` bootstrap value after the horizon.
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        ``(returns, advantages)``, same shape as ``rewards``, not normalised.
    """
    if rewards.shape != values.shape:
        raise ValueError(f"rewards {tuple(rewards.shape)} and values {tuple(values.shape)} must match")
    if rewards.dim() == 3 and dones.dim() == 2:
        dones = dones.unsqueeze(-1)
    if rewards.dim() == 2 and dones.dim() == 3:
        dones = dones.squeeze(-1)
    if rewards.dim() not in (2, 3):
        raise ValueError(f"rewards must be (T, B) or (T, B, R), got {tuple(rewards.shape)}")

    steps = rewards.shape[0]
    done = dones.clamp(0.0, 1.0).to(rewards.dtype)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)
    for step in reversed(range(steps)):
        if step == steps - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]
        not_done = 1.0 - done[step]
        delta = rewards[step] + gamma * next_values * not_done - values[step]
        gae = delta + gamma * lam * not_done * gae
        advantages[step] = gae
    return advantages + values, advantages


def normalize_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Standardise advantages over every non-masked element.

    Args:
        advantages: any shape.
        mask: optional broadcastable mask; zero-sized entries are excluded from the statistics.
        eps: numerical floor for the standard deviation.
    """
    if mask is None:
        return (advantages - advantages.mean()) / (advantages.std() + eps)
    mask_b = mask.to(torch.bool).expand_as(advantages)
    if not bool(mask_b.any()):
        raise ValueError("normalize_advantages got an all-masked tensor")
    selected = advantages[mask_b]
    return (advantages - selected.mean()) / (selected.std() + eps)


def lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Values-only variant: the critic regression target from :func:`compute_gae`."""
    returns, _ = compute_gae(rewards, values, dones, last_values, gamma, lam)
    return returns
