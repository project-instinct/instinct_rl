import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from instinct_rl.algorithms.ppo import PPO
from instinct_rl.algorithms.tppo import TPPO
from instinct_rl.utils.utils import get_subobs_by_components, unpad_trajectories


class EstimatorAlgoMixin:
    """A supervised algorithm implementation that trains a state predictor in the policy model"""

    def __init__(
        self,
        *args,
        target_from_critic=True,  # if True, get target subobs from critic_obs
        estimator_loss_func="mse_loss",
        estimator_loss_kwargs=dict(),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_from_critic = target_from_critic
        self.estimator_obs_components = self.actor_critic.estimator_obs_components
        self.estimator_target_obs_components = self.actor_critic.estimator_target_components
        self.estimator_loss_func = estimator_loss_func
        self.estimator_loss_kwargs = estimator_loss_kwargs

    def compute_losses(self, minibatch):
        losses, inter_vars, stats = super().compute_losses(minibatch)

        # Use the critic_obs from the same timestep for estimation target
        # Not considering predicting the next state for now.
        estimation_target = get_subobs_by_components(
            minibatch.critic_obs if self.target_from_critic else minibatch.actor_obs,
            component_names=self.estimator_target_obs_components,
            obs_segments=self.actor_critic.critic_obs_segments,
        )
        estimation = self.actor_critic.get_estimated_state()
        if self.actor_critic.is_recurrent:
            estimation_target = unpad_trajectories(estimation_target, minibatch.masks)
            # actor_critic must compute the estimated_state_ during act() as a intermediate variable
            estimation = unpad_trajectories(estimation, minibatch.masks)

        estimator_loss = getattr(F, self.estimator_loss_func)(
            estimation,
            estimation_target,
            **self.estimator_loss_kwargs,
            reduction="none",
        ).sum(dim=-1)

        losses["estimator_loss"] = estimator_loss.mean()

        return losses, inter_vars, stats


class EstimatorPPO(EstimatorAlgoMixin, PPO):
    pass


class EstimatorTPPO(EstimatorAlgoMixin, TPPO):
    pass
