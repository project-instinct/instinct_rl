import torch
import torch.nn as nn
import torch.optim as optim

from instinct_rl.algorithms.ppo import PPO
from instinct_rl.utils.buffer import buffer_func
from instinct_rl.utils.collections import namedarraytuple
from instinct_rl.utils.utils import get_subobs_by_components


class LipschitzyMixin:
    """A plugin loss term for penalizing network high gradient"""

    def __init__(
        self,
        *args,
        gradient_penalty_coef=10.0,
        gradient_penalty_coef_schedule=(
            2000,
            5000,
        ),  # the linear schedule for increasing the gradient penalty coefficient
        critic_gradient_penalty_coef=0.0,  # the coefficient for the gradient penalty for critic, 0.0 means no penalty
        gradient_penalty_coef_schedule_critic=(
            2000,
            5000,
        ),  # the linear schedule for increasing the gradient penalty coefficient for critic
        backbone_gradient_only=False,  # For EncoderActorCritic to speed up the gradient penalty computation, compute only the gradient w.r.t backbone input
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._gradient_penalty_coef = gradient_penalty_coef  # as a buffer variable
        self.gradient_penalty_coef_schedule = gradient_penalty_coef_schedule
        self._critic_gradient_penalty_coef = critic_gradient_penalty_coef  # as a buffer variable
        self.critic_gradient_penalty_coef_schedule = gradient_penalty_coef_schedule_critic
        self.gradient_penalty_coef = (
            gradient_penalty_coef[0] if isinstance(gradient_penalty_coef, (list, tuple)) else gradient_penalty_coef
        )
        self.critic_gradient_penalty_coef = (
            critic_gradient_penalty_coef[0]
            if isinstance(critic_gradient_penalty_coef, (list, tuple))
            else critic_gradient_penalty_coef
        )
        self.backbone_gradient_only = backbone_gradient_only

    def compute_losses(self, minibatch):
        losses, inter_vars, stats = super().compute_losses(minibatch)

        if self.gradient_penalty_coef > 0:
            if self.backbone_gradient_only:
                gradient_penalty = self.compute_actor_backbone_gradient(minibatch)
            else:
                gradient_penalty = self.compute_actor_gradient(minibatch)
            losses["gradient_penalty"] = gradient_penalty
            stats["gradient_penalty"] = gradient_penalty

            # update gradient penalty coefficient
            if isinstance(self._gradient_penalty_coef, (list, tuple)):
                gradient_stage = min(
                    max((self.current_learning_iteration - self.gradient_penalty_coef_schedule[0]), 0)
                    / self.gradient_penalty_coef_schedule[1],
                    1,
                )
                self.gradient_penalty_coef = (
                    gradient_stage * (self._gradient_penalty_coef[1] - self._gradient_penalty_coef[0])
                    + self._gradient_penalty_coef[0]
                )

        if self.critic_gradient_penalty_coef > 0:
            if self.backbone_gradient_only:
                critic_gradient_penalty = self.compute_critic_backbone_gradient(minibatch)
            else:
                critic_gradient_penalty = self.compute_critic_gradient(minibatch)
            losses["critic_gradient_penalty"] = critic_gradient_penalty
            stats["critic_gradient_penalty"] = critic_gradient_penalty

            # update critic gradient penalty coefficient
            if isinstance(self._critic_gradient_penalty_coef, (list, tuple)):
                gradient_stage = min(
                    max((self.current_learning_iteration - self.critic_gradient_penalty_coef_schedule[0]), 0)
                    / self.critic_gradient_penalty_coef_schedule[1],
                    1,
                )
                self.critic_gradient_penalty_coef = (
                    gradient_stage * (self._critic_gradient_penalty_coef[1] - self._critic_gradient_penalty_coef[0])
                    + self._critic_gradient_penalty_coef[0]
                )

        return losses, inter_vars, stats

    def compute_actor_backbone_gradient(self, minibatch):
        """To speed up the gradient penalty computation, compute only the gradient w.r.t backbone input
        where backbone is typically an MLP, not implemented for RNN yet.
        """
        if not hasattr(self.actor_critic, "encoders"):
            return self.compute_actor_gradient(minibatch)

        if self.actor_critic.is_recurrent:
            raise NotImplementedError("Not implemented for RNN yet")

        # Compute the gradient w.r.t backbone input
        latent = self.actor_critic.encoders(minibatch.obs).detach()
        latent.requires_grad = True

        self.actor_critic.backbone_act(latent)
        actions_log_prob = self.actor_critic.get_actions_log_prob(minibatch.actions)
        grad_outputs = torch.ones_like(actions_log_prob)

        grad = torch.autograd.grad(
            outputs=actions_log_prob,
            inputs=latent,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        return grad.norm(2, dim=-1).pow(2).mean()

    def compute_actor_gradient(self, minibatch):
        """Compute the gradient w.r.t policy input. It is used as a penalty to satisfy the condition of Lipschitz continuity"""
        buffer_func(minibatch.obs, setattr, "requires_grad", True)
        actor_hidden_states = minibatch.hidden_states.actor if self.actor_critic.is_recurrent else None
        self.actor_critic.act(minibatch.obs, masks=minibatch.masks, hidden_states=actor_hidden_states)
        actions_log_prob = self.actor_critic.get_actions_log_prob(minibatch.actions)
        grad_outputs = torch.ones_like(actions_log_prob)

        grad = torch.autograd.grad(
            outputs=actions_log_prob,
            inputs=minibatch.obs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Set the buffer variable back to False
        buffer_func(minibatch.obs, setattr, "requires_grad", False)

        return grad.norm(2, dim=-1).pow(2).mean()

    def compute_critic_backbone_gradient(self, minibatch):
        """To speed up the gradient penalty computation, compute only the gradient w.r.t backbone input
        where backbone is typically an MLP, not implemented for RNN yet.
        """
        if not hasattr(self.actor_critic, "critic_encoders"):
            return self.compute_critic_gradient(minibatch)

        if self.actor_critic.is_recurrent:
            raise NotImplementedError("Not implemented for RNN yet")

        # Compute the gradient w.r.t backbone input
        latent = self.actor_critic.critic_encoders(minibatch.critic_obs).detach()
        latent.requires_grad = True

        values = self.actor_critic.backbone_evaluate(latent, masks=minibatch.masks)
        grad_outputs = torch.ones_like(values)

        grad = torch.autograd.grad(
            outputs=values,
            inputs=latent,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        return grad.norm(2, dim=-1).pow(2).mean()

    def compute_critic_gradient(self, minibatch):
        """Compute the gradient w.r.t value function input. It is used as a penalty to satisfy the condition of Lipschitz continuity"""
        buffer_func(minibatch.critic_obs, setattr, "requires_grad", True)
        critic_hidden_states = minibatch.hidden_states.critic if self.actor_critic.is_recurrent else None
        values = self.actor_critic.evaluate(
            minibatch.critic_obs, masks=minibatch.masks, hidden_states=critic_hidden_states
        )
        grad_outputs = torch.ones_like(values)

        grad = torch.autograd.grad(
            outputs=values,
            inputs=minibatch.critic_obs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Set the buffer variable back to False
        buffer_func(minibatch.critic_obs, setattr, "requires_grad", False)

        return grad.norm(2, dim=-1).pow(2).mean()


class LipschitzPPO(LipschitzyMixin, PPO):
    pass
