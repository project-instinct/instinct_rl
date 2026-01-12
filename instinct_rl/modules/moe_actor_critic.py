import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic import ActorCritic, get_activation
from .moe import MoeLayer


class MoEActorCritic(ActorCritic):
    def __init__(
        self,
        obs_format,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        num_rewards=1,
        mu_activation=None,
        num_moe_experts=8,
        moe_gate_hidden_dims=[],
        **kwargs,
    ):
        self.num_moe_experts = num_moe_experts
        self.moe_gate_hidden_dims = moe_gate_hidden_dims
        super().__init__(
            obs_format,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            num_rewards,
            mu_activation,
            **kwargs,
        )

    def _build_actor(self, num_actions):
        moe = MoeLayer(
            self.mlp_input_dim_a,
            self.num_moe_experts,
            output_dim=num_actions,
            activation=self.activation,
            expert_hidden_dims=self.actor_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )
        if self.mu_activation:
            return nn.Sequential(moe, get_activation(self.mu_activation))
        return moe

    def _build_critic(self, num_values=1):
        return MoeLayer(
            self.mlp_input_dim_c,
            self.num_moe_experts,
            output_dim=num_values,
            activation=self.activation,
            expert_hidden_dims=self.critic_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )
