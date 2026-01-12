from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic import get_activation  # Assuming get_activation is needed


class MoeLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_experts,
        output_dim=None,
        activation="elu",
        expert_hidden_dims=[],
        gate_hidden_dims=[],
    ):
        super().__init__()
        self.act_fn = get_activation(activation)
        self.gate = self._build_gate(input_dim, num_experts, gate_hidden_dims)
        self.experts = nn.ModuleList(
            [self._build_expert(input_dim, output_dim, expert_hidden_dims) for _ in range(num_experts)]
        )

    def _build_gate(self, input_dim, num_experts, hidden_dims):
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(self.act_fn)
            curr_dim = h
        layers.append(nn.Linear(curr_dim, num_experts))
        return nn.Sequential(*layers)

    def _build_expert(self, input_dim, output_dim, hidden_dims):
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(self.act_fn)
            curr_dim = h
        if output_dim is not None:
            layers.append(nn.Linear(curr_dim, output_dim))  # no activation for the last layer
        return nn.Sequential(*layers)

    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [batch, num_experts] # gate the expert outputs
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, output_dim]
        output = torch.einsum("be,beo->bo", gate_scores, expert_outputs)  # mix the expert outputs
        return output
