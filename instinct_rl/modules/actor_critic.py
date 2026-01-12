# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

from instinct_rl.utils.utils import get_subobs_size


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        num_rewards=1,
        mu_activation=None,  # If set, the last layer will be added with a special activation layer.
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # obs_segmnets is a ordered dict that contains the observation components (string) and its shape.
        # We use this to slice the observations into the correct components.
        self.__obs_format = obs_format
        self.__obs_segments = obs_format["policy"]
        self.__critic_obs_segments = obs_format.get("critic", obs_format["policy"])

        self.activation = activation
        self.mu_activation = mu_activation
        self.mlp_input_dim_a = get_subobs_size(self.__obs_segments)
        self.actor_hidden_dims = actor_hidden_dims
        self.mlp_input_dim_c = get_subobs_size(self.__critic_obs_segments)
        self.critic_hidden_dims = critic_hidden_dims

        # Policy
        self.actor = self._build_actor(num_actions)

        # Value function
        critic = self._build_critic(1)
        if num_rewards > 1:
            self.critics = nn.ModuleList([critic] + [self._build_critic(1) for _ in range(num_rewards - 1)])
        else:
            self.critic = critic

        print(f"Actor MLP: {self.actor}")
        if num_rewards > 1:
            print(f"Multiple Critics MLP: {len(self.critics)} in total.")
        else:
            print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def _build_actor(self, num_actions):
        activation = get_activation(self.activation)
        actor_layers = []
        actor_layers.append(nn.Linear(self.mlp_input_dim_a, self.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(self.actor_hidden_dims)):
            if l == len(self.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(self.actor_hidden_dims[l], num_actions))
                if self.mu_activation:
                    actor_layers.append(get_activation(self.mu_activation))
            else:
                actor_layers.append(nn.Linear(self.actor_hidden_dims[l], self.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        return nn.Sequential(*actor_layers)

    def _build_critic(self, num_values=1):
        """Build the critic network, enabling multi-value prediction. Predict 1 value by default."""
        activation = get_activation(self.activation)
        critic_layers = []
        critic_layers.append(nn.Linear(self.mlp_input_dim_c, self.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(self.critic_hidden_dims)):
            if l == len(self.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(self.critic_hidden_dims[l], num_values))
            else:
                critic_layers.append(nn.Linear(self.critic_hidden_dims[l], self.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        return nn.Sequential(*critic_layers)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        if hasattr(self, "critics") and isinstance(critic_observations, list):
            value = torch.cat(
                [critic(critic_obs) for critic, critic_obs in zip(self.critics, critic_observations)],
                dim=-1,
            )
        elif hasattr(self, "critics"):
            value = torch.cat(
                [critic(critic_observations) for critic in self.critics],
                dim=-1,
            )
        else:
            value = self.critic(critic_observations)
        return value

    @torch.no_grad()
    def clip_std(self, min=None, max=None):
        self.std.copy_(self.std.clip(min=min, max=max))

    @property
    def obs_segments(self):
        """NOTE: make sure to copy these methods in subclasses if the subclass has its own obs_segments"""
        return self.__obs_segments

    @property
    def critic_obs_segments(self):
        return self.__critic_obs_segments

    def export_as_onnx(self, observations, filedir):
        """Export the model as an ONNX file. Input should be batch-wise observations with batchsize 1."""
        self.eval()
        with torch.no_grad():
            exported_program = torch.onnx.export(
                self.actor,
                observations,
                os.path.join(filedir, "actor.onnx"),
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
            )
            print(f"Exported ActorCritic model to {os.path.join(filedir, 'actor.onnx')}")


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
