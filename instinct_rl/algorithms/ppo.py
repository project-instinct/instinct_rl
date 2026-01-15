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
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from instinct_rl.modules import ActorCritic
from instinct_rl.storage import RolloutStorage
from instinct_rl.utils.utils import get_subobs_size


class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        advantage_mixing_weights=1.0,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        clip_min_std=1e-15,  # clip the policy.std if it supports, check update()
        optimizer_class_name="Adam",
        schedule="fixed",
        desired_kl=0.01,
        auxiliary_reward_per_env_reward_coefs: list[float] = list(),
        device="cpu",
        **kwargs,
    ):
        """

        Args:
            auxiliary_reward_per_env_reward_coefs: list of float, the coefficients for each of the auxiliary reward.
                The length of the list should be the same as the number of rewards from the environment (in case of multi-critic setting).
        """
        if kwargs:
            print("\033[43;33mWarning: PPO init got extra kwargs:", kwargs.keys(), "\033[0m")
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.auxiliary_reward_per_env_reward_coefs = (
            torch.tensor(auxiliary_reward_per_env_reward_coefs, device=self.device).unsqueeze(0)  # (1, num_rewards)
            if len(auxiliary_reward_per_env_reward_coefs) > 0
            else 1.0
        )
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = getattr(optim, optimizer_class_name)(self.actor_critic.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.advantage_mixing_weights: float | torch.Tensor = (
            torch.tensor(advantage_mixing_weights, device=self.device)
            if isinstance(advantage_mixing_weights, (tuple, list))
            else advantage_mixing_weights
        )
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_min_std = (
            torch.tensor(clip_min_std, device=self.device) if isinstance(clip_min_std, (tuple, list)) else clip_min_std
        )

        # algorithm status
        self.current_learning_iteration = 0

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        """
        Args:
            num_rewards: int, in case of multi-reward setting, with multiple critic networks.
        """
        self.transition = RolloutStorage.Transition()
        obs_size = get_subobs_size(obs_format["policy"])
        critic_obs_size = get_subobs_size(obs_format.get("critic")) if "critic" in obs_format else None
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            num_rewards=num_rewards,
            device=self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs if critic_obs is not None else obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        self.transition.rewards = rewards.clone()

        auxiliary_rewards = self.compute_auxiliary_reward(infos["observations"])
        # Add auxiliary rewards to the transition
        for k, v in auxiliary_rewards.items():
            coef = getattr(
                self, f"{k}_coef", 1.0
            )  # This coef is per-auxiliary wise, will apply to all the rewards (if multiple rewards from the environment)
            if coef != 0.0:
                # (num_envs, num_rewards) = scalar * (num_envs, num_rewards) * (1, num_rewards)
                self.transition.rewards += coef * v * self.auxiliary_reward_per_env_reward_coefs
            # Add the auxiliary rewards to info
            infos["step"][k] = v

        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += (
                self.gamma * self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device)
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the auxiliary reward depending on the algorithms. Default PPO does not have auxiliary reward."""
        return dict()

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        mean_losses = defaultdict(float)
        average_stats = defaultdict(float)
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for minibatch in generator:
            losses, _, stats = self.compute_losses(minibatch)

            loss = 0.0
            for k, v in losses.items():
                loss += getattr(self, k + "_coef", 1.0) * v
                mean_losses[k] = mean_losses[k] + v.detach()
            mean_losses["total_loss"] = mean_losses["total_loss"] + loss.detach()
            for k, v in stats.items():
                average_stats[k] = average_stats[k] + v.detach()

            # Gradient step
            self.gradient_step(loss, average_stats)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in mean_losses.keys():
            mean_losses[k] = mean_losses[k] / num_updates
        for k in average_stats.keys():
            average_stats[k] = average_stats[k] / num_updates
        self.storage.clear()
        if hasattr(self.actor_critic, "clip_std"):
            self.actor_critic.clip_std(min=self.clip_min_std)

        return mean_losses, average_stats

    def compute_losses(self, minibatch):
        actor_hidden_states = minibatch.hidden_states.actor if self.actor_critic.is_recurrent else None
        self.actor_critic.act(minibatch.obs, masks=minibatch.masks, hidden_states=actor_hidden_states)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(minibatch.actions)
        critic_hidden_states = minibatch.hidden_states.critic if self.actor_critic.is_recurrent else None
        value_batch = self.actor_critic.evaluate(
            minibatch.critic_obs, masks=minibatch.masks, hidden_states=critic_hidden_states
        )
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        try:
            entropy_batch = self.actor_critic.entropy
        except:
            entropy_batch = None

        # KL
        if self.desired_kl != None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / minibatch.old_sigma + 1.0e-5)
                    + (torch.square(minibatch.old_sigma) + torch.square(minibatch.old_mu - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                if dist.is_initialized():
                    dist.all_reduce(kl_mean, op=dist.ReduceOp.SUM)
                    kl_mean /= dist.get_world_size()

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                if dist.is_initialized():
                    # broadcast the learning rate to all processes
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    dist.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(minibatch.old_actions_log_prob))
        mixed_advantages = torch.mean(minibatch.advantages * self.advantage_mixing_weights, dim=-1)
        surrogate = -mixed_advantages * ratio
        surrogate_clipped = -mixed_advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = minibatch.values + (value_batch - minibatch.values).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - minibatch.returns).pow(2)
            value_losses_clipped = (value_clipped - minibatch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = (minibatch.returns - value_batch).pow(2)
        # preserve the last dimension for multi-reward setting
        value_loss = value_loss.reshape(-1, value_loss.shape[-1]).mean(dim=0)

        # pack the losses and stats
        stats_ = dict()
        if value_loss.numel() > 1:
            for i in range(minibatch.advantages.shape[-1]):
                stats_[f"advantage_{i}"] = minibatch.advantages[..., i].detach().mean()
            for i in range(value_loss.numel()):
                stats_[f"value_loss_{i}"] = value_loss.detach().cpu()[i]

        return_ = dict(
            surrogate_loss=surrogate_loss,
            value_loss=value_loss.mean(),
        )
        if entropy_batch is not None:
            return_["entropy"] = -entropy_batch.mean()

        inter_vars = dict(
            ratio=ratio,
            surrogate=surrogate,
            surrogate_clipped=surrogate_clipped,
        )
        if self.desired_kl != None and self.schedule == "adaptive":
            inter_vars["kl"] = kl
        if self.use_clipped_value_loss:
            inter_vars["value_clipped"] = value_clipped

        return return_, inter_vars, stats_

    def state_dict(self):
        state_dict = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if hasattr(self, "lr_scheduler"):
            state_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.actor_critic.load_state_dict(state_dict["model_state_dict"])
        if "optimizer_state_dict" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler_state_dict"])
        elif "lr_scheduler_state_dict" in state_dict:
            print("Warning: lr scheduler state dict loaded but no lr scheduler is initialized. Ignored.")

    def distributed_data_parallel(self):
        """Enable distributed data parallel on the networks. Making sure the parameters are the same
        across all processes.
        """
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                print("[INFO] Broadcasting parameters from rank 0 to all processes.")
            for param in self.actor_critic.parameters():
                dist.broadcast(param.data, src=0)

    def gradient_step(self, loss: torch.Tensor, average_stats: dict):
        """Perform a gradient step towards the loss and update the parameters of the actor critic.
        Distributed data parallel is supported.
        NOTE: if the parameters are not on self.actor_critic, this function will not work. Please update other
        networks in another optimizer.
        """
        self.optimizer.zero_grad()
        loss.backward()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for param in self.actor_critic.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
        grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        average_stats["grad_norm"] = average_stats["grad_norm"] + grad_norm.detach()
        self.optimizer.step()
