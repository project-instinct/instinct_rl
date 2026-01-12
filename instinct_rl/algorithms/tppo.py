""" PPO with teacher network """

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

import instinct_rl.modules as modules
from instinct_rl.algorithms.ppo import PPO
from instinct_rl.storage.rollout_storage import ActionLabelRollout
from instinct_rl.utils import unpad_trajectories
from instinct_rl.utils.utils import get_subobs_size


# assuming learning iteration is at an assumable iteration scale
def GET_PROB_FUNC(option, iteration_scale):
    PROB_options = {
        "linear": lambda x: max(0, 1 - 1 / iteration_scale * x),
        "exp": lambda x: max(0, (1 - 1 / iteration_scale) ** x),
        "tanh": lambda x: max(0, 0.5 * (1 - np.tanh(1 / iteration_scale * (x - iteration_scale)))),
    }
    return PROB_options[option]


class TPPO(PPO):
    def __init__(
        self,
        *args,
        teacher_logdir=None,
        teacher_policy_class_name="ActorCritic",
        teacher_policy=dict(),
        label_action_with_critic_obs=True,  # else, use actor obs
        teacher_act_prob="exp",  # a number or a callable to (0 ~ 1) to the selection of act using teacher policy
        update_times_scale=100,  # a rough estimation of how many times the update will be called
        using_ppo=True,  # If False, compute_losses will skip ppo loss computation and returns to DAGGR
        distillation_loss_coef=1.0,  # can also be string to select a prob function to scale the distillation loss
        distill_target="real",
        buffer_dilation_ratio=1.0,
        lr_scheduler_class_name=None,
        lr_scheduler=dict(),
        hidden_state_resample_prob=0.0,  # if > 0, Some hidden state in the minibatch will be resampled
        action_labels_from_sample=False,  # if True, the action labels from teacher policy will be from policy.act instead of policy.act_inference
        **kwargs,
    ):
        """
        Args:
        - teacher_logdir: the log directory of the teacher policy. Must contain a model_...pt file and the
            params/agent.yaml file.
        """
        super().__init__(*args, **kwargs)
        self.teacher_logdir = teacher_logdir
        self.teacher_policy_cfg_dict = teacher_policy
        self.label_action_with_critic_obs = label_action_with_critic_obs
        self.teacher_act_prob = teacher_act_prob
        self.update_times_scale = update_times_scale
        if isinstance(self.teacher_act_prob, str):
            self.teacher_act_prob = GET_PROB_FUNC(self.teacher_act_prob, update_times_scale)
        else:
            self.__teacher_act_prob = self.teacher_act_prob
            self.teacher_act_prob = lambda x: self.__teacher_act_prob
        self.using_ppo = using_ppo
        self.__distillation_loss_coef = distillation_loss_coef
        if isinstance(self.__distillation_loss_coef, str):
            self.distillation_loss_coef_func = GET_PROB_FUNC(self.__distillation_loss_coef, update_times_scale)
        self.distill_target = distill_target
        self.buffer_dilation_ratio = buffer_dilation_ratio
        self.lr_scheduler_class_name = lr_scheduler_class_name
        self.lr_scheduler_kwargs = lr_scheduler
        self.hidden_state_resample_prob = hidden_state_resample_prob
        self.action_labels_from_sample = action_labels_from_sample
        # self.transition = ActionLabelRollout.Transition() # suppose to be built at init_storage()

        # build and load teacher network
        self.teacher_actor_critic = modules.build_actor_critic(
            teacher_policy_class_name,
            self.teacher_policy_cfg_dict,
            self.teacher_policy_cfg_dict["obs_format"],
            self.teacher_policy_cfg_dict["num_actions"],
            self.teacher_policy_cfg_dict["num_rewards"],
        )
        if not self.teacher_logdir is None:
            self.load_teacher_policy()
        else:
            print(
                "TPPO Warning: No snapshot loaded for teacher policy. Make sure you have a pretrained teacher network"
            )

        # initialize lr scheduler if needed
        if not self.lr_scheduler_class_name is None:
            self.lr_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler_class_name)(
                self.optimizer,
                **self.lr_scheduler_kwargs,
            )

    def load_teacher_policy(self):
        # acquire the latest model file in the teacher logdir
        model_files = [
            file for file in os.listdir(self.teacher_logdir) if file.endswith(".pt") and file.startswith("model_")
        ]
        model_files.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
        model_file = model_files[-1]
        state_dict = torch.load(osp.join(self.teacher_logdir, model_file), map_location="cpu")
        self.teacher_actor_critic.load_state_dict(state_dict["model_state_dict"])

        self.teacher_actor_critic.to(self.device)
        self.teacher_actor_critic.eval()

        # if normalizer is provided, build the normalizer for current teacher policy
        if "policy_normalizer_state_dict" in state_dict:
            with open(osp.join(self.teacher_logdir, "params", "agent.yaml")) as f:
                env_cfg_dict = yaml.full_load(f)
            self.teacher_policy_normalizer = modules.build_normalizer(
                input_shape=get_subobs_size(self.teacher_policy_cfg_dict["obs_format"]["policy"]),
                normalizer_class_name=env_cfg_dict["normalizers"]["policy"].pop("class_name"),
                normalizer_kwargs=env_cfg_dict["normalizers"]["policy"],
            )
            self.teacher_policy_normalizer.to(self.device)
            self.teacher_policy_normalizer.load_state_dict(state_dict["policy_normalizer_state_dict"])
            self.teacher_policy_normalizer.eval()
        else:
            self.teacher_policy_normalizer = None

    def get_teacher_actions(self, obs, critic_obs=None):
        if critic_obs is not None and self.label_action_with_critic_obs and self.action_labels_from_sample:
            if self.teacher_policy_normalizer is not None:
                critic_obs = self.teacher_policy_normalizer(critic_obs)
            return self.teacher_actor_critic.act(critic_obs).detach()
        elif critic_obs is not None and self.label_action_with_critic_obs:
            if self.teacher_policy_normalizer is not None:
                critic_obs = self.teacher_policy_normalizer(critic_obs)
            return self.teacher_actor_critic.act_inference(critic_obs).detach()
        elif self.action_labels_from_sample:
            if self.teacher_policy_normalizer is not None:
                obs = self.teacher_policy_normalizer(obs)
            return self.teacher_actor_critic.act(obs).detach()
        else:
            if self.teacher_policy_normalizer is not None:
                obs = self.teacher_policy_normalizer(obs)
            return self.teacher_actor_critic.act_inference(obs).detach()

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        self.transition = ActionLabelRollout.Transition()
        obs_size = get_subobs_size(obs_format["policy"])
        critic_obs_size = get_subobs_size(obs_format.get("critic")) if "critic" in obs_format else None
        self.storage = ActionLabelRollout(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            num_rewards=num_rewards,
            buffer_dilation_ratio=self.buffer_dilation_ratio,
            device=self.device,
        )

    def act(self, obs, critic_obs):
        # get actions
        return_ = super().act(obs, critic_obs)
        self.transition.action_labels = self.get_teacher_actions(obs, critic_obs)

        # decide whose action to use
        if not hasattr(self, "use_teacher_act_mask"):
            self.use_teacher_act_mask = torch.ones(obs.shape[0], device=self.device, dtype=torch.bool)
        return_[self.use_teacher_act_mask] = self.transition.action_labels[self.use_teacher_act_mask]

        return return_

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        return_ = super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
        self.teacher_actor_critic.reset(dones)
        # resample teacher action mask for those dones env
        self.use_teacher_act_mask[dones.to(bool)] = torch.rand(dones.sum(), device=self.device) < self.teacher_act_prob(
            self.current_learning_iteration
        )
        return return_

    def collect_transition_from_dataset(self, transition, infos):
        """The interface to collect transition from dataset rather than env"""
        super().act(transition.observation, transition.privileged_observation)
        self.transition.action_labels = transition.action
        super().process_env_step(
            transition.reward,
            transition.done,
            infos,
            transition.next_observation,
            transition.next_privileged_observation,
        )

    def compute_returns(self, last_critic_obs):
        if not self.using_ppo:
            return
        return super().compute_returns(last_critic_obs)

    def update(self, *args, **kwargs):
        return_ = super().update(*args, **kwargs)
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return return_

    def compute_distill_loss(self, action_mean, action_labels):
        """Compute distillation loss between predicted actions and teacher action labels.

        Args:
            action_mean: Predicted actions from the current policy
            action_labels: Target actions from the teacher policy

        Returns:
            dist_loss: Computed distillation loss
        """
        if self.distill_target == "real":
            dist_loss = torch.norm(action_mean - action_labels, dim=-1)
        elif self.distill_target == "mse_sum":
            dist_loss = F.mse_loss(
                action_mean,
                action_labels,
                reduction="none",
            ).sum(-1)
        elif self.distill_target == "l1":
            dist_loss = torch.norm(
                action_mean - action_labels,
                dim=-1,
                p=1,
            )
        elif self.distill_target == "tanh":
            # for tanh, similar to loss function for sigmoid, refer to https://stats.stackexchange.com/questions/12754/matching-loss-function-for-tanh-units-in-a-neural-net
            dist_loss = (
                F.binary_cross_entropy(
                    (action_mean + 1) * 0.5,
                    (action_labels + 1) * 0.5,
                )
                * 2
            )
        elif self.distill_target == "scaled_tanh":
            l1 = torch.norm(
                action_mean - action_labels,
                dim=-1,
                p=1,
            )
            dist_loss = (
                F.binary_cross_entropy(
                    (action_mean + 1) * 0.5,
                    (action_labels + 1) * 0.5,  # (n, t, d)
                    reduction="none",
                ).mean(-1)
                * 2
                * l1
                / action_mean.shape[-1]
            )  # (n, t)
        elif self.distill_target == "max_log_prob":
            action_labels_log_prob = self.actor_critic.get_actions_log_prob(action_labels)
            dist_loss = -action_labels_log_prob
        elif self.distill_target == "kl":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown distill_target: {self.distill_target}")

        return dist_loss

    def compute_losses(self, minibatch):
        if self.hidden_state_resample_prob > 0.0:
            # assuming the hidden states are from LSTM or GRU, which are always betwein -1 and 1
            hidden_state_example = (
                minibatch.hidden_states[0][0]
                if isinstance(minibatch.hidden_states[0], tuple)
                else minibatch.hidden_states[0]
            )
            resample_mask = (
                torch.rand(hidden_state_example.shape[1], device=self.device) < self.hidden_state_resample_prob
            )
            # for each hidden state, resample from -1 to 1
            if isinstance(minibatch.hidden_states[0], tuple):
                # for LSTM not tested
                # iterate through actor and critic hidden state
                minibatch = minibatch._replace(
                    hidden_states=tuple(
                        tuple(
                            torch.where(
                                resample_mask.unsqueeze(-1).unsqueeze(-1),
                                torch.rand_like(minibatch.hidden_states[i][j], device=self.device) * 2 - 1,
                                minibatch.hidden_states[i][j],
                            )
                            for j in range(len(minibatch.hidden_states[i]))
                        )
                        for i in range(len(minibatch.hidden_states))
                    )
                )
            else:
                # for GRU
                # iterate through actor and critic hidden state
                minibatch = minibatch._replace(
                    hidden_states=tuple(
                        torch.where(
                            resample_mask.unsqueeze(-1),
                            torch.rand_like(minibatch.hidden_states[i], device=self.device) * 2 - 1,
                            minibatch.hidden_states[i],
                        )
                        for i in range(len(minibatch.hidden_states))
                    )
                )

        if self.using_ppo:
            losses, inter_vars, stats = super().compute_losses(minibatch)
        else:
            losses = dict()
            inter_vars = dict()
            stats = dict()
            self.actor_critic.act(
                minibatch.obs,
                masks=minibatch.masks,
                hidden_states=getattr(minibatch.hidden_states, "actor", None),
            )

        # distillation loss (with teacher actor)
        dist_loss = self.compute_distill_loss(self.actor_critic.action_mean, minibatch.action_labels)

        if "tanh" in self.distill_target:
            stats["l1distance"] = (
                torch.norm(
                    self.actor_critic.action_mean - minibatch.action_labels,
                    dim=-1,
                    p=1,
                )
                .mean()
                .detach()
            )
            stats["l1_before_tanh"] = (
                torch.norm(torch.tan(self.actor_critic.action_mean) - torch.tan(minibatch.action_labels), dim=-1, p=1)
                .mean()
                .detach()
            )
        # update distillation loss coef if applicable
        self.distillation_loss_coef = (
            self.distillation_loss_coef_func(self.current_learning_iteration)
            if hasattr(self, "distillation_loss_coef_func")
            else self.__distillation_loss_coef
        )
        losses["distillation_loss"] = dist_loss.mean()

        return losses, inter_vars, stats
