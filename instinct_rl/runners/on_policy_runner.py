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

import importlib
import os
import statistics
import time
from collections import deque

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

import instinct_rl
import instinct_rl.algorithms as algorithms
import instinct_rl.modules as modules
from instinct_rl.env import VecEnv
from instinct_rl.utils import ckpt_manipulator
from instinct_rl.utils.utils import get_subobs_size, store_code_state


class OnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        """
        ## Args:
            - mp_stat_all_reduce: All reduce statistics in multi-processing if True. NOTE: make sure statistics across all processes have
                the same items, otherwise it will raise an error.
        """
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        obs_format = env.get_obs_format()

        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
            obs_format,
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
        ).to(self.device)

        alg_class_name = self.alg_cfg.pop("class_name")
        alg_class = importlib.import_module() if ":" in alg_class_name else getattr(algorithms, alg_class_name)
        self.alg: algorithms.PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # handle normalizers if needed
        self.normalizers = {}
        for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
            config: dict = config.copy()
            normalizer = modules.build_normalizer(
                input_shape=get_subobs_size(obs_format[obs_group_name]),
                normalizer_class_name=config.pop("class_name"),
                normalizer_kwargs=config,
            )
            normalizer.to(self.device)
            self.normalizers[obs_group_name] = normalizer

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format=obs_format,
            num_actions=self.env.num_actions,
            num_rewards=self.env.num_rewards,
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)
        self.git_status_repos = [instinct_rl.__file__]  # store files whose repo status will be logged.

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # Must call distributed_data_parallel after load_run (in train.py). Otherwise, a DDP algo cannot
        # load from a checkpoint trained in non-DDP mode.
        if dist.is_initialized():
            self.alg.distributed_data_parallel()
            print(f"[INFO rank {dist.get_rank()}]: DistributedDataParallel enabled.")
        # initialize writer
        if self.log_dir is not None and self.writer is None and (not self.is_mp_rank_other_process()):
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        obs = obs.to(self.device)
        critic_obs = extras["observations"].get("critic", None)
        critic_obs = critic_obs.to(self.device) if critic_obs is not None else None
        self.train_mode()

        ep_infos = []
        step_infos = []
        rframebuffer = [deque(maxlen=2000) for _ in range(self.env.num_rewards)]
        rewbuffer = [deque(maxlen=100) for _ in range(self.env.num_rewards)]
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, self.env.num_rewards, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        print(
            "[INFO{}]: Initialization done, start learning.".format(
                f" rank {dist.get_rank()}" if dist.is_initialized() else ""
            )
        )
        print(
            "NOTE: you may see a bunch of `NaN or Inf found in input tensor` once and appears in the log. Just ignore"
            " it if it does not affect the performance."
        )
        if self.log_dir is not None and (not self.is_mp_rank_other_process()):
            store_code_state(self.log_dir, self.git_status_repos)
        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        tot_start_time = time.time()
        start = time.time()
        while self.current_learning_iteration < tot_iter:
            # Rollout
            with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
                for i in range(self.num_steps_per_env):
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
                    if len(rewards.shape) == 1:
                        rewards = rewards.unsqueeze(-1)

                    if self.log_dir is not None:
                        # Book keeping
                        if "step" in infos:
                            step_infos.append(infos["step"])
                        if "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)[:, 0]
                        for i in range(self.env.num_rewards):
                            rframebuffer[i].extend(rewards[dones < 1][:, i].cpu().numpy().tolist())
                            rewbuffer[i].extend(cur_reward_sum[new_ids][:, i].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs if critic_obs is not None else obs)

            losses, stats = self.alg.update(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
            if (
                self.current_learning_iteration % self.save_interval == 0
                and self.current_learning_iteration > start_iter
            ):
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            ep_infos.clear()
            step_infos.clear()
            self.current_learning_iteration = self.current_learning_iteration + 1
            start = time.time()

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def rollout_step(self, obs, critic_obs):
        actions = self.alg.act(obs, critic_obs)
        obs, rewards, dones, infos = self.env.step(actions)
        critic_obs = infos["observations"].get("critic", None)
        obs, critic_obs, rewards, dones = (
            obs.to(self.device),
            critic_obs.to(self.device) if critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )
        # Dealing with obs normalizers. At this line, obs and critic_obs and all observations in info are still flattened
        for obs_group_name, normalizer in self.normalizers.items():
            if obs_group_name == "policy":
                obs = normalizer(obs)
                infos["observations"]["policy"] = obs
            elif obs_group_name == "critic":
                # Doesn't need to worry about `critic is None`. Otherwise, error shall occur when normalizers are being built
                critic_obs = normalizer(critic_obs)
                infos["observations"]["critic"] = critic_obs
            else:
                infos["observations"][obs_group_name] = normalizer(infos["observations"][obs_group_name])
        self.alg.process_env_step(rewards, dones, infos, obs, critic_obs)
        return obs, critic_obs, rewards, dones, infos

    """
    Logging
    """

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time = time.time() - locs["tot_start_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = []
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor.append(ep_info[key].to(self.device))
                infotensor = torch.cat(infotensor)
                if "_max" in key:
                    value = self.gather_stat_values(infotensor, "max")
                elif "_min" in key:
                    value = self.gather_stat_values(infotensor, "min")
                else:
                    value = self.gather_stat_values(infotensor, "mean")
                self.writer_mp_add_scalar(
                    (key if key.startswith("Episode") else "Episode/" + key),
                    value,
                    self.current_learning_iteration,
                )
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if locs["step_infos"]:
            for key in locs["step_infos"][0]:
                infotensor = []
                texts = []
                for step_info in locs["step_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if isinstance(step_info[key], str):
                        texts.append(step_info[key])
                        continue
                    elif not isinstance(step_info[key], torch.Tensor):
                        step_info[key] = torch.Tensor([step_info[key]])
                    if len(step_info[key].shape) == 0:
                        step_info[key] = step_info[key].unsqueeze(0)
                    infotensor.append(step_info[key].to(self.device))
                infotensor = torch.cat(infotensor)
                if "_max" in key:
                    value = self.gather_stat_values(infotensor, "max")
                elif "_min" in key:
                    value = self.gather_stat_values(infotensor, "min")
                else:
                    value = self.gather_stat_values(infotensor, "mean")
                if len(texts) > 0 and (not self.is_mp_rank_other_process()):
                    self.writer.add_text(
                        (key if key.startswith("Step") else "Step/" + key),
                        "\n".join(texts),
                        self.current_learning_iteration,
                    )
                else:
                    self.writer_mp_add_scalar(
                        (key if key.startswith("Step") else "Step/" + key),
                        value,
                        self.current_learning_iteration,
                    )
                    ep_string += f"""{f'Mean step {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))
        if dist.is_initialized():
            fps *= dist.get_world_size()

        for k, v in locs["losses"].items():
            v = self.gather_stat_values(v, "mean")
            self.writer_mp_add_scalar("Loss/" + k, v.item(), self.current_learning_iteration)
        for k, v in locs["stats"].items():
            v = self.gather_stat_values(v, "mean")
            self.writer_mp_add_scalar("Train/" + k, v.item(), self.current_learning_iteration)

        self.writer_mp_add_scalar("Loss/learning_rate", self.alg.learning_rate, self.current_learning_iteration)
        self.writer_mp_add_scalar("Policy/mean_noise_std", mean_std.item(), self.current_learning_iteration)
        self.writer_mp_add_scalar("Perf/total_fps", fps, self.current_learning_iteration)
        self.writer_mp_add_scalar("Perf/collection_time", locs["collection_time"], self.current_learning_iteration)
        self.writer_mp_add_scalar("Perf/learning_time", locs["learn_time"], self.current_learning_iteration)
        self.writer_mp_add_scalar(
            "Perf/gpu_allocated", torch.cuda.memory_allocated(self.device) / 1024**3, self.current_learning_iteration
        )
        self.writer_mp_add_scalar(
            "Perf/gpu_total", torch.cuda.mem_get_info(self.device)[1] / 1024**3, self.current_learning_iteration
        )
        self.writer_mp_add_scalar(
            "Perf/gpu_global_free_mem",
            torch.cuda.mem_get_info(self.device)[0] / 1024**3,
            self.current_learning_iteration,
        )
        for i in range(self.env.num_rewards):
            self.writer_mp_add_scalar(
                f"Train/mean_reward_each_timestep_{i}",
                statistics.mean(locs["rframebuffer"][i]),
                self.current_learning_iteration,
            )
        if len(locs["rewbuffer"][0]) > 0:
            for i in range(self.env.num_rewards):
                self.writer_mp_add_scalar(
                    f"Train/mean_reward_{i}", statistics.mean(locs["rewbuffer"][i]), self.current_learning_iteration
                )
                self.writer_mp_add_scalar(
                    f"Train/ratio_above_mean_reward_{i}",
                    statistics.mean(
                        [(1.0 if rew > statistics.mean(locs["rewbuffer"][i]) else 0) for rew in locs["rewbuffer"][i]]
                    ),
                    self.current_learning_iteration,
                )
                self.writer_mp_add_scalar(
                    f"Train/time/mean_reward_{i}", statistics.mean(locs["rewbuffer"][i]), self.tot_time
                )
            self.writer_mp_add_scalar(
                "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/median_episode_length", statistics.median(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/min_episode_length", min(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/max_episode_length", max(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/time/mean_episode_length", statistics.mean(locs["lenbuffer"]), self.tot_time
            )

        info_str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"][0]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{info_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for k, v in locs["losses"].items():
                log_string += f"""{k:>{pad}} {v.item():.4f}\n"""
            log_string += (
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean([statistics.mean(buf) for buf in locs['rewbuffer']]):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{info_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for k, v in locs["losses"].items():
                log_string += f"""{k:>{pad}} {v.item():.4f}\n"""
            log_string += (
                f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration + 1 - locs["start_iter"]) * (
                               locs['tot_iter'] - self.current_learning_iteration):.1f}s\n"""
        )
        if not self.is_mp_rank_other_process():
            print(log_string)

    def add_git_repo_to_log(self, repo_path):
        self.git_status_repos.append(repo_path)

    def save(self, path, infos=None):
        """Save training state dict to file. Will not happen if in multi-process and not rank 0."""
        if self.is_mp_rank_other_process():
            return

        run_state_dict = self.alg.state_dict()
        run_state_dict.update(  # Nothing will update if there is no normalizers
            {
                f"{group_name}_normalizer_state_dict": normalizer.state_dict()
                for group_name, normalizer in self.normalizers.items()
            }
        )
        run_state_dict.update(
            {
                "iter": self.current_learning_iteration,
                "infos": infos,
            }
        )
        torch.save(run_state_dict, path)

    def load(self, path):
        """Load training state dict from file. Will not happen if in multi-process and not rank 0."""
        if self.is_mp_rank_other_process():
            return

        loaded_dict = torch.load(path, weights_only=True)
        if self.cfg.get("ckpt_manipulator", False):
            # suppose to be a string specifying which function to use
            print("\033[1;36m Warning: using a hacky way to load the model. \033[0m")
            loaded_dict = getattr(ckpt_manipulator, self.cfg["ckpt_manipulator"])(
                loaded_dict,
                self.alg.state_dict(),
                **self.cfg.get("ckpt_manipulator_kwargs", {}),
            )
            print("\033[1;36m Done: using a hacky way to load the model. \033[0m")

        self.alg.load_state_dict(loaded_dict)

        for group_name, normalizer in self.normalizers.items():
            if not f"{group_name}_normalizer_state_dict" in loaded_dict:
                print(
                    f"\033[1;36m Warning, normalizer for {group_name} is not found, the state dict is not loaded"
                    " \033[0m"
                )
            else:
                normalizer.load_state_dict(loaded_dict[f"{group_name}_normalizer_state_dict"])

        self.current_learning_iteration = loaded_dict["iter"]
        if self.cfg.get("ckpt_manipulator", False):
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            except Exception as e:
                print(f"\033[1;36m Save manipulated checkpoint failed with error: {str(e)} \n ignored... \033[0m")
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.actor_critic.to(device)

        if "policy" in self.normalizers:
            self.normalizers["policy"].to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.normalizers["policy"](x))  # noqa: E731
        else:
            policy = self.alg.actor_critic.act_inference

        return policy

    def export_as_onnx(self, obs, export_model_dir):
        self.eval_mode()
        if "policy" in self.normalizers:
            obs = self.normalizers["policy"](obs)
            # also export obs normalizer
            self.normalizers["policy"].export(os.path.join(export_model_dir, "policy_normalizer.npz"))
        self.alg.actor_critic.export_as_onnx(obs, export_model_dir)

    """
    Helper functions
    """

    def is_mp_rank_zero_process(self):
        """Check if the current process should do all reduce to summarize the training statistics and checkpoint."""
        return dist.is_initialized() and dist.get_rank() == 0

    def is_mp_rank_other_process(self):
        """Check if current process is in torch distributed multi-processing and not rank 0, or single process."""
        return dist.is_initialized() and dist.get_rank() != 0

    def gather_stat_values(self, values: torch.Tensor, gather_op: str = "mean", remove_nan: bool = True):
        """Properly gather the value across all processes. summarize the input values directly if not in multi-processing.
        ## Args:
            - values: torch.Tensor, the value to summarize. Do not summarize this into a scalar before calling this function.
            - gather_op: dist.ReduceOp, the operation to contat the value.
            - remove_nan: bool, whether to remove NaN values before summarizing.
        ## Return:
            The summarized value across all processes.
        """
        if remove_nan:
            values = values[~values.isnan()]
        values = values.to(self.device)  # make sure the values are on the all_reduc-able device
        if gather_op == "mean":
            num_values = torch.tensor([torch.numel(values)]).to(self.device)
            values = torch.sum(values)
            if dist.is_initialized():
                dist.all_reduce(values, dist.ReduceOp.SUM)
                dist.all_reduce(num_values, dist.ReduceOp.SUM)
            values = values / num_values.item()
        elif gather_op == "max":
            values = torch.max(values) if values.numel() > 0 else torch.tensor(float("-inf"), device=self.device)
            if dist.is_initialized():
                dist.all_reduce(values, dist.ReduceOp.MAX)
        elif gather_op == "min":
            values = torch.min(values) if values.numel() > 0 else torch.tensor(float("inf"), device=self.device)
            if dist.is_initialized():
                dist.all_reduce(values, dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unsupported gather_op: {gather_op}")
        return values

    def writer_mp_add_scalar(self, key, value, step):
        """Add scalar to tensorboard writer. Will not happen if in multi-process and not rank 0."""
        if not self.is_mp_rank_other_process():
            self.writer.add_scalar(key, value, step)

    def train_mode(self):
        """Change all related models into training mode (for dropout for example)"""
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        for normalizer in self.normalizers.values():
            normalizer.train()

    def eval_mode(self):
        """Change all related models into evaluation mode (for dropout for example)"""
        self.alg.actor_critic.eval()
        for normalizer in self.normalizers.values():
            normalizer.eval()
