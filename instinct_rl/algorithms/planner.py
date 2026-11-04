"""An upstream planner that fills part of the policy observation, trained by supervision from env samples.

ACP stage 2 (He et al., arXiv:2610.31608): a contact planner predicts the tracker's contact targets from its own
observation group. At every ``act`` it runs (without gradients) and its decoded targets replace the target slot of
the policy observation in the environments whose gate observation is set; the composed observation is what PPO
stores, so minibatches never re-run the planner. The environment labels finished episodes in hindsight and sends
the samples in ``infos[planner_samples_key]``; they go into a FIFO replay buffer, and after every PPO update the
planner takes a few optimizer steps on batches from it. The planner predictions are also returned to the
environment (``get_policy_feedback``) for rewards that follow them.

Normalization: the runner normalizes the policy observation before ``act``. The planner writes its targets after
that, normalized with the policy normalizer's current statistics of the slot (``(x - mean) / (std + eps)``, no
statistics update), so the actor sees planner and environment targets on the same scale as in a stage that only
had environment targets, and a checkpoint of that stage keeps its normalizer. The slot's statistics keep being
updated from the environment's (reference) targets, which every env still emits. The planner group must not be
normalized by the runner: the decoded targets are observed terrain points and must stay in meters (the planner
normalizes its context itself).
"""

from __future__ import annotations

import importlib
import inspect
import os
from collections import defaultdict
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import instinct_rl.modules as instinct_modules
from instinct_rl.algorithms.ppo import PPO
from instinct_rl.modules.contact_planner import ContactPlanner
from instinct_rl.modules.normalizer import EmpiricalNormalization
from instinct_rl.storage.replay_storage import FifoReplayBuffer, TensorSpec
from instinct_rl.utils.utils import get_subobs_indexing_by_components, get_subobs_size

#: Keys of the samples the environment sends (cross-repo contract, see the ACP reproduction plan, Sec. 8).
PLANNER_SAMPLE_KEYS = ("planner_obs", "target_pos", "target_valid")


class PlannerAlgoMixin:
    """Trains a planner (:class:`~instinct_rl.modules.contact_planner.ContactPlanner`) next to the base algorithm.

    Args:
        planner_class_name: Planner class in ``instinct_rl.modules`` or ``"module.path:Class"``.
        planner_kwargs: Its constructor arguments, besides ``input_segments`` (the planner obs group's segments).
        planner_obs_group: Observation group the planner reads.
        planner_target_components: Components of the policy observation the decoded targets fill (in segment
            order; their total size must be the planner's ``output_size``).
        planner_gate_group: Observation group of shape (N, 1): 1 = use the planner's targets, 0 = keep the env's.
        planner_target_noise: Half-width (raw units) of uniform noise added to the planner's targets before they
            enter the policy observation, like the env's observation noise on its own targets; 0 for none.
        planner_positions_feedback_key: Key of the decoded targets (N, Q, 3) in :meth:`get_policy_feedback`.
        planner_occupancy_feedback_key: Key of the occupancy mask (N, Q), as float, in :meth:`get_policy_feedback`.
        planner_samples_key: Key of the labeled samples in the step ``infos`` (dict with ``PLANNER_SAMPLE_KEYS``).
        planner_optimizer_class_name: ``torch.optim`` class.
        planner_optimizer_kwargs: Its arguments (default ``{"lr": 1e-4}``).
        planner_max_grad_norm: Gradient-norm clip of the planner; None for none.
        planner_num_updates: Optimizer steps per iteration (after the base algorithm's update).
        planner_batch_size: Replay samples per optimizer step.
        planner_accumulation_steps: Sub-batches the batch is split into, gradients accumulated over them.
        planner_min_samples: Replay size from which the planner trains (on every rank).
        planner_replay_capacity: Replay buffer size (samples).
        planner_replay_device: Device of the replay buffer; None = the algorithm's device.
        planner_replay_checkpoint: ``"latest"``: the replay is written to ``planner_replay_latest.pt`` beside the
            checkpoints, overwritten at every save, and restored when it belongs to the loaded checkpoint;
            ``"every"``: inside every checkpoint; ``"none"``: not saved.
    """

    REPLAY_FILE = "planner_replay_latest.pt"

    def __init__(
        self,
        *args,
        planner_class_name: str = "AttentionContactPlanner",
        planner_kwargs: dict | None = None,
        planner_obs_group: str = "planner",
        planner_target_components: Sequence[str] = ("contact_targets",),
        planner_gate_group: str = "contact_source",
        planner_target_noise: float = 0.0,
        planner_positions_feedback_key: str = "planned_contacts",
        planner_occupancy_feedback_key: str = "planned_occupancy",
        planner_samples_key: str = "planner_samples",
        planner_optimizer_class_name: str = "Adam",
        planner_optimizer_kwargs: dict | None = None,
        planner_max_grad_norm: float | None = None,
        planner_num_updates: int = 10,
        planner_batch_size: int = 4096,
        planner_accumulation_steps: int = 8,
        planner_min_samples: int = 1000,
        planner_replay_capacity: int = 100_000,
        planner_replay_device: str | None = None,
        planner_replay_checkpoint: str = "latest",
        **kwargs,
    ):
        # the base algorithm only warns on unknown arguments; a planner option with a typo must fail
        accepted = inspect.signature(super().__init__).parameters
        named = {n for n, p in accepted.items() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        unknown = sorted(set(kwargs) - named)
        if unknown:
            raise TypeError(f"{type(self).__name__} got unknown arguments: {unknown}")
        super().__init__(*args, **kwargs)
        if planner_batch_size % planner_accumulation_steps != 0:
            raise ValueError("planner_batch_size must be divisible by planner_accumulation_steps")
        if planner_replay_checkpoint not in ("latest", "every", "none"):
            raise ValueError(
                f"planner_replay_checkpoint must be 'latest', 'every' or 'none', got {planner_replay_checkpoint!r}"
            )
        self.planner_class_name = planner_class_name
        self.planner_kwargs = dict(planner_kwargs or {})
        self.planner_obs_group = planner_obs_group
        self.planner_target_components = list(planner_target_components)
        self.planner_gate_group = planner_gate_group
        self.planner_target_noise = float(planner_target_noise)
        self.planner_positions_feedback_key = planner_positions_feedback_key
        self.planner_occupancy_feedback_key = planner_occupancy_feedback_key
        self.planner_samples_key = planner_samples_key
        self.planner_optimizer_class_name = planner_optimizer_class_name
        self.planner_optimizer_kwargs = dict(planner_optimizer_kwargs or {"lr": 1e-4})
        self.planner_max_grad_norm = planner_max_grad_norm
        self.planner_num_updates = int(planner_num_updates)
        self.planner_batch_size = int(planner_batch_size)
        self.planner_accumulation_steps = int(planner_accumulation_steps)
        self.planner_min_samples = int(planner_min_samples)
        self.planner_replay_capacity = int(planner_replay_capacity)
        self.planner_replay_device = planner_replay_device if planner_replay_device is not None else self.device
        self.planner_replay_checkpoint = planner_replay_checkpoint
        self.policy_normalizer: EmpiricalNormalization | None = None
        self._policy_feedback: dict[str, torch.Tensor] | None = None
        self._planner_samples_this_iteration = 0
        # samples ever added; pairs a checkpoint with its replay sidecar
        self._planner_samples_total = 0

    def set_normalizers(self, normalizers: dict[str, nn.Module]) -> None:
        """Runner hook: the policy normalizer scales the planner's targets like the env's."""
        if hasattr(super(), "set_normalizers"):
            super().set_normalizers(normalizers)  # type: ignore
        if self.planner_obs_group in normalizers:
            raise ValueError(
                f"observation group {self.planner_obs_group!r} must not be normalized by the runner: the planner"
                " decodes observed terrain points in meters and normalizes its context itself"
            )
        normalizer = normalizers.get("policy")
        if normalizer is not None and not isinstance(normalizer, EmpiricalNormalization):
            raise TypeError(f"the policy normalizer must be an EmpiricalNormalization, got {type(normalizer).__name__}")
        self.policy_normalizer = normalizer

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)
        for group in (self.planner_obs_group, self.planner_gate_group):
            if group not in obs_format:
                raise KeyError(f"observation group {group!r} is missing; groups: {list(obs_format)}")
        if get_subobs_size(obs_format[self.planner_gate_group]) != 1:
            raise ValueError(
                f"{self.planner_gate_group!r} must have one value per env, got {obs_format[self.planner_gate_group]}"
            )
        if ":" in self.planner_class_name:
            module_name, class_name = self.planner_class_name.split(":")
            planner_class = getattr(importlib.import_module(module_name), class_name)
        else:
            planner_class = getattr(instinct_modules, self.planner_class_name)
        self.planner: ContactPlanner = planner_class(
            input_segments=obs_format[self.planner_obs_group], **self.planner_kwargs
        ).to(self.device)
        policy_segments = obs_format["policy"]
        missing = [c for c in self.planner_target_components if c not in policy_segments]
        if missing:
            raise KeyError(f"planner target components {missing} are not in the policy observation")
        self._target_index = get_subobs_indexing_by_components(policy_segments, self.planner_target_components).to(
            self.device
        )
        if self._target_index.numel() != self.planner.output_size:
            raise ValueError(
                f"the planner outputs {self.planner.output_size} values, the target components"
                f" {self.planner_target_components} hold {self._target_index.numel()}"
            )
        self.planner_optimizer = getattr(optim, self.planner_optimizer_class_name)(
            self.planner.parameters(), **self.planner_optimizer_kwargs
        )
        num_bodies = self.planner.num_bodies
        self.planner_replay = FifoReplayBuffer(
            self.planner_replay_capacity,
            {
                "planner_obs": TensorSpec((int(get_subobs_size(obs_format[self.planner_obs_group])),)),
                "target_pos": TensorSpec((num_bodies, 3)),
                "target_valid": TensorSpec((num_bodies,), torch.bool),
            },
            device=self.planner_replay_device,
        )

    def act(self, obs, extra_obs):
        return super().act(self.compose_policy_obs(obs, extra_obs, target_noise=True), extra_obs)

    def compose_policy_obs(self, obs, extra_obs, target_noise: bool) -> torch.Tensor:
        """The (normalized) policy obs with the planner's targets in the target slot where the gate is set.

        Also records the planner's decoded targets for :meth:`get_policy_feedback`. ``target_noise`` adds
        ``planner_target_noise`` (training); playback and evaluation pass False.
        """
        gate = extra_obs[self.planner_gate_group]
        with torch.no_grad():
            positions, occupied = self.planner.predict(extra_obs[self.planner_obs_group])
        self._policy_feedback = {
            self.planner_positions_feedback_key: positions,
            self.planner_occupancy_feedback_key: occupied.float(),
        }
        targets = positions.reshape(positions.shape[0], -1)
        if target_noise and self.planner_target_noise > 0.0:
            targets = targets + (2.0 * torch.rand_like(targets) - 1.0) * self.planner_target_noise
        if self.policy_normalizer is not None:
            index = self._target_index
            targets = (targets - self.policy_normalizer.mean[index]) / (
                self.policy_normalizer.std[index] + self.policy_normalizer.eps
            )
        composed = obs.clone()
        composed[:, self._target_index] = torch.where(gate > 0.5, targets, obs[:, self._target_index])
        return composed

    def get_policy_feedback(self) -> dict[str, torch.Tensor]:
        """Runner hook (sent to the env before its step): the planner's decoded targets of the last ``act``."""
        if self._policy_feedback is None:
            raise RuntimeError("get_policy_feedback() called before act()")
        return self._policy_feedback

    def process_env_step(self, rewards, dones, infos, next_obs, next_extra_obs):
        samples = infos.pop(self.planner_samples_key)
        added = self.planner_replay.add({k: samples[k] for k in PLANNER_SAMPLE_KEYS})
        self._planner_samples_this_iteration += added
        self._planner_samples_total += added
        super().process_env_step(rewards, dones, infos, next_obs, next_extra_obs)

    def _planner_ready(self) -> bool:
        size = torch.tensor(len(self.planner_replay), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(size, op=dist.ReduceOp.MIN)
        return int(size) >= self.planner_min_samples

    def update(self, *args, **kwargs):
        mean_losses, average_stats = super().update(*args, **kwargs)
        average_stats["planner_replay_size"] = torch.tensor(float(len(self.planner_replay)))
        average_stats["planner_samples_added"] = torch.tensor(float(self._planner_samples_this_iteration))
        self._planner_samples_this_iteration = 0
        if not self._planner_ready():
            return mean_losses, average_stats

        self.planner.train()
        sums = defaultdict(float)
        sub_batch = self.planner_batch_size // self.planner_accumulation_steps
        for _ in range(self.planner_num_updates):
            batch = self.planner_replay.sample(self.planner_batch_size)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.planner_optimizer.zero_grad()
            for start in range(0, self.planner_batch_size, sub_batch):
                part = {k: v[start : start + sub_batch] for k, v in batch.items()}
                losses, stats = self.planner.compute_loss(part["planner_obs"], part["target_pos"], part["target_valid"])
                (losses["loss"] / self.planner_accumulation_steps).backward()
                for k, v in {**losses, **stats}.items():
                    sums[k] += v.detach() / self.planner_accumulation_steps
            self._planner_gradient_step(sums)
        for k, v in sums.items():
            target = average_stats if k in ("top1_error", "occupancy_accuracy", "grad_norm") else mean_losses
            target[f"planner_{k}"] = v / self.planner_num_updates
        return mean_losses, average_stats

    def _planner_gradient_step(self, sums: dict) -> None:
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for param in self.planner.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
        if self.planner_max_grad_norm is not None:
            sums["grad_norm"] += nn.utils.clip_grad_norm_(
                self.planner.parameters(), self.planner_max_grad_norm
            ).detach()
        self.planner_optimizer.step()

    def distributed_data_parallel(self):
        super().distributed_data_parallel()
        for param in self.planner.parameters():
            dist.broadcast(param.data, src=0)
        if self.planner.context_normalizer is not None:
            self.planner.context_normalizer.init_broadcast()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["planner_state_dict"] = self.planner.state_dict()
        state_dict["planner_optimizer_state_dict"] = self.planner_optimizer.state_dict()
        state_dict["planner_replay_samples_total"] = self._planner_samples_total
        if self.planner_replay_checkpoint == "every":
            state_dict["planner_replay_state_dict"] = self.planner_replay.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        missing = [k for k in ("planner_state_dict", "planner_optimizer_state_dict") if k not in state_dict]
        if missing:
            raise KeyError(
                f"the checkpoint has no {missing}; to start from a checkpoint without a planner, load it with the"
                " runner option ckpt_manipulator='add_missing_algorithm_state'"
            )
        super().load_state_dict(state_dict)
        self.planner.load_state_dict(state_dict["planner_state_dict"])
        self.planner_optimizer.load_state_dict(state_dict["planner_optimizer_state_dict"])
        self._planner_samples_total = int(state_dict.get("planner_replay_samples_total", 0))
        if "planner_replay_state_dict" in state_dict:
            self.planner_replay.load_state_dict(state_dict["planner_replay_state_dict"])

    def save_auxiliary(self, directory: str) -> None:
        """Runner hook after each checkpoint: write the replay to ``planner_replay_latest.pt`` (``"latest"`` mode),
        through a temporary file so a failed write never replaces the last good one."""
        if hasattr(super(), "save_auxiliary"):
            super().save_auxiliary(directory)  # type: ignore
        if self.planner_replay_checkpoint != "latest":
            return
        path = os.path.join(directory, self.REPLAY_FILE)
        state = {"samples_total": self._planner_samples_total, "replay_state_dict": self.planner_replay.state_dict()}
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load_auxiliary(self, directory: str, state: dict) -> None:
        """Runner hook after loading: restore the replay sidecar if it was written with this checkpoint."""
        if hasattr(super(), "load_auxiliary"):
            super().load_auxiliary(directory, state)  # type: ignore
        path = os.path.join(directory, self.REPLAY_FILE)
        if self.planner_replay_checkpoint != "latest" or not os.path.exists(path):
            return
        saved = torch.load(path, weights_only=True, map_location=self.planner_replay.device)
        if int(saved["samples_total"]) != int(state.get("planner_replay_samples_total", -1)):
            print(f"[PlannerAlgoMixin] {path} does not belong to the loaded checkpoint; starting with an empty replay")
            return
        self.planner_replay.load_state_dict(saved["replay_state_dict"])

    def export_as_onnx(self, obs: torch.Tensor, extra_obs: dict[str, torch.Tensor], export_dir: str) -> None:
        """Runner hook: the actor files as usual, plus ``planner.onnx`` (input: the flat planner observation)."""
        self.actor_critic.export_as_onnx(obs, export_dir)
        self.planner.export_as_onnx(extra_obs[self.planner_obs_group][:1], export_dir)


class ContactPlannerPPO(PlannerAlgoMixin, PPO):
    pass
