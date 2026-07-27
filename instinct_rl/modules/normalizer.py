# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=2**62):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it. By default is int64.max / 2 constant.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        # Since-last-sync increments: in multi-process training, sync_across_processes() merges only
        # these into the shared estimate - merging full local estimates would re-count the shared
        # history x world_size per sync and overflow int64. Non-persistent so that old checkpoints
        # stay loadable with strict=True.
        self.register_buffer("_inc_mean", torch.zeros(shape).unsqueeze(0), persistent=False)
        self.register_buffer("_inc_var", torch.ones(shape).unsqueeze(0), persistent=False)
        self.register_buffer("_inc_count", torch.tensor(0, dtype=torch.long), persistent=False)

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            with torch.no_grad():
                self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.count < 0:
            # count overflowed (int64) or the checkpoint is corrupt: freeze, otherwise the negative
            # `rate` would diverge the running mean and poison every normalized observation.
            print(
                f"[EmpiricalNormalization] Error: count is negative ({self.count.item()}),"
                " freezing normalizer statistics. Check resumed checkpoints and multi-process sync."
            )
            return
        if self.until is not None and self.count >= self.until:
            return

        # skip non-finite batches: one nan/inf batch would permanently poison the running stats
        # (and spread to all processes in the next sync).
        if not torch.isfinite(x).all():
            print("[EmpiricalNormalization] Warning: skipping statistics update for a nan/inf batch.")
            return

        count_x = x.shape[0]
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        multi_process = dist.is_initialized() and dist.get_world_size() > 1
        # multi-process: accumulate only the since-last-sync increment; the shared estimate is
        # advanced exclusively by sync_across_processes(). Single-process behavior is unchanged.
        mean_buf = self._inc_mean if multi_process else self._mean
        var_buf = self._inc_var if multi_process else self._var
        count_buf = self._inc_count if multi_process else self.count

        new_count = count_buf + count_x
        rate = count_x / new_count
        delta_mean = mean_x - mean_buf
        mean_buf += rate * delta_mean
        var_buf += rate * (var_x - var_buf + delta_mean * (mean_x - mean_buf))
        if multi_process:
            self._inc_count = new_count
        else:
            self.count = new_count
            self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean

    def init_broadcast(self):
        """Broadcast buffers from rank 0 so all processes start from the same statistics."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        for buffer in self.buffers():
            dist.broadcast(buffer, src=0)

    def sync_across_processes(self):
        """Pool every process's since-last-sync increment into the shared running estimate.

        The shared estimate (identical on all ranks) joins the merge as one more participant,
        keeping `count` the true global sample count.
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        if self.until is not None and self.count >= self.until:
            return

        # do not contribute a poisoned increment to the shared running estimate
        local_count, local_mean, local_var = self._inc_count, self._inc_mean, self._inc_var
        if local_count < 0 or not torch.isfinite(local_mean).all() or not torch.isfinite(local_var).all():
            print(
                "[EmpiricalNormalization] Warning: local increment statistics are invalid (negative count"
                " or nan/inf), excluding this process from the sync."
            )
            local_count = torch.zeros_like(local_count)
            local_mean = torch.zeros_like(local_mean)
            local_var = torch.zeros_like(local_var)

        # pack [count, mean, var] into a single float64 all_gather (counts exact up to 2**53)
        payload = torch.cat([local_count.view(1).double(), local_mean.flatten().double(), local_var.flatten().double()])
        gathered = [torch.empty_like(payload) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, payload)
        gathered = torch.stack(gathered)  # (world_size, 1 + 2 * dim)

        # merge participants: one increment per rank + the shared estimate (identical on all ranks)
        dim = self._mean.numel()
        counts = torch.cat([gathered[:, 0], self.count.double().view(1)])
        means = torch.cat([gathered[:, 1 : 1 + dim], self._mean.double().reshape(1, -1)])
        vars_ = torch.cat([gathered[:, 1 + dim :], self._var.double().reshape(1, -1)])

        inc_total = counts[:-1].sum()
        if inc_total <= 0:  # no new data anywhere (or int64 overflow)
            return

        # Chan parallel merge: global stats of the pooled (shared + increments) dataset
        total = counts.sum()
        global_mean = (counts.unsqueeze(1) * means).sum(0) / total
        global_var = (counts.unsqueeze(1) * (vars_ + (means - global_mean) ** 2)).sum(0) / total

        self._mean = global_mean.view_as(self._mean).to(self._mean.dtype)
        self._var = global_var.view_as(self._var).to(self._var.dtype)
        self._std = torch.sqrt(self._var)
        self.count += inc_total.long()

        # the increments are now merged into the shared estimate; start a fresh increment
        self._inc_mean.zero_()
        self._inc_var.fill_(1.0)
        self._inc_count.zero_()

    def export(self, path):
        np.savez(
            path,
            mean=self._mean.cpu().numpy(),
            std=self._std.cpu().numpy(),
            eps=self.eps,
            until=self.until,
        )


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        if self.training:
            # update discounected rewards
            avg = self.disc_avg.update(rew)

            # update moments from discounted rewards
            self.emp_norm.update(avg)

        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew

    def init_broadcast(self):
        self.emp_norm.init_broadcast()

    def sync_across_processes(self):
        self.emp_norm.sync_across_processes()


class DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
