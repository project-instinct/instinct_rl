""" A implementation for additional losses for (Variational) Autoencoders. """

import importlib

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from instinct_rl.algorithms.tppo import TPPO
from instinct_rl.utils.utils import get_subobs_by_components


class VaeAlgoMixin:
    def __init__(
        self,
        *args,
        kl_loss_func="kl_divergence",
        kl_loss_coef=1.0,
        using_ppo=False,
        **kwargs,
    ):
        assert not using_ppo, "PPO is not supported for VaeDistill"
        super().__init__(*args, using_ppo=using_ppo, **kwargs)
        self.using_ppo = using_ppo
        if ":" in kl_loss_func:
            kl_loss_func_module, kl_loss_func_name = kl_loss_func.split(":")
            self.kl_loss_func = getattr(importlib.import_module(kl_loss_func_module), kl_loss_func_name)
        else:
            self.kl_loss_func = getattr(dist.kl, kl_loss_func) if "kl" in kl_loss_func else getattr(F, kl_loss_func)
        self.kl_loss_coef = kl_loss_coef

        self.normal_dist = dist.Normal(0, 1)

    def compute_losses(self, minibatch):
        losses, inter_vars, stats = super().compute_losses(minibatch)

        # VAE losses
        kl_loss = torch.zeros_like(minibatch.obs[..., 0])
        if hasattr(self.actor_critic, "distribution") and self.actor_critic.distribution is not None:
            # For VaeActor - use the VAE distribution directly
            kl_loss += self.kl_loss_func(
                self.actor_critic.distribution,
                self.normal_dist,
            ).sum(dim=-1)

        losses["kl_loss"] = kl_loss.mean()

        return losses, inter_vars, stats


class VaeDistill(VaeAlgoMixin, TPPO):
    pass
