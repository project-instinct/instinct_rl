import numpy as np
import torch
import torch.nn as nn

from instinct_rl.utils.utils import get_obs_slice


class ActResidualMixin:
    def __init__(self, *args, **kwargs):
        """This is a mixin for action residual based on a given obs_component
        (typically `joints_target`)
        """
        print("Not Ready Yet!")
        exit(-1)
        super().__init__(*args, **kwargs)

    def _get_action_residual(self, obs, privileged=False):
        obs_segments = (
            self.obs_segments
            if (not privileged) or (not hasattr(self, "critic_obs_segments"))
            else self.obs_segments_privileged
        )
        obs_slice = get_obs_slice(obs_segments, "joints_target")
        action_ = obs[obs_slice[0]].reshape(-1, *obs_slice[1])
        return action_[:, 0], action_[:, 1]

    def act(self, *args, **kwargs):
        action = super().act(*args, **kwargs)
