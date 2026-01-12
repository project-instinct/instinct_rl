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
import pathlib
from collections import OrderedDict
from typing import List

import git
import numpy as np
import torch


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
    NOTE: The output shape will also be [time, number of envs, additional dimensions].
        But `time` might be smaller than the input `time`.
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    if padded_trajectories.shape[0] < tensor.shape[0]:
        #     trajectory_masks = trajectory_masks[:padded_trajectories.shape[0]]
        padded_trajectories = torch.cat(
            [
                padded_trajectories,
                torch.empty(
                    (tensor.shape[0] - padded_trajectories.shape[0], *padded_trajectories.shape[1:]),
                    device=tensor.device,
                ),
            ],
            dim=0,
        )
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, masks.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            # skip if not a git repository
            continue
        # get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name
        t = repo.head.commit.tree
        commit_hash = repo.head.commit.hexsha[:7]  # short hash
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}_{commit_hash}.diff")
        # check if the diff file already exists
        if os.path.isfile(diff_file_name):
            continue
        # write the diff file
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x", encoding="utf-8") as f:
            content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            f.write(content)
        # add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths


"""
obs pack -> obs -> Obs component
obs format -> obs segment -> obs component
"""


def get_obs_slice(segments: OrderedDict[str, tuple], component_name: str):
    """Get the slice from segments and name. Return the slice and component shape"""
    obs_start = obs_end = 0
    component_shape = None
    for k, v in segments.items():
        obs_start = obs_end
        obs_end = obs_start + np.prod(v)
        if k == component_name:
            component_shape = v  # tuple
            break
    assert component_shape is not None, "No component ({}) is found in the given components {}".format(
        component_name, [segments.keys()]
    )
    return slice(obs_start, obs_end), component_shape


""" NOTE:
* Loop through obs_segments to get the same order of components defined in obs_segments
* These operations does not require the obs to be a 2-d tensor, but the last dimension must be packed
    with a connected set of components.
"""


def get_subobs_size(obs_segments: OrderedDict[str, tuple], component_names: str = None) -> int:
    """Compute the size of a subset of observations."""
    obs_size = 0
    for component in obs_segments.keys():
        if component_names is None or component in component_names:
            obs_size += np.prod(obs_segments[component])
    return obs_size


def get_subobs_by_components(observations, component_names, obs_segments: OrderedDict, cat=True, temporal=False):
    """Get a subset of observations from the full observation tensor.
    Args:
        cat: If True, concatenate the subobs along the last dimension.
        temporal: If True, the observations will be 3-d tensor with (batch_size, time, obs_size).
    """
    subobs = []
    for component in obs_segments.keys():
        if component in component_names:
            obs_slice, obs_shape = get_obs_slice(obs_segments, component)
            subobs.append(observations[..., obs_slice].reshape(observations.shape[:-1] + obs_shape))
            if temporal:
                subobs[-1] = subobs[-1].reshape(*observations.shape[:-1], obs_shape[0], -1)
            else:
                subobs[-1] = subobs[-1].reshape(*observations.shape[:-1], -1)
    # (batch_size, obs_size) or (batch_size, time, obs_size)
    return torch.cat(subobs, dim=-1) if cat else subobs


def replace_obs_components(
    observations: torch.Tensor, target_components: List[str], replace_vec: torch.Tensor, obs_segments: OrderedDict
):
    """Substitute the vector into a number of segments of the observations.
    NOTE: the output observation will be modified in-place.
    """
    replace_vec_start = 0
    for component in obs_segments:
        if component in target_components:
            obs_slice, obs_shape = get_obs_slice(obs_segments, component)
            replace_vec_end = replace_vec_start + np.prod(obs_shape)
            observations[..., obs_slice] = replace_vec[..., replace_vec_start:replace_vec_end]
            replace_vec_start = replace_vec_end
    return observations


"""
Math
"""


@torch.jit.script
def wrap_to_pi(angles):
    angles_negative_mask = angles < 0
    angles[angles_negative_mask] *= -1
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    angles[angles_negative_mask] *= -1
    return angles


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_to_rotmat(q):
    """q: shape (N, 4) quaternion"""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotmat = torch.zeros(q.shape[0], 3, 3, device=q.device)
    rotmat[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
    rotmat[:, 0, 1] = 2 * x * y - 2 * z * w
    rotmat[:, 0, 2] = 2 * x * z + 2 * y * w
    rotmat[:, 1, 0] = 2 * x * y + 2 * z * w
    rotmat[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
    rotmat[:, 1, 2] = 2 * y * z - 2 * x * w
    rotmat[:, 2, 0] = 2 * x * z - 2 * y * w
    rotmat[:, 2, 1] = 2 * y * z + 2 * x * w
    rotmat[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2
    return rotmat


def rotmat_to_euler_zxy(mat):
    """mat: shape (N, 3, 3) 3d rotation matrix"""
    # get the rotation parameters in y(q0)x(q1)z(q2) sequence
    y = torch.atan2(mat[:, 0, 2], mat[:, 2, 2])  # y
    x = torch.asin(-mat[:, 1, 2])  # x
    z = torch.atan2(mat[:, 1, 0], mat[:, 1, 1])  # z
    y = wrap_to_pi(y)
    x = wrap_to_pi(x)
    z = wrap_to_pi(z)
    return z, x, y


def rotmat_to_euler_yzx(mat):
    """mat: shape (N, 3, 3) 3d rotation matrix"""
    # get the rotation parameters in x(q0)z(q1)y(q2) sequence
    x = torch.atan2(mat[:, 2, 1], mat[:, 1, 1])  # x
    z = torch.asin(-mat[:, 0, 1])  # z
    y = torch.atan2(mat[:, 0, 2], mat[:, 0, 0])  # y
    x = wrap_to_pi(x)
    z = wrap_to_pi(z)
    y = wrap_to_pi(y)
    return y, z, x


def rotmat_to_euler_xzy(mat):
    """mat: shape (N, 3, 3) 3d rotation matrix"""
    # get the rotation parameters in y(q0)z(q1)x(q2) sequence
    y = torch.atan2(-mat[:, 2, 0], mat[:, 0, 0])  # y
    z = torch.asin(mat[:, 1, 0])  # z
    x = torch.atan2(-mat[:, 1, 2], mat[:, 1, 1])  # x
    y = wrap_to_pi(y)
    z = wrap_to_pi(z)
    x = wrap_to_pi(x)
    return x, z, y


def zxy_to_xyz(points):
    """Convert the points from y-up to z-up."""
    return points[..., [2, 0, 1]]


def xyz_to_zxy(points):
    """Convert the points from z-up to y-up."""
    return points[..., [1, 2, 0]]


def module_is_from_type(module, type) -> bool:
    """Check if the module is of the given type.
    Wrapping with this function in order to meet the torch DistributedDataParallel situation.
    """
    if not isinstance(module, type) and hasattr(module, "module"):
        return isinstance(module.module, type)
    else:
        return isinstance(module, type)
