"""World model, latent decoder, failure head and value head of the SuperMimic trainer (paper Table A.4).

Adapted from the SuperMimic reference implementation (https://github.com/zita-ch/supermimic_vibed, MIT License,
Copyright (c) 2026 ETH Zurich, Chong Zhang; see licenses/dependencies/supermimic_vibed_license.txt).
Task-specific changes: the world model also takes the local terrain height scan and predicts per-body contact logits
(used for the ``undesired_contacts`` reward); with a visual latent as part of its state it also carries the vision of
the task's image policy (`VisualSuperMimic`).

The robot is a floating base with ``num_joints`` actuated joints. Its state is the tuple ``S = (p, R, v, w, q, qd)``:
root position, root rotation matrix, world linear velocity of the root link-frame origin, world angular velocity, joint
positions and joint velocities. ``flatten_state`` / ``unflatten_state`` convert it to / from the (quaternion based) flat
vector of width ``13 + 2 * num_joints`` which is what the replay storage holds.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from instinct_rl.modules.normalizer import RunningNormalization
from instinct_rl.utils.rotations import (
    mat_to_6d,
    mat_to_quat,
    mm,
    mv,
    quat_to_mat,
    so3_exp,
    so3_log,
    yaw_mat,
)

# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32). A process-wide performance switch.
_AMP_DTYPE: torch.dtype | None = None


def set_amp_dtype(dtype: torch.dtype | None):
    """Set the autocast dtype of the MLP matmuls and the latent decoder. None disables autocast."""
    global _AMP_DTYPE
    _AMP_DTYPE = dtype


def get_amp_dtype() -> torch.dtype | None:
    return _AMP_DTYPE


"""
Floating-base state helpers
"""


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    num_joints = (x.shape[-1] - 13) // 2
    return (
        x[..., 0:3],
        quat_to_mat(x[..., 3:7]),
        x[..., 7:10],
        x[..., 10:13],
        x[..., 13 : 13 + num_joints],
        x[..., 13 + num_joints : 13 + 2 * num_joints],
    )


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: tilt (6D), body-frame velocities, joints.

    (The reference version also contains the root height; on terrain the height scan carries that information.)
    """
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            mat_to_6d(mm(H.transpose(-1, -2), R)),
            mv(Rt, v),
            mv(Rt, w),
            q,
            qd,
        ),
        -1,
    )


"""
Networks
"""


def build_mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def run_mlp(net: nn.Module, x: torch.Tensor, amp_dtype: torch.dtype | None) -> torch.Tensor:
    if amp_dtype is None:
        return net(x)
    with torch.autocast("cuda", dtype=amp_dtype):
        return net(x).float()


class FloatingBaseWorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi, terrain scan) predicting frame-local state increments, plus
    logits of the per-body contact indicators at s_{t+1}.

    ``z_dim > 0`` (`VisualSuperMimic`): the visual latent z (the latent of the image encoder of the policy) is part of
    the state, (s_{t+1}, z_{t+1}) = f(s_t, z_t, a_t, xi, scan): z_t is an extra input and the same MLP also predicts its
    increment (:meth:`step_z`). z is normalized with running statistics kept here (``update_z_stats``)."""

    def __init__(
        self,
        num_joints: int,
        context_dim: int,
        scan_dim: int,
        num_contacts: int,
        hidden_sizes=(2048, 2048, 2048),
        z_dim: int = 0,
    ):
        """
        Args:
            num_joints: number of actuated joints (the actions have the same size).
            context_dim: size of the (normalized) episode-constant context xi.
            scan_dim: size of the terrain height scan.
            num_contacts: number of the bodies whose contact indicator is predicted.
        """
        super().__init__()
        self.num_joints = num_joints
        # frame-local increments (p, theta, v, w, q, qd) of the state, see :meth:`targets`
        self.out_dim = 3 + 3 + 3 + 3 + num_joints + num_joints
        # invariant state features (6 + 3 + 3 + q + qd) + action + xi + terrain scan
        self.in_dim = 6 + 3 + 3 + num_joints + num_joints + num_joints + context_dim + scan_dim
        self.num_contacts, self.z_dim = num_contacts, z_dim
        self.net = build_mlp(
            self.in_dim + z_dim,
            list(hidden_sizes),
            self.out_dim + num_contacts + z_dim,
            act=nn.SiLU,
            layer_norm=True,
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[self.out_dim : self.out_dim + num_contacts] = -3.0  # contacts are rare
        self.in_norm = RunningNormalization(self.in_dim)
        self.out_norm = RunningNormalization(self.out_dim, eps=1e-4)
        if z_dim > 0:
            self.register_buffer("z_mean", torch.zeros(z_dim))
            self.register_buffer("z_std", torch.ones(z_dim))
            self.register_buffer("dz_std", torch.ones(z_dim))

    @torch.no_grad()
    def update_z_stats(self, z: torch.Tensor, dz: torch.Tensor, w: torch.Tensor, rate: float = 0.01):
        """EMA of the latent mean / std and of the std of its one-step increments (``w``: validity weights of
        ``dz``)."""
        z, dz, w = z.reshape(-1, self.z_dim), dz.reshape(-1, self.z_dim), w.reshape(-1, 1)
        self.z_mean.lerp_(z.mean(0), rate)
        self.z_std.lerp_(z.std(0).clamp(min=1e-3), rate)
        self.dz_std.lerp_(((dz**2 * w).sum(0) / w.sum().clamp(min=1)).sqrt().clamp(min=1e-3), rate)

    def inputs(self, S, a, xi_n, scan):
        return torch.cat((invariant_features(S), a, xi_n, scan), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv(H.transpose(-1, -2), p1 - p),
                so3_log(mm(Rt, R1)),
                mv(Rt, v1 - v),
                mv(Rt, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n, scan):
        """Returns the next state tuple and the contact logits at the next state."""
        assert self.z_dim == 0, "the world model has a visual latent state: use step_z"
        return self._step(S, run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n, scan)), get_amp_dtype()))

    def step_z(self, S, a, xi_n, scan, z):
        """Returns the next state tuple, the contact logits and the next visual latent (``z_dim > 0``)."""
        x = torch.cat((self.in_norm(self.inputs(S, a, xi_n, scan)), (z - self.z_mean) / self.z_std), -1)
        y = run_mlp(self.net, x, get_amp_dtype())
        S1, logits = self._step(S, y)
        return S1, logits, z + y[..., self.out_dim + self.num_contacts :] * self.dz_std

    def _step(self, S, y):
        p, R, v, w, q, qd = S
        d = y[..., : self.out_dim] * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        p1 = p + mv(H, dp)
        R1 = mm(R, so3_exp(dth))
        v1 = v + mv(R, dv)
        w1 = w + mv(R, dw)
        return (p1, R1, v1, w1, q + dq, qd + dqd), y[..., self.out_dim : self.out_dim + self.num_contacts]

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv(H.transpose(-1, -2), ph - p),
                so3_log(mm(Rt, Rh)),
                mv(Rt, vh - v),
                mv(Rt, wh - w),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


def _check_image_shape(image_shape):
    assert tuple(image_shape) == (1, 18, 32), f"the latent decoder is built for (1, 18, 32) images, got {image_shape}"


class LatentDecoder(nn.Module):
    """Training-only decoder of the visual latent to the CLEAN processed image (1 x 18 x 32): grounds the latent of the
    policy's image encoder (no collapse of the latent prediction loss). Never used in imagination."""

    def __init__(self, latent_dim: int, image_shape: tuple = (1, 18, 32)):
        super().__init__()
        _check_image_shape(image_shape)
        self.fc = nn.Sequential(nn.Linear(latent_dim, 32 * 5 * 8), nn.ELU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=(0, 1)), nn.ELU(),  # -> (9, 16)
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=(1, 1)), nn.ELU(),  # -> (18, 32)
            nn.Conv2d(16, 1, 3, 1, 1),
        )  # fmt: skip

    def forward(self, z):
        amp_dtype = get_amp_dtype()
        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            return torch.sigmoid(self.dec(self.fc(z).view(-1, 32, 5, 8))).float()


class FailureHead(nn.Module):
    """Dreamer-style "continue" predictor: P(the transition into this state is a REAL termination), from the privileged
    critic features of the (true or imagined) state (invariant state, robot-vs-reference errors, xi, goal, terrain
    scan)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.norm = RunningNormalization(in_dim)
        self.net = build_mlp(in_dim, [256, 128], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (B,)."""
        return run_mlp(self.net, self.norm(x), get_amp_dtype()).squeeze(-1)


class ValueHead(nn.Module):
    """Critic of the imagined returns, from the privileged critic features of the state."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = build_mlp(in_dim, [256, 128], 1)
        self.norm = RunningNormalization(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x), get_amp_dtype()).squeeze(-1)
