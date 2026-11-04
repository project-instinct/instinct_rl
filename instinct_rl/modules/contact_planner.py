"""Contact planners: for each of a set of bodies, select the next contact among observed terrain points.

ACP (He et al., "Humanoid Mobility via Attention-based Contacts", arXiv:2610.31608): from a grid of terrain points
in the robot body-yaw frame and a context vector (proprioceptive history and goal), a planner outputs for each body
a distribution over the points (the "affordance") and an occupancy logit (whether the body makes a new contact
within the planning horizon). Decoding picks the top-k points; targets of bodies with occupancy probability at or
below the threshold are zero, as in the tracker's observation.

``AttentionContactPlanner`` is the paper's planner (Supp. Table S3) and ``MlpContactPlanner`` its MLP baseline; both
share :class:`ContactPlanner`'s inputs, decoding, loss (paper Eq. 1-2) and export, so the baseline is a config switch.

Planner observation layout (one flat vector per sample, segments as in ``obs_format``):

- ``points_xy_component``: shape ``(rows, cols, 2)``, the x, y of every terrain point (body-yaw frame);
- ``points_z_component``: shape ``(rows, cols)``, their heights relative to the base (the CNN's image);
- ``context_components``: everything the context encoder reads (proprioception history, goal).
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FiLM, CrossAttention
from .conv2d import Conv2dModel
from .mlp import MlpModel
from .normalizer import EmpiricalNormalization
from instinct_rl.utils.utils import get_obs_slice, get_subobs_by_components


class ContactPlanner(nn.Module):
    """Inputs, decoding, loss and export shared by the contact planners. Subclasses implement :meth:`scores`.

    Args:
        input_segments: Segments of the planner observation group (``obs_format[group]``).
        points_xy_component: Component holding the points' x, y, shape ``(rows, cols, 2)``.
        points_z_component: Component holding the points' heights, shape ``(rows, cols)``.
        context_components: Components of the context vector, in any order (read in segment order).
        num_bodies: Number of contact bodies (queries).
        top_k: Number of points combined when decoding.
        temperature: Softmax temperature over the top-k log-weights when decoding.
        occupancy_threshold: Targets are kept when the occupancy probability exceeds it.
        target_sigma: Bandwidth (m) of the Gaussian soft target over the points (paper's sigma_a).
        affordance_loss_coef: Weight of the masked soft-target cross-entropy.
        occupancy_loss_coef: Weight of the occupancy binary cross-entropy.
        normalize_context: Normalize the context vector with running statistics, updated from training batches
            only (:meth:`compute_loss`); exported with the planner.
    """

    def __init__(
        self,
        input_segments: OrderedDict,
        points_xy_component: str,
        points_z_component: str,
        context_components: Sequence[str],
        num_bodies: int,
        top_k: int,
        temperature: float,
        occupancy_threshold: float,
        target_sigma: float,
        affordance_loss_coef: float,
        occupancy_loss_coef: float,
        normalize_context: bool,
    ):
        super().__init__()
        self.input_segments = OrderedDict(input_segments)
        self.points_xy_component = points_xy_component
        self.points_z_component = points_z_component
        self.context_components = list(context_components)
        xy_shape = tuple(self.input_segments[points_xy_component])
        z_shape = tuple(self.input_segments[points_z_component])
        if len(xy_shape) != 3 or xy_shape[-1] != 2 or xy_shape[:2] != z_shape:
            raise ValueError(
                f"points must be shaped (rows, cols, 2) and (rows, cols); got {points_xy_component}: {xy_shape},"
                f" {points_z_component}: {z_shape}"
            )
        missing = [c for c in self.context_components if c not in self.input_segments]
        if missing or not self.context_components:
            raise ValueError(f"context components {missing or '[]'} not in {list(self.input_segments)}")
        self.grid_shape = z_shape
        self.num_points = int(np.prod(z_shape))
        self.num_bodies = int(num_bodies)
        self.context_size = int(sum(np.prod(self.input_segments[c]) for c in self.context_components))
        if top_k < 1 or top_k > self.num_points:
            raise ValueError(f"top_k must be in [1, {self.num_points}], got {top_k}")
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.occupancy_threshold = float(occupancy_threshold)
        self.target_sigma = float(target_sigma)
        self.affordance_loss_coef = float(affordance_loss_coef)
        self.occupancy_loss_coef = float(occupancy_loss_coef)
        self.context_normalizer = EmpiricalNormalization(self.context_size) if normalize_context else None
        self._xy_slice = get_obs_slice(self.input_segments, points_xy_component)[0]
        self._z_slice = get_obs_slice(self.input_segments, points_z_component)[0]

    @property
    def output_size(self) -> int:
        """Size of the flattened decoded targets (the policy-observation slot they fill)."""
        return self.num_bodies * 3

    def split_obs(self, planner_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Points (B, P, 3) and the raw (not normalized) context (B, C) of flat planner observations."""
        batch = planner_obs.shape[0]
        xy = planner_obs[:, self._xy_slice].reshape(batch, self.num_points, 2)
        z = planner_obs[:, self._z_slice].reshape(batch, self.num_points, 1)
        context = get_subobs_by_components(planner_obs, self.context_components, self.input_segments)
        return torch.cat([xy, z], dim=-1), context

    def scores(self, points: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Affordance log-probabilities (B, Q, P), normalized over P, and occupancy logits (B, Q)."""
        raise NotImplementedError

    def forward(self, planner_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Points (B, P, 3), affordance log-probabilities (B, Q, P) and occupancy logits (B, Q)."""
        points, context = self.split_obs(planner_obs)
        if self.context_normalizer is not None:
            context = self.context_normalizer.normalize(context)
        log_affordance, occupancy_logits = self.scores(points, context)
        return points, log_affordance, occupancy_logits

    def decode(
        self, points: torch.Tensor, log_affordance: torch.Tensor, occupancy_logits: torch.Tensor, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Targets (B, Q, 3), the temperature-weighted mean of the top-k points, zero where not occupied, and the
        occupancy mask (B, Q)."""
        top_log_weights, top_index = log_affordance.topk(top_k, dim=-1)  # (B, Q, k)
        weights = torch.softmax(top_log_weights / self.temperature, dim=-1)
        batch, num_queries = top_index.shape[:2]
        expanded = points.unsqueeze(1).expand(batch, num_queries, self.num_points, 3)
        selected = torch.gather(expanded, 2, top_index.unsqueeze(-1).expand(-1, -1, -1, 3))  # (B, Q, k, 3)
        positions = (weights.unsqueeze(-1) * selected).sum(dim=2)
        occupied = torch.sigmoid(occupancy_logits) > self.occupancy_threshold
        return positions * occupied.unsqueeze(-1), occupied

    def predict(self, planner_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoded targets (B, Q, 3), zero where not occupied, and the occupancy mask (B, Q)."""
        points, log_affordance, occupancy_logits = self.forward(planner_obs)
        return self.decode(points, log_affordance, occupancy_logits, self.top_k)

    def compute_loss(
        self, planner_obs: torch.Tensor, target_pos: torch.Tensor, target_valid: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Paper Eq. 1-2: masked soft-target cross-entropy plus occupancy binary cross-entropy.

        Args:
            planner_obs: (B, D) planner observations.
            target_pos: (B, Q, 3) hindsight targets in the body-yaw frame (ignored where not valid).
            target_valid: (B, Q) whether the body has a target (the hindsight occupancy flag).

        Returns:
            Losses (``affordance_loss``, ``occupancy_loss``, ``loss`` = their weighted sum) and detached stats
            (``top1_error``: mean distance of the decoded top-1 target to valid labels; ``occupancy_accuracy``).
        """
        if self.training and self.context_normalizer is not None:
            self.context_normalizer.update(self.split_obs(planner_obs)[1])
        points, log_affordance, occupancy_logits = self.forward(planner_obs)
        valid = target_valid.to(log_affordance.dtype)
        squared_distance = (points.unsqueeze(1) - target_pos.unsqueeze(2)).square().sum(dim=-1)  # (B, Q, P)
        soft_target = torch.softmax(-squared_distance / (2.0 * self.target_sigma**2), dim=-1)
        cross_entropy = -(soft_target * log_affordance).sum(dim=-1)
        affordance_loss = (cross_entropy * valid).sum() / valid.sum().clamp_min(1.0)
        occupancy_loss = F.binary_cross_entropy_with_logits(occupancy_logits, valid)
        losses = {
            "affordance_loss": affordance_loss,
            "occupancy_loss": occupancy_loss,
            "loss": self.affordance_loss_coef * affordance_loss + self.occupancy_loss_coef * occupancy_loss,
        }
        with torch.no_grad():
            top1 = self.decode(points, log_affordance, torch.full_like(occupancy_logits, 1e4), 1)[0]
            error = (top1 - target_pos).norm(dim=-1)
            occupied = torch.sigmoid(occupancy_logits) > self.occupancy_threshold
            stats = {
                "top1_error": (error * valid).sum() / valid.sum().clamp_min(1.0),
                "occupancy_accuracy": (occupied == target_valid.bool()).float().mean(),
            }
        return losses, stats

    def export_as_onnx(self, planner_obs: torch.Tensor, filedir: str, filename: str = "planner.onnx") -> str:
        """Export :meth:`predict` (outputs ``positions`` (B, Q, 3) and ``occupancy`` probability (B, Q)) for a
        batch of planner observations, typically of size 1. Returns the written path."""
        self.eval()
        path = os.path.join(filedir, filename)
        with torch.no_grad():
            exported_program = torch.onnx.export(
                _PlannerInference(self),
                (planner_obs,),
                input_names=["input"],
                output_names=["positions", "occupancy"],
                dynamo=True,
                opset_version=18,
            )
        exported_program.save(path)
        print(f"Exported {type(self).__name__} to {path}")
        return path


class _PlannerInference(nn.Module):
    """ONNX view of a planner: gated targets and occupancy probability."""

    def __init__(self, planner: ContactPlanner):
        super().__init__()
        self.planner = planner

    def forward(self, planner_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points, log_affordance, occupancy_logits = self.planner(planner_obs)
        positions, _ = self.planner.decode(points, log_affordance, occupancy_logits, self.planner.top_k)
        return positions, torch.sigmoid(occupancy_logits)


class AttentionContactPlanner(ContactPlanner):
    """ACP's attention planner (Supp. Table S3; the defaults reproduce its 786,993 parameters with a 375-d
    proprioceptive history).

    Each terrain point gets a linear embedding of its (x, y, z) and the features of a two-layer CNN over the height
    image; projected to ``d_model`` they are the keys/values. A pointwise MLP with max pooling summarizes the
    terrain; with the context it is encoded into one shared vector, which FiLM turns into one query per body
    (conditioned on learned body embeddings). One cross-attention layer gives per-head attention weights over the
    points; their head average is the affordance. The occupancy head reads each body's attended feature (plus its
    query, see ``attention_residual``) and the pooled terrain summary.

    Args (in addition to :class:`ContactPlanner`'s):
        d_model: Width of keys, values, queries and the shared vector.
        num_heads: Attention heads.
        point_embedding_size: Width of the per-point (x, y, z) embedding.
        cnn_channels: Channels of the height-map CNN (zero-padded, stride 1, so one feature per point).
        cnn_kernel_size: Its square kernel size.
        terrain_context_sizes: Pointwise MLP before max pooling.
        encoder_hidden_sizes: Hidden layers of the context(+pooled terrain) encoder.
        film_hidden_sizes: Hidden layers of the FiLM parameter generator.
        occupancy_hidden_sizes: Hidden layers of the occupancy head.
        terrain_nonlinearity: Activation of the point embedding and the CNN.
        nonlinearity: Activation of the context, FiLM and occupancy MLPs.
        attention_residual: Add each body's query to its attended feature before the occupancy head (no
            parameters). Without it the occupancy head sees the context (proprioception, goal) only through where
            the attention points; the paper does not say which it uses.
    """

    def __init__(
        self,
        input_segments: OrderedDict,
        points_xy_component: str,
        points_z_component: str,
        context_components: Sequence[str],
        num_bodies: int = 9,
        d_model: int = 96,
        num_heads: int = 32,
        point_embedding_size: int = 16,
        cnn_channels: Sequence[int] = (16, 48),
        cnn_kernel_size: int = 5,
        terrain_context_sizes: Sequence[int] = (64, 64),
        encoder_hidden_sizes: Sequence[int] = (128, 128),
        film_hidden_sizes: Sequence[int] = (64,),
        occupancy_hidden_sizes: Sequence[int] = (512, 512, 512),
        terrain_nonlinearity: str = "ELU",
        nonlinearity: str = "ReLU",
        attention_residual: bool = True,
        top_k: int = 1,
        temperature: float = 1.0,
        occupancy_threshold: float = 0.5,
        target_sigma: float = 0.1,
        affordance_loss_coef: float = 1.0,
        occupancy_loss_coef: float = 1.0,
        normalize_context: bool = True,
    ):
        super().__init__(
            input_segments,
            points_xy_component,
            points_z_component,
            context_components,
            num_bodies,
            top_k,
            temperature,
            occupancy_threshold,
            target_sigma,
            affordance_loss_coef,
            occupancy_loss_coef,
            normalize_context,
        )
        self.attention_residual = bool(attention_residual)
        if cnn_kernel_size % 2 != 1:
            raise ValueError("cnn_kernel_size must be odd so the CNN keeps one feature per point")
        self.point_embedding = MlpModel(3, [point_embedding_size], nonlinearity=terrain_nonlinearity)
        self.height_encoder = Conv2dModel(
            in_channels=1,
            channels=list(cnn_channels),
            kernel_sizes=[cnn_kernel_size] * len(cnn_channels),
            strides=[1] * len(cnn_channels),
            paddings=[cnn_kernel_size // 2] * len(cnn_channels),
            nonlinearity=getattr(nn, terrain_nonlinearity),
        )
        self.feature_proj = nn.Linear(point_embedding_size + cnn_channels[-1], d_model)
        self.terrain_context = MlpModel(d_model, list(terrain_context_sizes), nonlinearity=nonlinearity)
        self.encoder = MlpModel(
            self.context_size + terrain_context_sizes[-1],
            list(encoder_hidden_sizes),
            output_size=d_model,
            nonlinearity=nonlinearity,
        )
        self.body_embeddings = nn.Parameter(torch.randn(self.num_bodies, d_model))
        self.film = FiLM(d_model, d_model, film_hidden_sizes, nonlinearity=nonlinearity)
        self.attention = CrossAttention(d_model, d_model, d_model, num_heads)
        self.occupancy_head = MlpModel(
            d_model + terrain_context_sizes[-1], list(occupancy_hidden_sizes), output_size=1, nonlinearity=nonlinearity
        )

    def scores(self, points: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = points.shape[0]
        heights = points[..., 2].reshape(batch, 1, *self.grid_shape)
        cnn_features = self.height_encoder(heights).flatten(2).transpose(1, 2)  # (B, P, C)
        features = self.feature_proj(torch.cat([self.point_embedding(points), cnn_features], dim=-1))
        terrain_summary = self.terrain_context(features).amax(dim=1)  # (B, T)
        shared = self.encoder(torch.cat([context, terrain_summary], dim=-1))
        queries = self.film(shared.unsqueeze(1), self.body_embeddings)  # (B, Q, d_model)
        attended, log_weights = self.attention(queries, features)
        if self.attention_residual:
            attended = attended + queries
        log_affordance = torch.logsumexp(log_weights, dim=1) - math.log(self.attention.num_heads)
        summary = terrain_summary.unsqueeze(1).expand(-1, self.num_bodies, -1)
        occupancy_logits = self.occupancy_head(torch.cat([attended, summary], dim=-1)).squeeze(-1)
        return log_affordance, occupancy_logits


class MlpContactPlanner(ContactPlanner):
    """ACP's MLP baseline planner (Supp. Table S3; the defaults reproduce its 833,322 parameters with a 375-d
    proprioceptive history): a shared MLP over the context and the flattened points, a softmax head over the points
    per body and a linear occupancy head.

    Args (in addition to :class:`ContactPlanner`'s):
        hidden_sizes: Hidden layers of the shared backbone (activation after each).
        nonlinearity: Its activation.
    """

    def __init__(
        self,
        input_segments: OrderedDict,
        points_xy_component: str,
        points_z_component: str,
        context_components: Sequence[str],
        num_bodies: int = 9,
        hidden_sizes: Sequence[int] = (285, 285),
        nonlinearity: str = "ELU",
        top_k: int = 1,
        temperature: float = 1.0,
        occupancy_threshold: float = 0.5,
        target_sigma: float = 0.1,
        affordance_loss_coef: float = 1.0,
        occupancy_loss_coef: float = 1.0,
        normalize_context: bool = True,
    ):
        super().__init__(
            input_segments,
            points_xy_component,
            points_z_component,
            context_components,
            num_bodies,
            top_k,
            temperature,
            occupancy_threshold,
            target_sigma,
            affordance_loss_coef,
            occupancy_loss_coef,
            normalize_context,
        )
        self.backbone = MlpModel(self.context_size + 3 * self.num_points, list(hidden_sizes), nonlinearity=nonlinearity)
        self.affordance_head = nn.Linear(hidden_sizes[-1], self.num_bodies * self.num_points)
        self.occupancy_head = nn.Linear(hidden_sizes[-1], self.num_bodies)

    def scores(self, points: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = points.shape[0]
        hidden = self.backbone(torch.cat([context, points.reshape(batch, -1)], dim=-1))
        logits = self.affordance_head(hidden).reshape(batch, self.num_bodies, self.num_points)
        return torch.log_softmax(logits, dim=-1), self.occupancy_head(hidden)
