"""Attention building blocks: feature-wise linear modulation and a cross-attention head.

Both take two inputs (features and a condition, or queries and keys/values), so they are used inside modules
such as the contact planner rather than as ``ParallelLayer`` blocks, which take one flat tensor.

The attention is written out instead of using ``nn.MultiheadAttention``: callers need the per-head attention
log-weights (the contact planner trains them with a cross-entropy), and plain matmuls export to ONNX without
the scaled-dot-product kernel switches ``TransformerHeadModel`` needs. Parameter layout and count match
``nn.MultiheadAttention`` with biases (query, key, value and output projections).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from .mlp import MlpModel


class FiLM(nn.Module):
    """Feature-wise linear modulation: ``x * (1 + scale(cond)) + shift(cond)``.

    The ``1 +`` keeps the modulation near the identity at initialization.

    Args:
        cond_size: Size of the condition vector.
        feature_size: Size of the modulated features.
        hidden_sizes: Hidden layers of the MLP that maps the condition to ``(scale, shift)``.
        nonlinearity: Activation of that MLP.
    """

    def __init__(
        self,
        cond_size: int,
        feature_size: int,
        hidden_sizes: Sequence[int] = (),
        nonlinearity: str = "ReLU",
    ):
        super().__init__()
        self.feature_size = int(feature_size)
        self.generator = MlpModel(
            cond_size, list(hidden_sizes), output_size=2 * self.feature_size, nonlinearity=nonlinearity
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``x`` (..., feature_size) and ``cond`` (..., cond_size) broadcast against each other."""
        scale, shift = self.generator(cond).split(self.feature_size, dim=-1)
        return x * (1.0 + scale) + shift


class CrossAttention(nn.Module):
    """One layer of multi-head cross-attention from queries to a set of keys/values.

    No feed-forward block, normalization or dropout (ACP's planner, Supp. Table S3, uses exactly this).

    Args:
        query_size: Size of each query.
        kv_size: Size of each key/value element.
        d_model: Attention width; must be divisible by ``num_heads``.
        num_heads: Number of heads.
    """

    def __init__(self, query_size: int, kv_size: int, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_size = self.d_model // self.num_heads
        self.query_proj = nn.Linear(query_size, d_model)
        self.key_proj = nn.Linear(kv_size, d_model)
        self.value_proj = nn.Linear(kv_size, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, d_model) -> (B, H, S, head_size)
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_size).transpose(1, 2)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: (B, Q, query_size)
            keys_values: (B, P, kv_size)

        Returns:
            The attended features (B, Q, d_model) and the per-head attention log-weights (B, H, Q, P), which are
            normalized over P (``logsumexp`` over P is 0).
        """
        q = self._split_heads(self.query_proj(queries))
        k = self._split_heads(self.key_proj(keys_values))
        v = self._split_heads(self.value_proj(keys_values))
        log_weights = torch.log_softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_size), dim=-1)
        attended = torch.exp(log_weights) @ v  # (B, H, Q, head_size)
        attended = attended.transpose(1, 2).reshape(queries.shape[0], queries.shape[1], self.d_model)
        return self.out_proj(attended), log_weights
