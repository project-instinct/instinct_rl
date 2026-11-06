from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from instinct_rl.modules.mlp import MlpModel


class SingleQueryAttention(nn.Module):
    """Multi-head attention of one query over ``N`` tokens, as ``nn.MultiheadAttention`` computes it, but without
    projecting every token: the key projection is folded into the query (``q_h . (W_h t + b_h) = (W_h^T q_h) . t +
    q_h . b_h``) and the value projection is applied after the attention-weighted token sum, so the cost per token
    is one ``num_heads x dim`` product instead of two ``dim x dim`` projections."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """``query`` ``(B, dim)``, ``tokens`` ``(B, N, dim)`` -> ``(B, dim)``."""
        batch, dim = query.shape
        heads, head_dim = self.num_heads, self.head_dim
        q = self.q_proj(query).view(batch, heads, head_dim) / head_dim**0.5
        k_weight = self.k_proj.weight.view(heads, head_dim, dim)
        q_tokens = torch.einsum("bhd,hdc->bhc", q, k_weight)  # (B, heads, dim)
        q_bias = torch.einsum("bhd,hd->bh", q, self.k_proj.bias.view(heads, head_dim))
        scores = torch.einsum("bhc,bnc->bhn", q_tokens, tokens) + q_bias.unsqueeze(-1)
        weighted = torch.einsum("bhn,bnc->bhc", scores.softmax(dim=-1), tokens)  # (B, heads, dim)
        v_weight = self.v_proj.weight.view(heads, head_dim, dim)
        values = torch.einsum("bhc,hdc->bhd", weighted, v_weight) + self.v_proj.bias.view(heads, head_dim)
        return self.out_proj(values.reshape(batch, dim))


class AME2ImageEncoder(nn.Module):
    """AME-2 attention-based map encoder (Zhang et al., "Agile and Generalized Legged Locomotion via Attention-Based
    Neural Map Encoding", Sec. IV-A) on an image instead of a height map.

    Every pixel is a token: a CNN that keeps the image size gives its local features and an MLP on its normalized
    pixel coordinates and values gives its positional features, fused into ``local_dim`` features. An MLP and a max
    pool over the tokens give the global features. The remaining input components (proprioception, commands) are
    embedded by an MLP; with the global features, they form the query of a multi-head attention over the tokens.

    The input is the concatenation of ``component_names`` in the order of ``input_segments``, as the parallel layer
    passes it. The output is ``[attention output, proprioception embedding, global features]``.
    """

    def __init__(
        self,
        input_segments: Dict[str, tuple],
        component_names: List[str],
        image_component: str,
        num_heads: int = 32,
        local_dim: int = 96,
        global_dim: int = 64,
        prop_dim: int = 128,
        hidden_size: int = 512,
        nonlinearity: str = "ELU",
    ):
        super().__init__()
        if image_component not in component_names:
            raise ValueError(f"image_component {image_component!r} is not one of {component_names}")
        activation = getattr(nn, nonlinearity)
        self.image_shape = tuple(input_segments[image_component])
        channels, height, width = self.image_shape
        # slices of the image and the other components in the concatenated input
        start = 0
        prop_slices = []
        for name, shape in input_segments.items():
            if name not in component_names:
                continue
            size = int(np.prod(shape))
            if name == image_component:
                self.image_slice = slice(start, start + size)
            else:
                prop_slices.append(slice(start, start + size))
            start += size
        if not prop_slices:
            raise ValueError("AME2ImageEncoder needs at least one non-image component for the attention query")
        self.prop_slices = prop_slices
        prop_input_size = start - int(np.prod(self.image_shape))

        v, u = torch.meshgrid(torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width), indexing="ij")
        self.register_buffer("pixel_coords", torch.stack([u, v], dim=-1).reshape(1, height * width, 2))

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=5, padding=2),
            activation(),
            nn.Conv2d(8, 48, kernel_size=5, padding=2),
            activation(),
        )
        self.positional = nn.Sequential(nn.Linear(2 + channels, 16), activation())
        self.local_proj = nn.Sequential(nn.Linear(16 + 48, local_dim), activation())
        self.global_mlp = MlpModel(local_dim, [global_dim], output_size=global_dim, nonlinearity=activation)
        self.prop_encoder = MlpModel(
            prop_input_size, [hidden_size, hidden_size], output_size=prop_dim, nonlinearity=activation
        )
        self.query_mlp = MlpModel(
            prop_dim + global_dim, [hidden_size, hidden_size], output_size=local_dim, nonlinearity=activation
        )
        self.attention = SingleQueryAttention(local_dim, num_heads)
        self._output_size = local_dim + prop_dim + global_dim

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, flat_input: torch.Tensor) -> torch.Tensor:
        leading_shape = flat_input.shape[:-1]
        flat_input = flat_input.reshape(-1, flat_input.shape[-1])
        batch = flat_input.shape[0]
        image = flat_input[:, self.image_slice].reshape(batch, *self.image_shape)
        prop = torch.cat([flat_input[:, s] for s in self.prop_slices], dim=-1)

        pixels = image.flatten(2).transpose(1, 2)  # (B, H * W, C)
        positional = self.positional(torch.cat([self.pixel_coords.expand(batch, -1, -1), pixels], dim=-1))
        cnn_features = self.cnn(image).flatten(2).transpose(1, 2)  # (B, H * W, 48)
        local = self.local_proj(torch.cat([positional, cnn_features], dim=-1))  # (B, H * W, local_dim)

        global_features = self.global_mlp(local).max(dim=1)[0]
        prop_embedding = self.prop_encoder(prop)
        query = self.query_mlp(torch.cat([prop_embedding, global_features], dim=-1))
        attended = self.attention(query, local)
        output = torch.cat([attended, prop_embedding, global_features], dim=-1)
        return output.reshape(*leading_shape, self._output_size)
