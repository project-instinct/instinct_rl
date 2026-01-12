from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MlpModel


class TransformerHeadModel(torch.nn.Module):
    """Sequential Transformer model with final heads to output corresponding values.

    NOTE: When exporting to ONNX, must disable memory efficient sdp by `enable_mem_efficient_sdp(False)`
    """

    def __init__(
        self,
        input_shapes: Sequence[torch.Size],
        output_size: int,
        num_heads: int,
        num_layers: int = 1,  # form transformer encoder from multiple layers.
        d_model: int = 256,  # latent size of the transformer encoder.
        dim_feedforward: int = 512,  # transformer default is 2048, which is too big.
        dropout: float = 0.1,
        activation: str = "relu",
        nonlinearity: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        mask_from_input_dim: int = -1,
        output_selection: Literal["maxpool", "smallest_positive", "smallest_nonnegative"] = "maxpool",
        input_hidden_sizes: Sequence[
            int
        ] = [],  # if None, no input mlp is added, feature_size is the sum of input features.
        output_hidden_sizes: Sequence[int] = [],  # if None, no output mlp is added.
    ):
        """
        Args:
            activation: The activation function for the transformer encoder.
            nonlinearity: The nonlinearity layer for the mlp network.
            mask_from_input_dim: The dimension to get the self-attention mask from the input tensor.
                If -1, no mask is used.
                Based on transformer docs, the input to the transformer encoder will be in shape (N, S, S), where
                    m[:, i, j] = and(masks[i], masks[j])
            output_selection: Since transformer encoder outputs is still a sequential tensor,
                we need to select a single tensor to be used as the final batchwise output.
                Use "maxpool" to select the maximum value from the sequence.
                Use "smallest_positive" to select the embedding with the smallest positive value from the mask_from_input_dim.
                Use "smallest_nonnegative" to select the embedding with the smallest non-negative value from the mask_from_input_dim.
        """
        super().__init__()
        # store all parameters
        self.input_shapes = input_shapes
        self.output_size = output_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.nonlinearity = nonlinearity
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.mask_from_input_dim = mask_from_input_dim
        self.output_selection = output_selection
        self.input_hidden_sizes = (
            [] if input_hidden_sizes is None else input_hidden_sizes
        )  # legacy code for None input_hidden_sizes
        self.output_hidden_sizes = [] if output_hidden_sizes is None else output_hidden_sizes

        self._parse_input_shape()

        self._build_modules()

    def forward(self, x, mask=None):
        leading_dim = x.shape[:-2]
        input_seq = x
        # input layer (batch, seq, input_size) -> (batch, seq, d_model)
        if len(x.shape) > 3:
            # NOTE: Due to ONNX tracing, we does skip the reshaping of the input tensor in
            # testing rollouts.
            x = input_seq = x.reshape((np.prod(leading_dim), x.shape[-2], self._input_feature_size))
        if self.mask_from_input_dim >= 0:
            x_mask = x[..., self.mask_from_input_dim] < 0
            mask = x_mask.unsqueeze(-2) * x_mask.unsqueeze(-1)
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).flatten(0, 1).to(torch.float)
        x = self.input_layer(x)

        # transformer encoder (batch, seq, d_model)
        x = self.tf_encoder(x, mask=mask)

        # output selection (batch, seq, d_model) -> (batch, d_model)
        if self.output_selection == "maxpool":
            x = x.amax(dim=1)
        elif self.output_selection == "smallest_positive":
            mask_from = input_seq[..., self.mask_from_input_dim]  # (batch, seq)
            torch_inf = torch.ones_like(mask_from) * float("inf")
            mask_from = torch.where(mask_from <= 0, torch_inf, mask_from)
            min_idx = torch.argmin(mask_from, dim=1).detach()
            x = x[torch.arange(x.shape[0]), min_idx]
        elif self.output_selection == "smallest_nonnegative":
            mask_from = input_seq[..., self.mask_from_input_dim]  # (batch, seq)
            torch_inf = torch.ones_like(mask_from) * float("inf")
            mask_from = torch.where(mask_from < 0, torch_inf, mask_from)
            min_idx = torch.argmin(mask_from, dim=1).detach()
            x = x[torch.arange(x.shape[0]), min_idx]
        # output layer (batch, d_model) -> (batch, output_size)
        x = self.output_layer(x)

        return x.reshape(leading_dim + (self.output_size,))

    """
    Helper functions.
    """

    def _parse_input_shape(self) -> None:
        """Parse the input shapes to get the feature size."""
        # NOTE: assuming all input shapes are (N, L, d), where N is the batch size, L is the sequence length, d is the feature size.
        self._input_feature_size = sum([np.prod(s[1:]) for s in self.input_shapes])

    def _build_modules(self) -> None:
        # check and build input layers
        self.input_layer = MlpModel(
            input_size=self._input_feature_size,
            hidden_sizes=self.input_hidden_sizes + [self.d_model],
            nonlinearity=self.nonlinearity,
        )

        # check and build transformer encoder layers
        tf_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=self.batch_first,
            norm_first=self.norm_first,
        )
        self.tf_encoder = nn.TransformerEncoder(
            encoder_layer=tf_encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model, eps=self.layer_norm_eps),
        )

        # check and build transformer encoder
        self.output_layer = MlpModel(
            input_size=self.d_model,
            hidden_sizes=self.output_hidden_sizes + [self.output_size],
            nonlinearity=self.nonlinearity,
        )
