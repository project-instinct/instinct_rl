import torch
import torch.distributions as dist
import torch.nn as nn

from .mlp import MlpModel


class MlpVae(nn.Module):
    """A simple MLP-based VAE.
    Args:
        input_size: int, the input size of the encoder.
        hidden_sizes: list[int], the hidden sizes of the encoder.
        codebook_size: int, the size of the codebook.
        codebook_dim: int, the dimension of the codebook.
        commitment_cost: float, the commitment cost of the VAE.
        nonlinearity: str, the nonlinearity of the encoder.
    """

    def __init__(
        self, encoder_kwargs: dict(), decoder_kwargs: dict(), latent_size: int, decoder_aux_input_size: int = 0
    ):
        super().__init__()
        encoder_kwargs["output_size"] = latent_size * 2
        decoder_kwargs["input_size"] = latent_size + decoder_aux_input_size
        self.encoder = MlpModel(**encoder_kwargs)
        self.decoder = MlpModel(**decoder_kwargs)
        self.latent_size = latent_size

    def forward(self, x, decoder_aux_input: torch.Tensor = None) -> tuple[torch.Tensor, dist.Distribution]:
        z_mean, z_log_std = self.encoder(x).chunk(2, dim=-1)
        z = z_mean + z_log_std.exp() * torch.randn_like(z_mean)
        if decoder_aux_input is not None:
            z = torch.cat([z, decoder_aux_input], dim=-1)
        return self.decoder(z), dist.Normal(z_mean, z_log_std.exp())
