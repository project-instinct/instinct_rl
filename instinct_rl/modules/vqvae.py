import torch
import torch.nn as nn
import torch.nn.functional as F


class VqVae(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
        nonlinearity: str = "ReLU",
    ):
        super().__init__()
        from .mlp import MlpModel

        self.encoder = MlpModel(input_size, hidden_sizes, codebook_dim, nonlinearity=nonlinearity)

        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        self.decoder = MlpModel(codebook_dim, list(reversed(hidden_sizes)), input_size, nonlinearity=nonlinearity)

        self.commitment_cost = commitment_cost

    def forward(self, x):
        z = self.encoder(x)

        distances = (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z, self.codebook.weight.t())
        )

        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(indices.shape[0], self.codebook.num_embeddings, device=z.device)
        encodings.scatter_(1, indices, 1)

        quantized = torch.matmul(encodings, self.codebook.weight).view(z.shape)

        if self.training:
            e_latent_loss = F.mse_loss(quantized.detach(), z)
            q_latent_loss = F.mse_loss(quantized, z.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        else:
            loss = torch.tensor(0.0, device=z.device)

        quantized = z + (quantized - z).detach()

        reconstructed = self.decoder(quantized)

        return reconstructed, quantized, loss, indices
