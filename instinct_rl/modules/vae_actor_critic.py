"""This is a implementation of the VAE actor, with full actor_critic interface,
"""

import os
from typing import Dict

import torch
import torch.distributions as dist
import torch.nn as nn

from instinct_rl.utils.utils import get_subobs_indexing_by_components, get_subobs_size

from .actor_critic import ActorCritic
from .vae import MlpVae


class VaeActorCritic(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions,
        vae_encoder_kwargs: dict(),
        vae_decoder_kwargs: dict(),
        vae_latent_size: int,
        vae_input_subobs_components: list[str] | None = None,
        vae_aux_subobs_components: list[str] | None = None,
        num_rewards=1,
        **kwargs,
    ):
        """
        Args:
            vae_subobs_components: list[str], the components of the observation to be encoded by the VAE encoder.
                If None, all components will be encoded.
                If provided, the rest of the components will be passed toggether as a single input to the VAE decoder.
        """
        if kwargs:
            print(
                "VaeActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        self.__obs_format = obs_format
        self.__obs_segments = obs_format["policy"]
        print(f"VaeActorCritic: obs segments: {self.__obs_segments}")
        self.__critic_obs_segments = obs_format.get("critic", obs_format["policy"])
        self._vae_input_subobs_components = vae_input_subobs_components
        self._vae_aux_subobs_components = vae_aux_subobs_components
        self._vae_encoder_kwargs = vae_encoder_kwargs
        self._vae_decoder_kwargs = vae_decoder_kwargs
        self._vae_latent_size = vae_latent_size
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        # parse vae arguments
        if self._vae_aux_subobs_components is None:
            self._vae_aux_subobs_components = set(self.__obs_segments.keys()) - set(self._vae_input_subobs_components)
            if len(self._vae_aux_subobs_components) > 0:
                self._vae_aux_subobs_components = list(self._vae_aux_subobs_components)
                print(
                    "VaeActorCritic: aux subobs components inferred from input subobs components:"
                    f" {self._vae_aux_subobs_components}"
                )
        self._vae_input_dim = get_subobs_size(self.__obs_segments, component_names=self._vae_input_subobs_components)
        self._vae_encoder_kwargs["input_size"] = self._vae_input_dim

        # Note: Only actor is a VAE, the critic is still the same as the original ActorCritic.
        super().__init__(obs_format, num_actions, **kwargs)
        # Pre-compute indexing for optimized sub-observation extraction
        if self._vae_input_subobs_components is not None:
            self.register_buffer(
                "_vae_input_indices",
                get_subobs_indexing_by_components(self.__obs_segments, self._vae_input_subobs_components),
            )
        else:
            self._vae_input_indices = None

        if self._vae_aux_subobs_components is not None:
            self.register_buffer(
                "_vae_aux_indices",
                get_subobs_indexing_by_components(self.__obs_segments, self._vae_aux_subobs_components),
            )
        else:
            self._vae_aux_indices = None

    def _build_actor(self, num_actions) -> MlpVae:
        self._vae_decoder_kwargs["output_size"] = num_actions
        if self._vae_aux_subobs_components is not None:
            decoder_aux_input_size = get_subobs_size(
                self.__obs_segments, component_names=self._vae_aux_subobs_components
            )
        else:
            decoder_aux_input_size = 0
        return MlpVae(
            encoder_kwargs=self._vae_encoder_kwargs,
            decoder_kwargs=self._vae_decoder_kwargs,
            latent_size=self._vae_latent_size,
            decoder_aux_input_size=decoder_aux_input_size,
        )

    def _run_actor(self, observations):
        if self._vae_input_subobs_components is not None:
            # Use pre-computed indices with torch.gather for optimized extraction
            batch_indices = self._vae_input_indices.unsqueeze(0).expand(observations.shape[0], -1)
            vae_input = torch.gather(observations, 1, batch_indices)

            batch_aux_indices = self._vae_aux_indices.unsqueeze(0).expand(observations.shape[0], -1)
            decoder_aux_input = torch.gather(observations, 1, batch_aux_indices)

            decoded, distribution = self.actor(vae_input, decoder_aux_input=decoder_aux_input)
        else:
            decoded, distribution = self.actor(observations)
        return decoded, distribution

    def update_distribution(self, observations):
        decoded, distribution = self._run_actor(observations)
        self.latent_distribution = distribution
        self.distribution = dist.Normal(decoded, decoded * 0.0 + self.std)

    def act_inference(self, observations):
        decoded, _ = self._run_actor(observations)
        return decoded

    @property
    def obs_segments(self):
        return self.__obs_segments

    @property
    def critic_obs_segments(self):
        return self.__critic_obs_segments

    def export_as_onnx(self, observations, filedir):
        """Export the model as an ONNX file. Input should be batch-wise observations with batchsize 1."""
        self.eval()
        with torch.no_grad():
            onnx_vae_network = OnnxVaeNetwork(self)
            exported_program = torch.onnx.export(
                onnx_vae_network,
                observations,
                os.path.join(filedir, "vae_actor.onnx"),
                input_names=["input"],
                output_names=["output", "latent_mean", "latent_std"],
                opset_version=12,
            )
            print(f"Exported VaeActorCritic model to {os.path.join(filedir, 'vae_actor.onnx')}")


class OnnxVaeNetwork(nn.Module):
    """ONNX exportable VAE network that performs static observation indexing.

    This module takes full observations as input, performs static indexing to extract
    VAE input and auxiliary components, and outputs the decoded action and latent distribution.
    The observation indexing logic is baked into the network architecture for ONNX compatibility.
    """

    def __init__(self, vae_actor_critic: VaeActorCritic):
        """Initialize from an existing VaeActorCritic instance.

        Args:
            vae_actor_critic: The VaeActorCritic instance to extract components from
        """
        super().__init__()

        # Copy the VAE actor (this contains the encoder and decoder)
        self.vae = vae_actor_critic.actor

        # Copy pre-computed indices from VaeActorCritic for compact indexing
        if hasattr(vae_actor_critic, "_vae_input_indices") and vae_actor_critic._vae_input_indices is not None:
            self.register_buffer("vae_input_indices", vae_actor_critic._vae_input_indices.clone())
        else:
            self.vae_input_indices = None

        if hasattr(vae_actor_critic, "_vae_aux_indices") and vae_actor_critic._vae_aux_indices is not None:
            self.register_buffer("vae_aux_indices", vae_actor_critic._vae_aux_indices.clone())
        else:
            self.vae_aux_indices = None

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the ONNX-exportable VAE network.

        Args:
            observations: Full observation tensor of shape (batch_size, obs_size)

        Returns:
            Tuple of (decoded_action, latent_mean, latent_std)
        """
        if self.vae_input_indices is not None and self.vae_aux_indices is not None:
            # Extract VAE input and auxiliary components using pre-computed indices
            batch_input_indices = self.vae_input_indices.unsqueeze(0).expand(observations.shape[0], -1)
            vae_input = torch.gather(observations, 1, batch_input_indices)

            batch_aux_indices = self.vae_aux_indices.unsqueeze(0).expand(observations.shape[0], -1)
            vae_aux_input = torch.gather(observations, 1, batch_aux_indices)

            # Forward through VAE
            decoded_action, latent_dist = self.vae(vae_input, decoder_aux_input=vae_aux_input)

        elif self.vae_input_indices is None:
            # Use full observations as VAE input (no auxiliary input)
            decoded_action, latent_dist = self.vae(observations)

        else:
            # Only VAE input components, no auxiliary
            batch_input_indices = self.vae_input_indices.unsqueeze(0).expand(observations.shape[0], -1)
            vae_input = torch.gather(observations, 1, batch_input_indices)
            decoded_action, latent_dist = self.vae(vae_input)

        # Return decoded action, latent mean, and latent std for ONNX compatibility
        return decoded_action, latent_dist.mean, latent_dist.stddev
