"""This is a implementation of the VAE actor, with full actor_critic interface,
but no actor distribution and no critic network. It outputs the action
"""

import os
from typing import Dict

import torch
import torch.nn as nn

from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size

from .actor_critic import ActorCritic
from .vae import MlpVae


class VaeActor(ActorCritic):
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
                "VaeActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)  # not using super() to avoid calling ActorCritic.__init__
        self.__obs_format = obs_format
        self.__obs_segments = obs_format["policy"]
        print(f"VaeActor: obs segments: {self.__obs_segments}")
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
                    "VaeActor: aux subobs components inferred from input subobs components:"
                    f" {self._vae_aux_subobs_components}"
                )
        self._vae_input_dim = get_subobs_size(self.__obs_segments, component_names=self._vae_input_subobs_components)
        self._vae_encoder_kwargs["input_size"] = self._vae_input_dim
        self._vae_decoder_kwargs["output_size"] = self.num_actions

        self.actor = self._build_actor()
        self.std = torch.ones(self.num_actions)

    def _build_actor(self) -> MlpVae:
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

    # def _build_critic(self, num_values=1):
    #     return None

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def get_actions_log_prob(self, actions):
        return torch.ones(
            actions.shape[0],
            device=actions.device,
            dtype=actions.dtype,
        )

    @property
    def entropy(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self._action_mean

    @property
    def action_std(self):
        return self.std.unsqueeze(0).expand(self.action_mean.shape[0], -1)

    def update_distribution(self, observations):
        if self._vae_input_subobs_components is not None:
            vae_input = get_subobs_by_components(observations, self._vae_input_subobs_components, self.__obs_segments)
            vae_input = vae_input.reshape(observations.shape[0], -1)
            decoder_aux_input = get_subobs_by_components(
                observations, self._vae_aux_subobs_components, self.__obs_segments
            )
            decoder_aux_input = decoder_aux_input.reshape(observations.shape[0], -1)
            decoded, distribution = self.actor(vae_input, decoder_aux_input=decoder_aux_input)
        else:
            decoded, distribution = self.actor(observations)
        self.distribution = distribution
        self._action_mean = decoded

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.action_mean

    def act_inference(self, observations):
        self.update_distribution(observations)
        return self.action_mean

    def evaluate(self, critic_observations, **kwargs):
        return torch.zeros(
            critic_observations.shape[0],
            self.num_rewards,
            device=critic_observations.device,
            dtype=critic_observations.dtype,
        )

    @property
    def obs_segments(self):
        return self.__obs_segments

    @property
    def critic_obs_segments(self):
        return self.__critic_obs_segments
