import importlib
import os
from collections import OrderedDict
from copy import copy, deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

import instinct_rl.modules as instinct_modules
from instinct_rl.utils.utils import get_obs_slice, get_subobs_size


class EncoderActorCriticMixin:
    """A general implementation where a separate encoder is used to embed the obs/privileged_obs"""

    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions,
        encoder_configs: Dict[str, dict],
        critic_encoder_configs=None,  # None, "shared", or an ordereddict of configs (in the same order as encoder_configs)
        reference_residual: dict | None = None,
        **kwargs,
    ):
        """NOTE: recurrent encoder is not implemented and tested yet.
        Args:
            reference_residual: Make the actor output a correction on top of a reference read from the
                policy observation (InstantTrack's reference-residual policy): ``{"component": name,
                "frame": k, "scale": [num_actions floats]}`` adds ``scale * obs[name][k]`` to the action,
                with ``obs[name]`` a ``(frames, num_actions)`` component taken in raw (un-normalised) units.
                For a joint-position action ``q_default + alpha * a`` and a component holding
                ``q_ref - q_default``, ``scale = 1 / alpha`` makes the untrained policy command the
                reference. Only the deterministic action (``act_inference``) supports it.
        """
        self.__obs_segments = obs_format["policy"]
        self.__critic_obs_segments = obs_format.get("critic", obs_format["policy"])
        self.num_actor_obs = get_subobs_size(self.__obs_segments)
        self.num_critic_obs = get_subobs_size(self.__critic_obs_segments)
        self.num_actions = num_actions

        self.encoder_configs = copy(encoder_configs)
        self.critic_encoder_configs = copy(critic_encoder_configs)
        encoder_class_name = encoder_configs.pop("class_name", "ParallelLayer")
        EncoderClass = (
            getattr(importlib.import_module(encoder_class_name.split(":")[0]), encoder_class_name.split(":")[1])
            if ":" in encoder_class_name
            else getattr(instinct_modules, encoder_class_name)
        )
        encoders = EncoderClass(
            input_segments=self.__obs_segments,
            block_configs=self.encoder_configs,
        )
        if self.critic_encoder_configs == "shared":
            critic_encoders = encoders
        elif self.critic_encoder_configs is None:
            critic_encoders = None
        else:
            critic_encoder_class_name = critic_encoder_configs.pop("class_name", "ParallelLayer")
            CriticEncoderClass = (
                getattr(
                    importlib.import_module(critic_encoder_class_name.split(":")[0]),
                    critic_encoder_class_name.split(":")[1],
                )
                if ":" in critic_encoder_class_name
                else getattr(instinct_modules, critic_encoder_class_name)
            )
            critic_encoders = CriticEncoderClass(
                input_segments=self.__critic_obs_segments,
                block_configs=self.critic_encoder_configs,
            )

        embedded_obs_format = deepcopy(obs_format)
        embedded_obs_format["policy"] = encoders.output_segment
        if critic_encoders is not None:
            embedded_obs_format["critic"] = critic_encoders.output_segment

        super().__init__(
            embedded_obs_format,
            num_actions,
            **kwargs,
        )
        self.encoders = encoders
        self.critic_encoders = critic_encoders

        print(f"Actor Encoder: {self.encoders}")
        print(f"Critic Encoder: {self.critic_encoders}")

        self.encoder_latents_buf = dict()
        self.critic_encoder_latents_buf = dict()

        self.reference_residual = reference_residual
        if reference_residual is not None:
            component_slice, shape = get_obs_slice(self.__obs_segments, reference_residual["component"])
            if len(shape) != 2 or shape[-1] != num_actions:
                raise ValueError(
                    f"reference_residual component {reference_residual['component']!r} has shape {shape}, expected"
                    f" (frames, {num_actions})"
                )
            frame = int(reference_residual["frame"])
            if not 0 <= frame < shape[0]:
                raise ValueError(f"reference_residual frame {frame} is outside the component's {shape[0]} frames")
            start = int(component_slice.start) + frame * num_actions
            self._reference_slice = slice(start, start + num_actions)
            scale = torch.as_tensor(reference_residual["scale"], dtype=torch.float32)
            if scale.shape != (num_actions,):
                raise ValueError(f"reference_residual scale has shape {tuple(scale.shape)}, expected ({num_actions},)")
            self.register_buffer("reference_residual_scale", scale)
        # The policy-group normaliser the observations arrive scaled by (see ``set_policy_obs_normalizer``).
        self.policy_obs_normalizer = None

    def set_policy_obs_normalizer(self, normalizer):
        """Name the normaliser the actor's observations were scaled with, so ``reference_residual`` can read
        its component in raw units. Held by reference and not registered as a submodule: the runner saves
        its state, and it may be moved to another device after this call."""
        object.__setattr__(self, "policy_obs_normalizer", normalizer)

    def reference_action(self, observations):
        """``scale * obs[component][frame]`` of ``reference_residual``, in raw observation units."""
        reference = observations[..., self._reference_slice]
        normalizer = self.policy_obs_normalizer
        if normalizer is not None:
            mean = normalizer._mean[..., self._reference_slice]
            std = normalizer._std[..., self._reference_slice]
            reference = reference * (std + normalizer.eps) + mean
        return reference * self.reference_residual_scale

    def backbone_act(self, flatten_observations, **kwargs):
        super().act(flatten_observations, **kwargs)

    def act(self, observations, **kwargs):
        if self.reference_residual is not None:
            raise NotImplementedError("reference_residual shifts the deterministic action only; act() samples")
        obs = self.encoders(observations)
        return super().act(obs, **kwargs)

    def act_inference(self, observations):
        obs = self.encoders(observations)
        actions = super().act_inference(obs)
        if self.reference_residual is not None:
            actions = actions + self.reference_action(observations)
        return actions

    def backbone_evaluate(self, flatten_observations, masks=None, hidden_states=None):
        """Evaluate the model with the backbone input, which is typically an MLP."""
        return super().evaluate(flatten_observations, masks=masks, hidden_states=hidden_states)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        obs = self.critic_encoders(critic_observations) if self.critic_encoders is not None else critic_observations
        return super().evaluate(obs, masks=masks, hidden_states=hidden_states)

    def forward(self, observations):
        """Fill the function of a torch.nn.Module and make it compatible with the torch.onnx.export"""
        return self.act_inference(observations)

    @property
    def obs_segments(self):
        """NOTE: make sure to copy these methods in subclasses if the subclass has its own obs_segments"""
        return self.__obs_segments

    @property
    def critic_obs_segments(self):
        return self.__critic_obs_segments

    def export_as_onnx(self, observations, filedir, encoder_as_seperate_file=True):
        """Export the model as an ONNX file. Input should be batch-wise observations with batchsize 1."""
        self.eval()
        if encoder_as_seperate_file:
            if self.reference_residual is not None:
                raise NotImplementedError(
                    "reference_residual adds to the action outside the encoder and the backbone; export with"
                    " encoder_as_seperate_file=False"
                )
            self.encoders.export_as_onnx(observations, filedir, encoder_as_seperate_file)
            with torch.no_grad():
                obs = self.encoders(observations)
            super().export_as_onnx(obs, filedir)
        else:
            with torch.no_grad():
                exported_program = torch.onnx.export(
                    self,
                    observations,
                    "/tmp/encoder_actor_critic.onnx",
                    # This file does not contain the model weight, we call the save later to save the onnx with model weight.
                    input_names=["input"],
                    output_names=["output"],
                    dynamo=True,
                    opset_version=18,
                )
                exported_program.save(os.path.join(filedir, "encoder_actor_critic.onnx"))
                print(f"Exported encoder_actor_critic to {os.path.join(filedir, 'encoder_actor_critic.onnx')}")


from .actor_critic import ActorCritic


class EncoderActorCritic(EncoderActorCriticMixin, ActorCritic):
    pass


from .actor_critic_recurrent import ActorCriticRecurrent


class EncoderActorCriticRecurrent(EncoderActorCriticMixin, ActorCriticRecurrent):
    pass
