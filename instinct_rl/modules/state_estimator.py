import numpy as np
import torch
import torch.nn as nn

from instinct_rl.modules.actor_critic_recurrent import Memory
from instinct_rl.modules.mlp import MlpModel
from instinct_rl.utils import unpad_trajectories
from instinct_rl.utils.collections import namedarraytuple
from instinct_rl.utils.utils import (
    get_subobs_by_components,
    get_subobs_size,
    replace_obs_components,
)

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticHiddenState, ActorCriticRecurrent

EstimatorActorHiddenState = namedarraytuple(
    "EstimatorActorHiddenState",
    [
        "estimator",
        "actor",
    ],
)


class EstimatorMixin:
    def __init__(
        self,
        *args,
        obs_format: dict[str, dict[str, tuple]],
        estimator_obs_components: list[str] = None,  # a list of strings used to get obs slices
        estimator_target_components: list[str] = None,  # a list of strings used to get obs slices
        estimator_configs=dict(),
        replace_state_prob=0.0,  # if 0~1, replace the actor observation with the estimated state with this probability
        **kwargs,
    ):
        super().__init__(
            *args,
            obs_format=obs_format,
            **kwargs,
        )
        self.__obs_segments = obs_format["policy"]
        self.estimator_obs_components = estimator_obs_components
        self.estimator_target_components = estimator_target_components
        self.estimator_configs = estimator_configs
        self.replace_state_prob = replace_state_prob
        self.build_estimator(**kwargs)

    def build_estimator(self, **kwargs):
        """This implementation is not flexible enough, but it is enough for now."""
        estimator_input_size = get_subobs_size(self.obs_segments, self.estimator_obs_components)
        estimator_output_size = get_subobs_size(self.obs_segments, self.estimator_target_components)
        if self.is_recurrent:
            # estimate required state using a recurrent network
            self.memory_s = Memory(
                estimator_input_size,
                type=kwargs.get("rnn_type", "lstm"),
                num_layers=kwargs.get("rnn_num_layers", 1),
                hidden_size=kwargs.get("rnn_hidden_size", 256),
            )
            self.state_estimator = MlpModel(
                input_size=self.memory_s.rnn.hidden_size,
                output_size=estimator_output_size,
                **self.estimator_configs,
            )
        else:
            # estimate required state using a feedforward network
            self.state_estimator = MlpModel(
                input_size=estimator_input_size,
                output_size=estimator_output_size,
                **self.estimator_configs,
            )

    def reset(self, dones=None):
        super().reset(dones)
        if self.is_recurrent:
            self.memory_s.reset(dones)

    def act(self, observations, masks=None, hidden_states=None, inference=False):
        observations = observations.clone()
        if inference:
            assert masks is None and hidden_states is None, "Inference mode does not support masks and hidden_states. "
        if self.is_recurrent:
            # TODO: allows non-recurrent state estimator with recurrent actor
            subobs = get_subobs_by_components(
                observations,
                component_names=self.estimator_obs_components,
                obs_segments=self.obs_segments,
            )
            input_s = self.memory_s(
                subobs,
                None,
                hidden_states if hidden_states is None else hidden_states.estimator,  # use None to prevent unpadding
            )
            # QUESTION: after memory_s, the estimated_state is already unpadded. How to get
            # the padded format and feed it into actor's observation?
            # SOLUTION: modify the code of Memory module, use masks= None to stop the unpadding.
            self.estimated_state_ = self.state_estimator(input_s)
            use_estimated_state_mask = torch.rand_like(observations[..., 0]) < self.replace_state_prob
            observations[use_estimated_state_mask] = replace_obs_components(
                observations[use_estimated_state_mask],
                self.estimator_target_components,
                self.estimated_state_[use_estimated_state_mask].detach(),
                self.obs_segments,
            )
            if inference:
                action = super().act_inference(observations)
            else:
                action = super().act(
                    observations,
                    masks=masks,
                    hidden_states=hidden_states if hidden_states is None else hidden_states.actor,
                )
        else:
            # both state estimator and actor are feedforward (non-recurrent)
            subobs = get_subobs_by_components(
                observations,
                component_names=self.estimator_obs_components,
                obs_segments=self.obs_segments,
            )
            self.estimated_state_ = self.state_estimator(subobs)
            use_estimated_state_mask = torch.rand_like(observations[..., 0]) < self.replace_state_prob
            observations[use_estimated_state_mask] = replace_obs_components(
                observations[use_estimated_state_mask],
                self.estimator_target_components,
                self.estimated_state_[use_estimated_state_mask].detach(),
                self.obs_segments,
            )
            if inference:
                action = super().act_inference(observations)
            else:
                action = super().act(observations, masks=masks, hidden_states=hidden_states)
        return action

    def act_inference(self, observations):
        return self.act(observations, inference=True)

    """ No modification required for evaluate() """

    def get_estimated_state(self):
        """In order to maintain the same interface of ActorCritic(Recurrent),
        the user must call this function after calling act() to get the estimated state.
        """
        return self.estimated_state_

    def get_hidden_states(self):
        return_ = super().get_hidden_states()
        if self.is_recurrent:
            return_ = return_._replace(
                actor=EstimatorActorHiddenState(
                    self.memory_s.hidden_states,
                    return_.actor,
                )
            )
        return return_


class EstimatorActorCritic(EstimatorMixin, ActorCritic):
    pass


class EstimatorActorCriticRecurrent(EstimatorMixin, ActorCriticRecurrent):
    pass
