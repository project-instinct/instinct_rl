""" A file put all mixin class combinations that are not only mixing ActorCritic or ActorCriticRecurrent"""

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .encoder_actor_critic import EncoderActorCriticMixin
from .moe_actor_critic import MoEActorCritic
from .state_estimator import EstimatorMixin
from .vae_actor_critic import VaeActorCritic


class EncoderStateAc(EstimatorMixin, EncoderActorCriticMixin, ActorCritic):
    pass


class EncoderStateAcRecurrent(EstimatorMixin, EncoderActorCriticMixin, ActorCriticRecurrent):
    def load_misaligned_state_dict(self, module, obs_segments, critic_obs_segments=None):
        pass


class EncoderMoEActorCritic(EncoderActorCriticMixin, MoEActorCritic):
    pass


class EncoderVaeActorCritic(EncoderActorCriticMixin, VaeActorCritic):
    pass
