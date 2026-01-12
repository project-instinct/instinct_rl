import importlib
from typing import Dict

from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .all_mixer import (
    EncoderMoEActorCritic,
    EncoderStateAc,
    EncoderStateAcRecurrent,
    EncoderVaeActor,
)
from .discriminator import Discriminator
from .encoder_actor_critic import EncoderActorCritic, EncoderActorCriticRecurrent
from .moe_actor_critic import MoEActorCritic
from .normalizer import (
    EmpiricalDiscountedVariationNormalization,
    EmpiricalNormalization,
)
from .parallel_layer import ParallelLayer
from .state_estimator import (
    EstimatorActorCritic,
    EstimatorActorCriticRecurrent,
    EstimatorMixin,
)
from .vae_actor import VaeActor


def build_actor_critic(
    policy_class_name: str,
    policy_cfg: dict,
    obs_format: Dict[str, Dict[str, tuple]],
    num_actions: int,
    num_rewards: int,
):
    """NOTE: This method allows to hack the policy kwargs by adding the env attributes to the policy_cfg."""
    if ":" in policy_class_name:
        module_name, class_name = policy_class_name.split(":")
        module = importlib.import_module(module_name)
        actor_critic_class = getattr(module, class_name)
    else:
        actor_critic_class = globals()[policy_class_name]  # ActorCritic

    policy_cfg = policy_cfg.copy()

    print(f"obs_format to build {policy_class_name}: {obs_format}")

    policy_cfg["obs_format"] = obs_format
    """ Get all shape information for each obs term in each observation group.
    """

    if not "num_actions" in policy_cfg:
        policy_cfg["num_actions"] = num_actions

    if not "num_rewards" in policy_cfg:
        policy_cfg["num_rewards"] = num_rewards

    actor_critic: ActorCritic = actor_critic_class(**policy_cfg)

    return actor_critic


def build_normalizer(
    input_shape: tuple,
    normalizer_class_name: str,
    normalizer_kwargs: dict = None,
):
    """Build the normalizer given your tensor shape and the normalizer class name + kwargs
    - If normalizer_class_name is None, this method will just return None.
    - If you want to use not-supported normalizer for quick test, you may set normalizer_class_name
        to a module direction, e.g. `instinct_rl.modules:EmpiricalNormalization`
    """
    if normalizer_class_name is not None:
        if normalizer_class_name == "EmpiricalNormalization":
            normalizer = EmpiricalNormalization(shape=input_shape, **normalizer_kwargs)
        elif normalizer_class_name == "EmpiricalDiscountedVariationNormalization":
            normalizer = EmpiricalDiscountedVariationNormalization(shape=input_shape, **normalizer_kwargs)
        elif ":" in normalizer_class_name:
            module_name, class_name = normalizer_class_name.split(":")
            module = importlib.import_module(module_name)
            normalizer_class = getattr(module, class_name)
            normalizer = normalizer_class(shape=input_shape, **normalizer_kwargs)
        else:
            raise ValueError(f"Unknown normalizer class name: {normalizer_class_name}")
    else:
        normalizer = None

    return normalizer
