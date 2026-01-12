import importlib
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import instinct_rl.modules as instinct_modules
from instinct_rl.algorithms.ppo import PPO
from instinct_rl.storage.amp_storage import AmpStorage
from instinct_rl.utils.buffer import buffer_func
from instinct_rl.utils.utils import get_subobs_size


class WasabiAlgoMixin:
    """A plugin algorithm for [WASABI](https://github.com/martius-lab/wasabi), which also includes traditional AMP algorithm
    (Set `discriminator_loss_func` to "BCEWithLogitsLoss" for AMP).
    """

    def __init__(
        self,
        *args,
        actor_state_key="amp_policy",  # the key of getting policy's state sequence
        reference_state_key="amp_reference",  # the key of getting expert's reference sequence
        discriminator_class_name="Discriminator",
        discriminator_kwargs={},
        discriminator_optimizer_class_name="AdamW",
        discriminator_optimizer_kwargs={},
        discriminator_reward_coef=1.0,
        discriminator_reward_type: Literal[
            "log", "quad", "wasserstein"
        ] = "log",  # check more on `compute_discriminator_reward`
        discriminator_loss_func: Literal[
            "WassersteinLoss", "BCEWithLogitsLoss", "MSELoss"
        ] = "BCEWithLogitsLoss",  # by default, lead to AMP
        discriminator_loss_coef=1.0,
        discriminator_gradient_penalty_coef=10.0,
        discriminator_weight_decay_coef=0.0,
        discriminator_logit_weight_decay_coef=0.0,  # loss for last layer weight
        discriminator_gradient_torlerance=0.0,  # If the computed gradient is smaller than this value, the gradient will not be penalized.
        discriminator_backbone_gradient_only=False,  # If True, the discriminator must support encoders and backbone_run function.
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.actor_state_key = actor_state_key
        self.reference_state_key = reference_state_key
        self.discriminator_class_name = discriminator_class_name
        self.discriminator_kwargs = discriminator_kwargs
        self.discriminator_optimizer_class_name = discriminator_optimizer_class_name
        self.discriminator_optimizer_kwargs = discriminator_optimizer_kwargs
        self.discriminator_reward_coef = discriminator_reward_coef
        self.discriminator_reward_type = discriminator_reward_type
        self.discriminator_loss_func = discriminator_loss_func
        self.discriminator_loss_coef = discriminator_loss_coef
        self.discriminator_gradient_penalty_coef = discriminator_gradient_penalty_coef
        self.discriminator_weight_decay_coef = discriminator_weight_decay_coef
        self.discriminator_logit_weight_decay_coef = discriminator_logit_weight_decay_coef
        self.discriminator_gradient_torlerance = discriminator_gradient_torlerance
        self.discriminator_backbone_gradient_only = discriminator_backbone_gradient_only

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)
        # build discriminator
        if ":" in self.discriminator_class_name:
            DiscriminatorClass = getattr(
                importlib.import_module(self.discriminator_class_name.split(":")[0]),
                self.discriminator_class_name.split(":")[1],
            )
        else:
            DiscriminatorClass = getattr(instinct_modules, self.discriminator_class_name)
        self.discriminator = DiscriminatorClass(
            input_segment=obs_format[self.actor_state_key],
            **self.discriminator_kwargs,
        ).to(self.device)
        if not "lr" in self.discriminator_optimizer_kwargs:
            self.discriminator_optimizer_kwargs = dict(lr=self.learning_rate, **self.discriminator_optimizer_kwargs)
        self.discriminator_optimizer = getattr(optim, self.discriminator_optimizer_class_name)(
            self.discriminator.parameters(),
            **self.discriminator_optimizer_kwargs,
        )

        self.amp_transition = AmpStorage.Transition()
        reference_state_size = get_subobs_size(obs_format[self.actor_state_key])
        actor_state_size = get_subobs_size(obs_format[self.reference_state_key])
        assert actor_state_size == reference_state_size, "The shape of robot state and reference state must be the same"
        self.amp_storage = AmpStorage(
            num_envs, num_transitions_per_env, [actor_state_size], [reference_state_size], device=self.device
        )

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        if not (self.actor_state_key in infos["observations"] and self.reference_state_key in infos["observations"]):
            raise ValueError(
                "The key of trajectory observations ({}) or reference observations ({}) is not found in the observation"
                " dictionary".format(self.actor_state_key, self.reference_state_key)
            )

        actor_state = infos["observations"][self.actor_state_key]
        reference_state = infos["observations"][self.reference_state_key]
        self.amp_transition.actor_states = actor_state
        self.amp_transition.reference_states = reference_state
        if self.discriminator.is_recurrent:
            self.amp_transition.hidden_states = self.discriminator.get_hidden_states()
        self.amp_transition.dones = dones
        self.amp_storage.add_transitions(self.amp_transition)
        self.amp_transition.clear()

        # do not call compute_auxilary_reward here, because it is called in the baseclass function
        super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        super_auxilary_reward = super().compute_auxiliary_reward(obs_pack)
        actor_state = obs_pack[self.actor_state_key]
        disc = self.discriminator(actor_state).detach()
        if self.discriminator_reward_type == "log":
            # Typically discimination is the output of a direct linear layer. This is the default AMP implementation.
            reward = -torch.log(1 - torch.clamp(torch.sigmoid(disc), 1e-6, 1 - 1e-6))
        elif self.discriminator_reward_type == "quad":
            # Copied from WASABI, not sure if this is correct. This assumes disc is the output of a direct linear layer.
            reward = torch.clamp(1 - (1 / 4) * torch.square(disc - 1), min=0)
        elif self.discriminator_reward_type == "wasserstein":
            # Copied from WASABI, not sure if this is correct. This assumes disc is the output of a direct linear layer.
            reward = disc
        super_auxilary_reward["discriminator_reward"] = reward
        return super_auxilary_reward

    def update(self, *args, **kwargs):
        mean_losses, average_stats = super().update(*args, **kwargs)

        # iterate over the discriminator optimization steps
        if self.discriminator.is_recurrent:
            # Leaving possibility for recurrent discriminator, but this is not well designed yet.
            amp_generator = self.amp_storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            amp_generator = self.amp_storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for amp_minibatch in amp_generator:
            losses, _, stats = self.compute_amp_losses(amp_minibatch)

            loss = 0.0
            for k, v in losses.items():
                loss += getattr(self, k + "_coef", 1.0) * v
                mean_losses[k] = mean_losses[k] + v.detach() / num_updates
            mean_losses["amp_total_loss"] = mean_losses["amp_total_loss"] + loss.detach() / num_updates
            for k, v in stats.items():
                average_stats[k] = average_stats[k] + v.detach() / num_updates

            self.wasabi_gradient_step(loss, average_stats)

        self.amp_storage.clear()

        return mean_losses, average_stats

    def compute_amp_losses(self, amp_minibatch: AmpStorage.MiniBatch):
        losses, stats, inter_vars = dict(), dict(), dict()

        # discriminator must compute the discriminator during act()
        # TODO: run recurrent discriminator.
        actor_d = self.discriminator(
            amp_minibatch.actor_states,
            masks=amp_minibatch.masks,
        )
        reference_d = self.discriminator(
            amp_minibatch.reference_states,
            masks=amp_minibatch.masks,
        )

        # compute losses of the distriminator
        if self.discriminator_loss_func == "WassersteinLoss":
            actor_d_loss = actor_d.mean()
            reference_d_loss = -reference_d.mean()
        elif self.discriminator_loss_func == "BCEWithLogitsLoss":
            actor_d_loss = torch.nn.BCEWithLogitsLoss()(actor_d, torch.zeros_like(actor_d))
            reference_d_loss = torch.nn.BCEWithLogitsLoss()(reference_d, torch.ones_like(reference_d))
        elif self.discriminator_loss_func == "MSELoss":
            actor_d_loss = torch.nn.MSELoss()(actor_d, -torch.ones_like(actor_d))
            reference_d_loss = torch.nn.MSELoss()(reference_d, torch.ones_like(reference_d))
        else:
            raise NotImplementedError(f"discriminator loss function {self.discriminator_loss_func} not implemented")
        discriminator_loss = (actor_d_loss + reference_d_loss) * 0.5
        if self.discriminator_gradient_penalty_coef > 0:
            if self.discriminator_backbone_gradient_only:
                discriminator_gradient_penalty = self.compute_discriminator_backbone_gradient(amp_minibatch)
            else:
                discriminator_gradient_penalty = self.compute_discriminator_gradient(amp_minibatch)
        else:
            discriminator_gradient_penalty = torch.zeros(1, device=self.device)[0]

        # compute weight decay loss
        weight_decay_loss = torch.zeros(1, device=self.device)[0]
        for param in self.discriminator.parameters():
            weight_decay_loss += torch.sum(param**2)

        # Following ProtoMotions, compute the weight decay loss for the last layer of the discriminator.
        logit_weight_decay_loss = torch.zeros(1, device=self.device)[0]
        if self.discriminator_logit_weight_decay_coef > 0:
            logit_weight = self.discriminator.logit_layer_weights()
            logit_weight_decay_loss = torch.sum(logit_weight**2)

        losses["discriminator_loss"] = discriminator_loss
        losses["discriminator_gradient_penalty"] = discriminator_gradient_penalty
        losses["discriminator_weight_decay"] = weight_decay_loss
        losses["logit_weight_decay"] = logit_weight_decay_loss
        stats["discriminator_actor"] = actor_d.mean()
        stats["discriminator_reference"] = reference_d.mean()

        return losses, inter_vars, stats

    def compute_discriminator_backbone_gradient(self, amp_minibatch: AmpStorage.MiniBatch):
        """Compute the gradient w.r.t discriminator input.
        In WASABI algorithm, it is used as a penalty to satisfy the condition of Lipschitz continuity
        """
        reference_states = amp_minibatch.reference_states
        reference_states = buffer_func(reference_states, torch.clone)

        actor_states = amp_minibatch.actor_states
        actor_states = buffer_func(actor_states, torch.clone)

        combined_states = torch.cat([actor_states, reference_states], dim=0)

        # NOTE: assumeing discriminator has encoders and backbone_run function
        latent = self.discriminator.encoders(combined_states).detach()
        latent.requires_grad = True

        disc = self.discriminator.backbone_run(latent)

        ones = torch.ones_like(disc)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=latent,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return torch.clamp(grad.norm(2, dim=1) - self.discriminator_gradient_torlerance, 0).pow(2).mean()

    def compute_discriminator_gradient(self, amp_minibatch: AmpStorage.MiniBatch):
        """Compute the gradient w.r.t expert input.
        In WASABI algorithm, it is used as a penalty to satisfy the condition of Lipschitz continuity
        """
        reference_states = amp_minibatch.reference_states
        reference_states = buffer_func(reference_states, torch.clone)
        buffer_func(reference_states, setattr, "requires_grad", True)

        actor_states = amp_minibatch.actor_states
        actor_states = buffer_func(actor_states, torch.clone)
        buffer_func(actor_states, setattr, "requires_grad", True)

        combined_states = torch.cat([actor_states, reference_states], dim=0)

        disc = self.discriminator(
            combined_states,
            masks=amp_minibatch.masks,  # The mask is designed as if the discriminator is recurrent. But it is typically not.
        )

        ones = torch.ones_like(disc)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=combined_states,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return torch.clamp(grad.norm(2, dim=1) - self.discriminator_gradient_torlerance, 0).pow(2).mean()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["discriminator"] = self.discriminator.state_dict()
        state_dict["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "discriminator" in state_dict:
            self.discriminator.load_state_dict(state_dict["discriminator"])
        else:
            print("[Warning] The discriminator state_dict is not found in the checkpoint")
        if "discriminator_optimizer" in state_dict:
            self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])
        else:
            print("[Warning] The discriminator_optimizer state_dict is not found in the checkpoint")

    def wasabi_gradient_step(self, loss: torch.Tensor, average_stats: dict):
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for param in self.discriminator.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
        # TODO: add clip grad norm
        self.discriminator_optimizer.step()


class WasabiPPO(WasabiAlgoMixin, PPO):
    pass
