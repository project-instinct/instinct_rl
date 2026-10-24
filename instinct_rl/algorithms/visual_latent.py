"""Visual latent state for `SuperMimic`: the policy sees an image, the world model does not render it.

Adapted from the SuperMimic reference implementation (https://github.com/zita-ch/supermimic_vibed, MIT License,
Copyright (c) 2026 ETH Zurich, Chong Zhang; see licenses/dependencies/supermimic_vibed_license.txt) (``train.py``).

The image encoder ``E`` of the policy (an entry of ``actor_critic.encoders``) maps the processed image to a latent
``z = E(image)``. The latent is made a state of the world model: the dynamics MLP takes ``z_t`` as an extra input and
predicts its increment, ``(s_{t+1}, z_{t+1}) = f(s_t, z_t, a_t, xi, scan)``, so nothing is rendered in imagination and
the actor MLP acts on the imagined latent directly (WMP-style). The latent is trained in the same k-step windows as the
dynamics, against ``E`` of the real (noised) image at every step. ``E`` is trained by the world model, not by the policy
loss: its latent is grounded by the reconstruction of the CLEAN image by a `LatentDecoder` (training only), which keeps
the latent prediction loss from collapsing.

The ``world_model`` observation group of the env has two more components: the image as the policy sees it
(``image_component``, with the sensor noise, not normalized) and the CLEAN processed image (``clean_image_component``).
The task model is a `VisualTaskModel`, the policy an actor critic with an encoder for the image term of its ``policy``
observation group, whose output goes to the actor MLP after the other terms (e.g. `EncoderActorCritic`).
"""

from __future__ import annotations

import torch

import instinct_rl.modules as instinct_modules
from instinct_rl.env import VisualTaskModel
from instinct_rl.modules.world_model import unflatten_state
from instinct_rl.utils.utils import get_obs_slice

from .supermimic import SuperMimic

LATENT_GAP_STEPS = (1, 8, 16, 32)


class VisualLatentAlgoMixin:
    """Adds the visual latent state to `SuperMimic`. Stored in the replay storage are the image at the start of a
    transition (``image``), the image at its end (``image1``, both as the policy sees them, with the sensor noise) and
    the clean image at its end (``image1_clean``), read from the terminal observation (BEFORE the reset of the finished
    envs)."""

    def __init__(
        self,
        *args,
        image_obs_name="depth_image",
        image_component="depth_image",
        clean_image_component="depth_image_clean",
        latent_windows=2048,
        latent_prediction_loss_coef=1.0,
        reconstruction_images=512,
        reconstruction_loss_coef=1.0,
        latent_grad=False,
        **kwargs,
    ):
        """

        Args:
            image_obs_name: Name of the image term in the ``policy`` observation group of the env. The actor critic must
                have an encoder of that name (``encoders._parallel_blocks``).
            image_component: Name of the component of the ``world_model`` group with the image of the policy.
            clean_image_component: Name of the component of the ``world_model`` group with the clean image.
            latent_windows: Windows (of ``world_model_batch_size``) on which the latent prediction loss is computed;
                every window still rolls out with its latent (the dynamics take it as input), only the target images of
                these are encoded.
            latent_prediction_loss_coef: Weight of the k-step latent prediction loss (in units of the latent variance)
                relative to the dynamics loss.
            reconstruction_images: Images per world-model batch for the clean-image reconstruction loss of the latent.
            reconstruction_loss_coef: Weight of the reconstruction (L1) loss.
            latent_grad: Back-propagate the policy loss through the latent fed to the actor into the imagined states;
                False = stop-gradient on the latent fed to the actor (as WMP does), so the actor cannot exploit errors
                of the latent prediction. The world model's own use of the latent (the input of the next step) keeps
                its gradient.
        """
        super().__init__(*args, **kwargs)
        self.image_obs_name = image_obs_name
        self.image_component = image_component
        self.clean_image_component = clean_image_component
        self.latent_windows = latent_windows
        self.latent_prediction_loss_coef = latent_prediction_loss_coef
        self.reconstruction_images = reconstruction_images
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.latent_grad = latent_grad

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1, **kwargs):
        image_slice, self.image_shape = get_obs_slice(obs_format["policy"], self.image_obs_name)
        self.image_slice = slice(int(image_slice.start), int(image_slice.stop))
        assert obs_format[self.world_model_obs_group][self.image_component] == self.image_shape
        assert obs_format[self.world_model_obs_group][self.clean_image_component] == self.image_shape
        self.encoder = self.actor_critic.encoders._parallel_blocks[self.image_obs_name]  # the policy's image encoder E
        self.z_dim = self.encoder.output_size
        # the actor MLP input is the encoder output segment: every non-image term in order, then the image latent
        segments = list(self.actor_critic.encoders.output_segment)
        assert segments[:-1] == [name for name in obs_format["policy"] if name != self.image_obs_name]
        assert segments[-1].endswith(self.image_obs_name)
        self.latent_decoder = instinct_modules.LatentDecoder(self.z_dim, image_shape=self.image_shape).to(self.device)

        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards, **kwargs)
        assert isinstance(self.task_model, VisualTaskModel), "VisualSuperMimic needs a VisualTaskModel"
        self.window_keys += ("image", "image1_clean", "image1")
        self.start_keys += ("image",)

    def storage_fields(self, num_actions):
        field_specs = super().storage_fields(num_actions)
        field_specs["image"] = (self.image_shape, torch.float16)  # the env's (noised) processed policy image at s
        field_specs["image1_clean"] = (self.image_shape, torch.float16)  # CLEAN processed image at s1 (pre-reset)
        field_specs["image1"] = (self.image_shape, torch.float16)  # the env's (noised) policy image at s1
        return field_specs

    def transition_fields(self, start, end):
        transition = super().transition_fields(start, end)
        transition["image"] = self.component(start["world_model_obs"], self.image_component).half()
        transition["image1"] = self.component(end["world_model_obs"], self.image_component).half()
        transition["image1_clean"] = self.component(end["world_model_obs"], self.clean_image_component).half()
        return transition

    def collect_world_model_parameters(self):
        # E is trained by the world model (reconstruction), not by the policy loss
        return [
            *super().collect_world_model_parameters(),
            *self.encoder.parameters(),
            *self.latent_decoder.parameters(),
        ]

    def encode_image(self, image):
        """Latent of the policy's image encoder for processed images (B, *image_shape), normalized as in the policy
        observation (the encoder sees exactly what it sees in ``act_inference``)."""
        normalizer = self.policy_normalizer
        mean = normalizer.mean[self.image_slice]
        std = normalizer.std[self.image_slice] + normalizer.eps
        return self.encoder(((image.flatten(1) - mean) / std).view(-1, *self.image_shape))

    def world_model_step(self, state, action, context_normalized, scan, latent):
        return self.world_model.step_z(state, action, context_normalized, scan, latent)

    def policy_action(self, state, proprio_history, command, latent):
        """The actor MLP acts on the imagined latent in place of the encoder output: the observation of the other terms
        is assembled as usual (the image term is cut out)."""
        latent_for_actor = latent if self.latent_grad else latent.detach()
        image = latent.new_zeros(latent.shape[0], *self.image_shape)
        obs = self.task_model.policy_observation(proprio_history, state, command, image)
        obs = self.normalize_policy_observation(obs)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.policy_autocast):  # training imagination only
            actor_input = torch.cat(
                (obs[:, : self.image_slice.start], obs[:, self.image_slice.stop :], latent_for_actor), -1
            )
            return self.actor_critic.actor(actor_input).float()

    def start_latent(self, starts):
        with torch.no_grad():
            return self.encode_image(starts["image"].float())

    def rollout(self, batch, context_normalized):
        if "initial_latent" not in batch:
            with torch.no_grad():
                batch["initial_latent"] = self.encode_image(batch["image"][:, 0].float())
        return super().rollout(batch, context_normalized)

    def compute_world_model_losses(self, batch):
        """The latent state of the world model: k-step autoregressive latent error against E(real noised image) at every
        step, in units of the target latent variance, on the first ``latent_windows`` windows; plus the reconstruction
        of the clean image from E(noised image) at the same instant, which grounds E (training only)."""
        with torch.no_grad():
            num_latent_windows, window = min(self.latent_windows, batch["mask"].shape[0]), batch["mask"].shape[1]
            initial_latent = self.encode_image(batch["image"][:, 0].float())
            target_latents = self.encode_image(batch["image1"][:num_latent_windows].flatten(0, 1).float()).view(
                num_latent_windows, window, -1
            )
            # before the rollout: the latent statistics are used (in place updated) inside its graph
            latent_increments = target_latents - torch.cat(
                (initial_latent[:num_latent_windows].unsqueeze(1), target_latents[:, :-1]), 1
            )
            self.world_model.update_z_stats(target_latents, latent_increments, batch["mask"][:num_latent_windows])
        batch["initial_latent"] = initial_latent
        losses, inter_vars, stats = super().compute_world_model_losses(batch)
        mask = batch["mask"][:num_latent_windows]
        latent_scale = target_latents.var((0, 1)).mean().clamp(min=1e-6)
        latent_error = ((inter_vars["predicted_latents"][:num_latent_windows] - target_latents) ** 2).mean(-1)
        latent_error = latent_error / latent_scale  # (num_latent_windows, k)
        losses["latent_prediction_loss"] = (latent_error * mask).sum() / num_latent_windows
        stats["latent_prediction_error"] = (latent_error.detach() * mask).sum() / mask.sum().clamp(min=1)
        # k-step latent error
        last_step_mask = mask[:, -1]
        stats["latent_prediction_error_last_step"] = (
            latent_error.detach()[:, -1] * last_step_mask
        ).sum() / last_step_mask.sum().clamp(min=1)
        num_images = min(self.reconstruction_images, num_latent_windows * window)
        reconstruction = self.latent_decoder(self.encode_image(batch["image1"].flatten(0, 1)[:num_images].float()))
        losses["reconstruction_loss"] = (
            (reconstruction - batch["image1_clean"].flatten(0, 1)[:num_images].float()).abs().mean()
        )
        return losses, inter_vars, stats

    @torch.no_grad()
    def compute_diagnostics(self, termination_check):
        diagnostics = super().compute_diagnostics(termination_check)
        diagnostics.update(self.latent_gap())
        return diagnostics

    def latent_gap(self, num_windows: int = 256) -> dict:
        """Latent error vs open-loop horizon step on real windows of the storage (anchor = the real image at the window
        start): the latent rolled out with the states from E(anchor) (``pred``) and "copy the anchor latent" (``copy``),
        as the squared error against E(real noised image) in units of the target latent variance. The tags keep the
        ``depth_gap`` name of the image modes."""
        steps = [step for step in LATENT_GAP_STEPS if step <= self.imagination_horizon]
        batch = self.storage.sample_windows(
            num_windows, self.imagination_horizon, self.window_keys, self.storage.holdout_envs
        )
        anchor_latent = self.encode_image(batch["image"][:, 0].float())
        state, latent, diagnostics = unflatten_state(batch["s"][:, 0]), anchor_latent, {}
        for step in range(self.imagination_horizon):
            state, _, latent = self.world_model_step(
                state, batch["a"][:, step], self.context_normalized[batch["env"]], self.task_model.scan(state), latent
            )
            if step + 1 in steps:
                mask = batch["mask"][:, step]
                target_latent = self.encode_image(batch["image1"][:, step].float())
                latent_scale = target_latent.var(0).mean().clamp(min=1e-6)
                for name, predicted_latent in (("pred", latent), ("copy", anchor_latent)):
                    error = ((predicted_latent - target_latent) ** 2).mean(-1) / latent_scale
                    masked_mean = (error * mask).sum() / mask.sum().clamp(min=1)
                    diagnostics[f"depth_gap/step_{step + 1}_{name}"] = float(masked_mean)
        return diagnostics

    def state_dict(self):
        return {**super().state_dict(), "latent_decoder": self.latent_decoder.state_dict()}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.latent_decoder.load_state_dict(state_dict["latent_decoder"])


class VisualSuperMimic(VisualLatentAlgoMixin, SuperMimic):
    pass
