"""Generic learned dynamics model.

Predicts the next simulator state from the current state, the action and a per-environment
context vector. It knows nothing about a particular robot, task or reward: the state layout,
the action size and the context size are all supplied by the caller, and the same class serves
any dynamics model that is queried on flat tensors.

Three conventions make it reusable across tasks:

* **State layout.** ``state_segments`` uses the same ``OrderedDict[str, tuple]`` shape as
  Instinct-RL's ``obs_format`` groups, so the existing helpers (``get_obs_slice``,
  ``get_subobs_size``, ``get_subobs_by_components``) apply to states as well as observations.
  A plain ``int`` is also accepted for a single flat vector.
* **Frames.** Every component may declare how it is expressed (``"world"``, ``"root"`` or
  ``"invariant"``). The module records this so that callers, the loss terms and the training
  script agree on units; it does not transform frames itself, because the right transform
  depends on the task. Predicting heading-invariant quantities (root-frame values, gravity
  directions, joint angles) is what lets one model serve every initial heading.
* **Delta vs absolute.** Components listed in ``predict_delta`` are regressed as a delta w.r.t.
  the current state, the rest as absolute values, so a mixed layout needs no special-casing.

Normalisation lives inside the module, so the loss scale does not depend on the physical units
of the state, and saving the model saves its statistics with it.
"""

from collections import OrderedDict
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .mlp import MlpModel
from .normalizer import EmpiricalNormalization


def _as_segments(state_segments) -> "OrderedDict[str, tuple]":
    """Accept an OrderedDict of segments, an int, or a shape for a single flat state."""
    if isinstance(state_segments, int):
        return OrderedDict({"state": (state_segments,)})
    if isinstance(state_segments, (list, tuple)):
        if not all(isinstance(v, int) for v in state_segments):
            raise TypeError("a sequence state_segments must be a flat shape of ints, or use an OrderedDict")
        size = int(np.prod(state_segments)) if len(state_segments) else 0
        return OrderedDict({"state": (size,)})
    if not isinstance(state_segments, dict):
        raise TypeError(f"state_segments must be a dict, int or sequence; got {type(state_segments)}")
    return OrderedDict(state_segments)


def _apply_normalizer(normalizer, x: torch.Tensor, update: bool) -> torch.Tensor:
    """Normalise ``x``, optionally updating the statistics.

    ``EmpiricalNormalization`` updates itself in ``forward()`` whenever it is in training mode, so
    the no-update path flips to eval for the duration of the call rather than reimplementing the
    arithmetic here.
    """
    if normalizer is None:
        return x
    if update:
        return normalizer(x)
    was_training = normalizer.training
    normalizer.eval()
    try:
        return normalizer(x)
    finally:
        normalizer.train(was_training)


class WorldModel(nn.Module):
    """MLP dynamics model over flattened (state, action, context) tuples.

    Args:
        state_segments: Layout of one state frame: ``OrderedDict[str, tuple]`` (same convention as
            ``obs_format`` groups), an ``int``, or a flat shape sequence.
        action_size: Size of the action vector.
        context_size: Size of the per-environment context vector; ``0`` disables it.
        predict_delta: Names of components regressed as a delta w.r.t. the current state.
            ``"all"`` (default) regresses every component as a delta, ``"none"`` as absolute
            values, or pass a sequence of component names for a mix.
        frame_of_component: How each component is expressed: ``"world"``, ``"root"`` or
            ``"invariant"``. Unlisted components default to ``"invariant"``.
        hidden_sizes: Hidden layer sizes of the dynamics MLP.
        nonlinearity: Activation name (``torch.nn`` attribute) or module class.
        use_normalizer: Wrap the network input and output in ``EmpiricalNormalization``.
        normalizer_kwargs: Extra kwargs for the normaliser constructor.
        matmul_dtype: Optional dtype for the MLP matmuls (e.g. ``"bfloat16"``); inputs and
            outputs are cast back to float32. ``None`` keeps the module dtype.
    """

    def __init__(
        self,
        state_segments,
        action_size: int,
        context_size: int = 0,
        predict_delta: Sequence[str] | str = "all",
        frame_of_component: dict[str, str] | None = None,
        hidden_sizes: Sequence[int] = (2048, 2048, 2048),
        nonlinearity: str = "SiLU",
        use_normalizer: bool = True,
        normalizer_kwargs: dict | None = None,
        matmul_dtype: str | None = None,
    ):
        super().__init__()
        self.state_segments: "OrderedDict[str, tuple]" = _as_segments(state_segments)
        self.state_size = int(sum(int(np.prod(s)) for s in self.state_segments.values()))
        if self.state_size <= 0:
            raise ValueError("state_segments describe an empty state")
        # Any component ending in ``quat_w`` holds unit quaternions. The network regresses them as four
        # free numbers, so their norm drifts (measured: 1.0 -> 0.1 after ~10 open-loop steps) and every
        # consumer that inverts a rotation -- ``quat_inv`` divides by |q|^2 -- then amplifies the error.
        # Renormalising after every prediction keeps the rollout on the unit sphere.
        self._quaternion_components = [name for name in self.state_segments if name.endswith("quat_w")]
        self.action_size = int(action_size)
        if self.action_size <= 0:
            raise ValueError("action_size must be positive")
        self.context_size = int(context_size)
        if self.context_size < 0:
            raise ValueError("context_size must be >= 0")

        self.frame_of_component: dict[str, str] = {
            name: (frame_of_component or {}).get(name, "invariant") for name in self.state_segments
        }
        for name, frame in self.frame_of_component.items():
            if frame not in ("world", "root", "invariant"):
                raise ValueError(f"frame_of_component[{name!r}] must be 'world', 'root' or 'invariant', got {frame!r}")
        if frame_of_component:
            unknown = set(frame_of_component) - set(self.state_segments)
            if unknown:
                raise ValueError(f"frame_of_component names components absent from state_segments: {sorted(unknown)}")

        if predict_delta == "all":
            self.predict_delta = set(self.state_segments)
        elif predict_delta == "none":
            self.predict_delta = set()
        elif isinstance(predict_delta, str):
            raise ValueError(f"predict_delta must be 'all', 'none' or a sequence of names; got {predict_delta!r}")
        else:
            unknown = set(predict_delta) - set(self.state_segments)
            if unknown:
                raise ValueError(f"predict_delta names components absent from state_segments: {sorted(unknown)}")
            self.predict_delta = set(predict_delta)

        self.matmul_dtype = getattr(torch, matmul_dtype) if isinstance(matmul_dtype, str) else matmul_dtype
        if self.matmul_dtype is not None and not self.matmul_dtype.is_floating_point:
            raise ValueError(f"matmul_dtype must be a floating dtype, got {self.matmul_dtype}")

        input_size = self.state_size + self.action_size + self.context_size
        self.net = MlpModel(
            input_size=input_size,
            hidden_sizes=list(hidden_sizes),
            output_size=self.state_size,
            nonlinearity=nonlinearity,
        )

        if use_normalizer:
            normalizer_kwargs = dict(normalizer_kwargs or {})
            self.input_normalizer = EmpiricalNormalization(shape=[input_size], **normalizer_kwargs)
            self.output_normalizer = EmpiricalNormalization(shape=[self.state_size], **normalizer_kwargs)
        else:
            self.input_normalizer = None
            self.output_normalizer = None

    # ------------------------------------------------------------------ helpers

    def _concat(self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        leading = state.shape[:-1]
        parts = [
            state.reshape(*leading, self.state_size),
            action.reshape(*leading, self.action_size),
        ]
        if self.context_size:
            if context is None:
                raise ValueError("context_size > 0 but no context tensor was given")
            parts.append(context.reshape(*leading, self.context_size))
        elif context is not None:
            raise ValueError("a context tensor was given but context_size == 0")
        return torch.cat(parts, dim=-1)

    def _segment_slices(self):
        offset = 0
        for name, shape in self.state_segments.items():
            size = int(np.prod(shape))
            yield name, slice(offset, offset + size)
            offset += size

    def _renormalize_quaternions(self, state: torch.Tensor) -> torch.Tensor:
        """Project every quaternion back onto the unit sphere.

        A component may hold several quaternions (``(num_links, 4)``); each is normalised on its own.
        Normalising the flattened component as one vector would scale ``n`` unit quaternions to norm
        ``1 / sqrt(n)`` each.
        """
        if not self._quaternion_components:
            return state
        chunks = []
        for name, sl in self._segment_slices():
            chunk = state[..., sl]
            if name in self._quaternion_components:
                quats = chunk.reshape(*chunk.shape[:-1], -1, 4)
                chunk = (quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-6)).reshape(chunk.shape)
            chunks.append(chunk)
        return torch.cat(chunks, dim=-1)

    def _split_target(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Regression target: delta where requested, absolute value otherwise."""
        chunks = []
        for name, sl in self._segment_slices():
            if name in self.predict_delta:
                chunks.append(next_state[..., sl] - state[..., sl])
            else:
                chunks.append(next_state[..., sl])
        return torch.cat(chunks, dim=-1)

    def _assemble_next(self, state: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_split_target`."""
        chunks = []
        for name, sl in self._segment_slices():
            if name in self.predict_delta:
                chunks.append(state[..., sl] + prediction[..., sl])
            else:
                chunks.append(prediction[..., sl])
        return torch.cat(chunks, dim=-1)

    def _update_normalizers(self, state0, actions, context, targets, mask) -> None:
        """Update both normalisers from the *logged* transitions of a window.

        Statistics come from ground truth only: an autoregressive fit also feeds the network its own
        predictions, and folding those in would let the model's errors redefine its input scale.
        The output normaliser sees the regression targets, so it also defines the error scale.
        """
        if self.input_normalizer is None:
            return
        steps = actions.shape[1]
        logged_inputs = torch.cat([state0.unsqueeze(1), targets[:, :-1]], dim=1)  # s_0 .. s_{k-1}
        valid = (
            torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
            if mask is None
            else mask.to(torch.bool)
        )
        if not bool(valid.any()):
            return
        step_context = None if context is None else context.unsqueeze(1).expand(-1, steps, -1)
        with torch.no_grad():
            self.input_normalizer.update(self._concat(logged_inputs, actions, step_context)[valid])
            self.output_normalizer.update(self._split_target(logged_inputs, targets)[valid])

    def _error_scale(self) -> torch.Tensor | float:
        """Per-dimension scale of the fitting error: the spread of the one-step regression target.

        For delta components this is the typical one-step change; for absolute components, the spread
        of the value itself. Without normalisers the error is left in raw units.
        """
        if self.output_normalizer is None:
            return 1.0
        return (self.output_normalizer._std + self.output_normalizer.eps).detach()

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
        update_normalizer: bool = False,
    ) -> torch.Tensor:
        """Predict the next state (absolute values, world layout). Leading dims are preserved."""
        x = self._concat(state, action, context)
        x = _apply_normalizer(self.input_normalizer, x, update=update_normalizer)
        if self.matmul_dtype is not None:
            x = x.to(self.matmul_dtype)
        prediction = self.net(x)
        if self.matmul_dtype is not None:
            prediction = prediction.float()
        if self.output_normalizer is not None:
            prediction = self.output_normalizer.inverse(prediction)
        next_state = self._assemble_next(state.reshape(*state.shape[:-1], self.state_size), prediction)
        return self._renormalize_quaternions(next_state)

    def rollout(
        self,
        state0: torch.Tensor,
        actions: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Open-loop rollout, one :meth:`forward` call per step.

        Args:
            state0: ``(B, state_size)`` initial state.
            actions: ``(B, k, action_size)`` actions applied at each step.
            context: ``(B, context_size)`` context, constant over the rollout.
            mask: Optional ``(B, k)`` mask; where False the previous state is carried forward, which
                keeps a padded batch's shape stable.
        Returns:
            ``(B, k, state_size)`` predicted states *after* each action.
        """
        if actions.dim() != 3:
            raise ValueError(f"actions must be (B, k, action_size), got {tuple(actions.shape)}")
        state = state0.reshape(state0.shape[0], -1)
        out = []
        for step in range(actions.shape[1]):
            next_state = self.forward(state, actions[:, step], context)
            if mask is not None:
                next_state = torch.where(mask[:, step].unsqueeze(-1), next_state, state)
            out.append(next_state)
            state = next_state
        return torch.stack(out, dim=1)

    def rollout_loss(
        self,
        state0: torch.Tensor,
        actions: torch.Tensor,
        context: torch.Tensor | None,
        target_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        update_normalizer: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Multi-step open-loop loss, the SuperMimic k-step fit expressed generically.

        Args:
            state0: ``(B, state_size)``.
            actions: ``(B, k, action_size)``.
            context: ``(B, context_size)`` or None.
            target_states: ``(B, k, state_size)`` ground-truth next states.
            mask: Optional ``(B, k)`` validity mask (e.g. stops at episode ends).
            update_normalizer: Update normaliser statistics from this batch.
        Returns:
            ``{"loss": scalar, "loss_<component>": scalar}``; per-component terms are diagnostics.
        """
        if actions.shape[:2] != target_states.shape[:2]:
            raise ValueError(
                f"actions and target_states disagree on (B, k): {tuple(actions.shape)} vs {tuple(target_states.shape)}"
            )
        batch, steps = actions.shape[0], actions.shape[1]
        state = state0.reshape(batch, self.state_size)
        targets = target_states.reshape(batch, steps, self.state_size)

        if update_normalizer:
            self._update_normalizers(state, actions, context, targets, mask)

        # Autoregressive (paper Eq. 1): each step starts from the model's own previous prediction, and
        # the gradient flows back through it. Restarting every step from the logged state instead would
        # make this ``k`` independent one-step fits, and the model would never learn to correct the
        # errors it makes on its own outputs -- the property long imagined rollouts depend on.
        predictions = []
        for step in range(steps):
            state = self.forward(state, actions[:, step], context)
            predictions.append(state)
        predictions = torch.stack(predictions, dim=1)  # (B, k, state_size)

        # Errors are measured in units of each dimension's typical one-step change, so a joint angle
        # and a link velocity count alike; in raw units the loss is dominated by the largest-valued
        # components (here a ~70x spread in scale).
        sq_err = ((predictions - targets) / self._error_scale()) ** 2  # (B, k, state_size)

        if mask is None:
            per_step = sq_err.mean(dim=-1)  # (B, k)
            weight = None
        else:
            weight = mask.to(sq_err.dtype)  # (B, k)
            per_step = sq_err.mean(dim=-1)

        losses: dict[str, torch.Tensor] = {}
        if weight is None:
            losses["loss"] = per_step.mean()
        else:
            losses["loss"] = (per_step * weight).sum() / weight.sum().clamp_min(1.0)
        for name, sl in self._segment_slices():
            comp = sq_err[..., sl].mean(dim=-1)  # (B, k)
            if weight is None:
                losses[f"loss_{name}"] = comp.mean()
            else:
                losses[f"loss_{name}"] = (comp * weight).sum() / weight.sum().clamp_min(1.0)
        return losses

    def extra_repr(self) -> str:
        return (
            f"state_size={self.state_size}, action_size={self.action_size}, context_size={self.context_size}, "
            f"delta_components={sorted(self.predict_delta)}"
        )
