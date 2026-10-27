"""Ring-buffer replay storage for model-based algorithms.

Stores a rolling window of transitions, indexed by ``(time, env)``, and serves two kinds of batches:

* :meth:`ReplayStorage.sample_windows` -- short contiguous windows, for fitting a world model.
* :meth:`ReplayStorage.sample_starts` -- single start states (with just enough history rebuilt from the
  ring), for imagining rollouts from.

The storage is generic: the caller declares which tensors to keep with ``TensorSpec`` entries, so the
same class serves a dynamics-model fit (state/action/context) and anything else that needs raw
transitions.

Design notes:

* Allocation is ``(capacity_per_env, num_envs, *feature)``; the newest step overwrites the oldest. A
  per-env ``episode_step`` tensor is required so that windows never cross an episode boundary.
* Windows are masked rather than dropped: a window that runs into an episode start is returned with a
  mask so the caller can ignore the invalid steps, which keeps batch shapes static (important for
  compile and for deterministic tests).
* Only tensors and ints are exposed through ``state_dict()`` so checkpoints load with
  ``weights_only=True``.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from instinct_rl.utils.buffer import buffer_method


@dataclass
class TensorSpec:
    """Shape and dtype of one stored tensor (per step, per env)."""

    shape: tuple = field(default_factory=tuple)
    dtype: torch.dtype = torch.float32


class ReplayStorage:
    """Time x env ring buffer over a fixed set of named tensors.

    Args:
        num_envs: Number of parallel environments.
        capacity_per_env: Ring length per environment.
        specs: ``{name: TensorSpec}`` for every stored tensor. Each tensor is stored as
            ``(capacity, num_envs, *spec.shape)``.
        episode_key: Name of the tensor that identifies a step's index within its episode. Used to
            keep sampled windows inside one episode. Must be in ``specs`` and must be an integer dtype.
        storage_dtype: If given, floating tensors are stored in this dtype (e.g. ``torch.bfloat16`` to
            halve memory); they are cast back to float32 when sampled.
        device: Device of the storage. ``"cpu"`` keeps GPU memory free.
    """

    def __init__(
        self,
        num_envs: int,
        capacity_per_env: int,
        specs: dict[str, TensorSpec],
        episode_key: str = "episode_step",
        storage_dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
    ):
        if episode_key not in specs:
            raise ValueError(f"episode_key {episode_key!r} must be one of the stored specs: {sorted(specs)}")
        if not str(specs[episode_key].dtype).startswith("torch.int"):
            raise ValueError(f"{episode_key!r} must have an integer dtype, got {specs[episode_key].dtype}")
        if num_envs <= 0 or capacity_per_env <= 0:
            raise ValueError("num_envs and capacity_per_env must be positive")

        self.num_envs = int(num_envs)
        self.capacity_per_env = int(capacity_per_env)
        self.episode_key = episode_key
        self.device = torch.device(device)
        self.storage_dtype = storage_dtype
        self.specs = dict(specs)

        self._buffers: dict[str, torch.Tensor] = {}
        for name, spec in self.specs.items():
            dtype = spec.dtype
            if storage_dtype is not None and dtype.is_floating_point:
                dtype = storage_dtype
            self._buffers[name] = torch.zeros(
                self.capacity_per_env, self.num_envs, *spec.shape, dtype=dtype, device=self.device
            )

        # ring cursor: index of the newest step; number of steps written so far
        self._cursor = 0
        self._filled = 0

    # ------------------------------------------------------------------ writing

    def add(self, step: dict[str, torch.Tensor]) -> None:
        """Append one step for every env.

        Args:
            step: ``{name: (num_envs, *spec.shape)}`` for every stored tensor.
        """
        for name, buffer in self._buffers.items():
            if name not in step:
                raise KeyError(f"missing tensor {name!r} in the step; expected {sorted(self.specs)}")
            value = step[name]
            if value.shape[0] != self.num_envs:
                raise ValueError(f"{name!r}: expected {self.num_envs} envs on dim 0, got {value.shape[0]}")
            buffer[self._cursor].copy_(value.to(dtype=buffer.dtype, device=buffer.device))

        self._cursor = (self._cursor + 1) % self.capacity_per_env
        self._filled = min(self._filled + 1, self.capacity_per_env)

    def __len__(self) -> int:
        """Number of steps currently stored per env."""
        return self._filled

    def ready(self, min_size: int) -> bool:
        return self._filled >= min_size

    # ------------------------------------------------------------------ sampling

    def _sample_valid_starts(
        self,
        batch_size: int,
        length: int,
        generator: torch.Generator | None,
        require_contiguous: bool,
    ) -> torch.Tensor:
        """Return ring indices of window starts, shape ``(batch_size,)``.

        A start is valid when the whole window lies inside the buffer and (if ``require_contiguous``)
        inside a single episode.
        """
        if self._filled < length:
            raise RuntimeError(f"replay holds {self._filled} steps, cannot sample windows of length {length}")

        episode = self._buffers[self.episode_key]
        # ring position of the oldest valid start so that start + length - 1 is already written
        newest = (self._cursor - 1) % self.capacity_per_env
        starts = []
        attempts = 0
        max_attempts = 20 * batch_size
        while len(starts) < batch_size:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "could not find enough valid window starts; the replay may be too fragmented by episode "
                    "resets for the requested window length"
                )
            # Draw from the written region only. Sampling the whole ring wastes nearly every candidate
            # while the buffer is still filling, which used to yield empty batches (and NaN losses).
            lowest = (self._cursor - self._filled) % self.capacity_per_env
            span = max(self._filled - length + 1, 1)
            cand = (lowest + torch.randint(0, span, (max(batch_size, 64),), generator=generator)) % self.capacity_per_env
            for c in cand.tolist():
                ok = True
                for offset in range(length):
                    idx = (c + offset) % self.capacity_per_env
                    # the slot must be written at least once and not overwritten by a newer lap
                    age = (self._cursor - 1 - idx) % self.capacity_per_env
                    if age >= self._filled:
                        ok = False
                        break
                    if require_contiguous and offset > 0:
                        prev_idx = (c + offset - 1) % self.capacity_per_env
                        # episode_step must increment by exactly one inside a window
                        if int(episode[idx].max()) != int(episode[prev_idx].max()) + 1:
                            ok = False
                            break
                if ok:
                    starts.append(c)
                    if len(starts) >= batch_size:
                        break
        return torch.as_tensor(starts[:batch_size], dtype=torch.long)

    def _gather_window(self, starts: torch.Tensor, length: int, keys: Sequence[str]):
        """Gather ``(B, length, num_envs, ...)`` windows, plus a per-start validity mask.

        Returns:
            ``(windows, mask)`` where ``mask`` has shape ``(B,)``: False means the window crosses an
            episode reset, so the caller should drop or down-weight that sample.
        """
        offsets = torch.arange(length)
        idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)) % self.capacity_per_env  # (B, L)
        wanted = list(dict.fromkeys(list(keys) + [self.episode_key]))
        out = {}
        for key in wanted:
            if key not in self._buffers:
                raise KeyError(f"unknown key {key!r}; stored: {sorted(self._buffers)}")
            buf = self._buffers[key]
            # Advanced indexing on dim 0 keeps the remaining dims: (B, L, num_envs, *spec).
            gathered = buf[idx]
            if self.storage_dtype is not None and gathered.dtype != self.specs[key].dtype:
                gathered = gathered.to(self.specs[key].dtype)
            out[key] = gathered

        # Validity: episode_step must increment by exactly one across each window. All envs advance in
        # lockstep, so the first env decides; a reset inside the window breaks contiguity.
        episode = out[self.episode_key]
        while episode.dim() > 2:
            episode = episode[..., 0]
        expected = episode[:, :1] + offsets.to(episode.device).unsqueeze(0)
        mask = (episode == expected).all(dim=-1)  # (B,)

        return {k: out[k] for k in keys}, mask


    def sample_windows(
        self,
        batch_size: int,
        length: int,
        keys: Sequence[str],
        generator: torch.Generator | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Sample contiguous windows for world-model fitting.

        Returns:
            ``(batch, mask)``: ``batch[k]`` has shape ``(B, length, num_envs, *spec)`` when ``env_ids`` is
            None (all envs of the sampled start index), else ``(B, length, *spec)`` for the chosen env;
            ``mask`` is ``(B, length)``.
        """
        starts = self._sample_valid_starts(batch_size, length, generator, require_contiguous=True)
        windows, mask = self._gather_window(starts, length, keys)
        if env_ids is not None:
            # (B, L, E, ...) -> pick one env per sample
            env_sel = torch.as_tensor(env_ids) if not torch.is_tensor(env_ids) else env_ids
            if env_sel.numel() == 1:
                env_sel = env_sel.expand(batch_size)
            windows = {k: v[torch.arange(batch_size), :, env_sel] for k, v in windows.items()}
        return windows, mask

    def sample_starts(
        self,
        batch_size: int,
        keys: Sequence[str],
        generator: torch.Generator | None = None,
        history: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Sample single start states, plus ``history - 1`` earlier frames per start.

        Returns:
            ``{key: (B, history, ...)}`` with the newest frame last. Frames older than the episode start
            are filled by repeating the oldest valid frame, so the shape is static; use the returned
            ``history_mask`` to ignore them.
        """
        if history < 1:
            raise ValueError("history must be >= 1")
        starts = self._sample_valid_starts(batch_size, history, generator, require_contiguous=False)
        windows, mask = self._gather_window(starts, history, list(keys) + [self.episode_key])
        # One env per start: gathered tensors are (B, history, E, *spec).
        env_sel = torch.randint(0, self.num_envs, (batch_size,), generator=generator)
        batch_idx = torch.arange(batch_size)
        out = {k: windows[k][batch_idx, :, env_sel] for k in keys}
        out["history_mask"] = mask
        # which environment each start came from: callers that need per-env side data (e.g. the
        # reference window) align on this
        out["env_index"] = env_sel
        return out

    def observed_range(self, keys: Sequence[str]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Per-key ``(min, max)`` over the written region of the buffer.

        Buffers are ``(capacity_per_env, num_envs, *spec)``, so the reduction covers both time *and*
        environment: an imagined rollout may start from any env's state, so a single global bound is the
        only one a per-sample clamp can use.

        A learned dynamics model is only trustworthy near the states it was fit on; rolling it out past
        that range produces finite but meaningless numbers that a critic will happily extrapolate from.
        Callers clamp to these bounds so imagination stays within the data.

        Returns:
            ``{key: (min, max)}`` with each bound shaped like the spec (dim 0 is the time/env axis).
        """
        filled = self._filled
        if filled == 0:
            raise RuntimeError("cannot compute an observed range from an empty replay")
        out = {}
        for key in keys:
            buf = self._buffers[key]
            # ring positions of the written steps
            idx = (self._cursor - filled + torch.arange(filled)) % self.capacity_per_env
            window = buf[idx.to(buf.device)]  # (filled, num_envs, *spec)
            dims = tuple(range(window.dim() - (buf.dim() - 2)))
            out[key] = (window.amin(dim=dims), window.amax(dim=dims))
        return out

    # ------------------------------------------------------------------ checkpointing

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        state: dict[str, torch.Tensor | int] = {
            "cursor": int(self._cursor),
            "filled": int(self._filled),
            "num_envs": int(self.num_envs),
            "capacity_per_env": int(self.capacity_per_env),
        }
        for name, buffer in self._buffers.items():
            state[f"buffer/{name}"] = buffer
        return state

    def load_state_dict(self, state: dict) -> None:
        if int(state["num_envs"]) != self.num_envs or int(state["capacity_per_env"]) != self.capacity_per_env:
            raise ValueError(
                "replay buffer shape mismatch: checkpoint was "
                f"(num_envs={state['num_envs']}, capacity={state['capacity_per_env']}), "
                f"current is (num_envs={self.num_envs}, capacity={self.capacity_per_env})"
            )
        missing = [n for n in self._buffers if f"buffer/{n}" not in state]
        if missing:
            raise KeyError(f"checkpoint is missing replay buffers: {missing}")
        for name, buffer in self._buffers.items():
            saved = state[f"buffer/{name}"]
            if saved.shape != buffer.shape:
                raise ValueError(f"replay buffer {name!r} shape mismatch: {tuple(saved.shape)} vs {tuple(buffer.shape)}")
            buffer.copy_(saved.to(dtype=buffer.dtype, device=buffer.device))
        self._cursor = int(state["cursor"])
        self._filled = int(state["filled"])

    def to(self, device: str | torch.device) -> "ReplayStorage":
        for name in self._buffers:
            self._buffers[name] = self._buffers[name].to(device)
        self.device = torch.device(device)
        return self

    def extra_repr(self) -> str:
        return (
            f"num_envs={self.num_envs}, capacity_per_env={self.capacity_per_env}, "
            f"filled={self._filled}, keys={sorted(self.specs)}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"
