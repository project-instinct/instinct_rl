from collections import namedtuple

import torch

from instinct_rl.utils import split_and_pad_trajectories
from instinct_rl.utils.buffer import buffer_from_example, buffer_method
from instinct_rl.utils.collections import is_namedarraytuple


class AmpStorage:
    class Transition:
        def __init__(self):
            self.actor_states = None
            self.reference_states = None
            self.hidden_states = None
            self.dones = None

        def clear(self):
            self.__init__()

    MiniBatch = namedtuple(
        "MiniBatch",
        [
            "actor_states",
            "reference_states",
            "hidden_states",
            "masks",
        ],
    )

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        actor_state_shape,
        reference_state_shape,
        device="cpu",
    ):
        self.device = device

        self.actor_states_shape = actor_state_shape
        self.reference_states_shape = reference_state_shape

        self.actor_states = torch.zeros(num_transitions_per_env, num_envs, *actor_state_shape, device=self.device)
        self.reference_states = torch.zeros(
            num_transitions_per_env, num_envs, *reference_state_shape, device=self.device
        )
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.actor_states[self.step].copy_(transition.actor_states)
        self.reference_states[self.step].copy_(transition.reference_states)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        """Assuming hidden_states is a torch tensor or a namedarraytuple of torch tensor"""
        if hidden_states is None:
            return
        if is_namedarraytuple(hidden_states):
            try:
                leading_dims = hidden_states.get_leading_dims()
            except AttributeError as e:
                if "None" in str(e):
                    return

        # initialize if needed
        if self.saved_hidden_states is None:
            self.saved_hidden_states = buffer_from_example(hidden_states, self.observations.shape[0])
        # copy the states
        self.saved_hidden_states[self.step] = hidden_states

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        T_indices = (indices // self.num_envs).to(torch.long)
        B_indices = (indices % self.num_envs).to(torch.long)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                T_idx = T_indices[start:end]
                B_idx = B_indices[start:end]

                yield self.get_minibatch_from_selection(T_idx, B_idx)

    # for RNNs only
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        self._padded_actor_trajectories, self._trajectory_masks = split_and_pad_trajectories(
            self.actor_states, self.dones
        )
        self._padded_reference_trajectories, _ = split_and_pad_trajectories(self.reference_states, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                yield self.get_minibatch_from_selection(
                    slice(None),
                    slice(start, stop),
                    padded_B_slice=slice(first_traj, last_traj),
                    prev_done_mask=last_was_done,
                )

                first_traj = last_traj

    def get_minibatch_from_selection(self, T_select, B_select, padded_B_slice=None, prev_done_mask=None):
        """Extract minibatch based on selected indices/slice.
        An independent method allows override by subclasses.
        Args:
            - padded_B_slice: For recurrent trajectories, the observations are already expanded and padded with zeros.
            - prev_done_mask: For recurrent trajectories,
        Outputs:
            - MiniBatch:
                only batch dimension if not padded_B_slice (non-recurrent case)
                with time, batch dimension if padded_B_slice (recurrent case)
        """
        if padded_B_slice is None:
            actor_batch = self.actor_states[T_select, B_select]
            reference_batch = self.reference_states[T_select, B_select]
            hid_batch = None
            state_mask_batch = None
        else:
            actor_batch = self._padded_actor_trajectories[T_select, padded_B_slice]
            reference_batch = self._padded_reference_trajectories[T_select, padded_B_slice]

            # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
            # then take only time steps after dones (flattens num envs and time dimensions),
            # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
            prev_done_mask = prev_done_mask.permute(1, 0)  # (T, B) -> (B, T)
            hid_batch = buffer_method(
                buffer_method(
                    buffer_method(self.saved_hidden_states, "permute", 2, 0, 1, 3)[prev_done_mask][padded_B_slice],
                    "transpose",
                    1,
                    0,
                ),
                "contiguous",
            )
            state_mask_batch = self._trajectory_masks[T_select, padded_B_slice]

        return AmpStorage.MiniBatch(
            actor_batch,
            reference_batch,
            hid_batch,
            state_mask_batch,
        )
