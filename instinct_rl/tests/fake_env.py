"""Minimal environment + imagination interface for algorithm unit tests.

A toy MDP small enough to run on CPU in a second, used to check that an algorithm's plumbing works
without starting Isaac Sim. The dynamics are linear (a damped second-order system) and the
imagination interface is *analytic*, so the imagined reward curve has a known optimum: a policy that
drives the state to zero gets a higher return than one that does not, which lets a test assert that
learning actually happened rather than merely that no exception was raised.

Not a benchmark: it exists to catch interface mistakes (wrong observation group, wrong sign of the
reward, forgetting to reset, a replay buffer that never sees a done flag).
"""

from collections import OrderedDict

import numpy as np
import torch

STATE_SEGMENTS = OrderedDict(
    {
        "state_pos": (2,),
        "state_vel": (2,),
        "joint_pos": (2,),
    }
)
CONTEXT_SEGMENTS = OrderedDict({"friction": (1,)})
OBS_FORMAT = OrderedDict(
    {
        "policy": OrderedDict(
            {
                "state_pos": (2,),
                "state_vel": (2,),
                "joint_pos": (2,),
                "friction": (1,),
            }
        ),
        "critic": OrderedDict(
            {
                "state_pos": (2,),
                "state_vel": (2,),
                "joint_pos": (2,),
                "friction": (1,),
            }
        ),
    }
)
NUM_ACTIONS = 2


class ToyImagination:
    """Analytic stand-in for a task's differentiable MDP."""

    state_format = STATE_SEGMENTS
    context_format = CONTEXT_SEGMENTS
    history_length = 0
    history_terms = ()

    def __init__(self, device="cpu"):
        self.device = device

    # -- observation assembly -------------------------------------------------
    def policy_obs(self, state, context, prev_action, ref_index, history_state=None, history_action=None):
        if context is None:
            raise ValueError("ToyImagination requires a context vector")
        return torch.cat([state, context], dim=-1)

    def add_obs_noise(self, obs):
        return obs

    def critic_obs(self, state, context, ref_index, prev_action=None):
        return self.policy_obs(state, context, prev_action, ref_index)

    # -- imagined MDP --------------------------------------------------------
    def reward(self, state, action, prev_action, ref_index):
        pos = state[..., 0:2]
        vel = state[..., 2:4]
        joint = state[..., 4:6]
        # Quadratic cost: the optimal policy drives everything to zero.
        cost = (pos**2).sum(-1) + 0.1 * (vel**2).sum(-1) + 0.1 * (joint**2).sum(-1)
        cost = cost + 0.01 * (action**2).sum(-1) + 0.01 * ((action - prev_action) ** 2).sum(-1)
        return -cost.unsqueeze(-1) * 0.02  # per-step, scaled like a 50 Hz task

    def termination_prob(self, state, ref_index):
        pos = state[..., 0:2]
        # Softened: a hard flag would cut the gradient.
        return torch.sigmoid(((pos**2).sum(-1) - 4.0) * 5.0).unsqueeze(-1)

    def advance(self, ref_index):
        return None

    def valid_start(self, ref_index, horizon):
        raise NotImplementedError("not used by the tests")


class ToyAnalyticWorld:
    """The ground-truth dynamics the world model must learn.

    ``x_{t+1} = x_t + dt * (A x_t + B a_t)`` for the position/velocity block, and the joint block is
    driven directly by the action, so a small MLP can fit it in a few dozen steps.
    """

    def __init__(self, dt=0.02, device="cpu"):
        self.dt = dt
        self.device = device

    def step(self, state, action, context):
        pos = state[..., 0:2]
        vel = state[..., 2:4]
        joint = state[..., 4:6]
        friction = context[..., 0:1] if context is not None else torch.ones_like(pos[..., :1])
        accel = 4.0 * action - 2.0 * vel * friction
        next_pos = pos + self.dt * vel
        next_vel = vel + self.dt * accel
        next_joint = joint + self.dt * action
        return torch.cat([next_pos, next_vel, next_joint], dim=-1)

    def sample(self, batch_size, episode_step=0, generator=None):
        state = torch.randn(batch_size, 6, generator=generator) * 0.3
        context = torch.rand(batch_size, 1, generator=generator) * 0.8 + 0.3
        return state, context


class FakeVecEnv:
    """Minimal vectorised env returning Instinct-RL observation groups.

    Enough surface for a training loop: ``get_observations``, ``step``, ``num_envs``,
    ``num_rewards``, ``max_episode_length``, ``episode_length_buf`` and ``get_algorithm_inputs``.
    """

    def __init__(self, num_envs=16, episode_length=24, seed=0, device="cpu"):
        self.num_envs = int(num_envs)
        self.num_rewards = 1
        self.max_episode_length = int(episode_length)
        self.device = torch.device(device)
        self.world = ToyAnalyticWorld(device=device)
        self.imagination = ToyImagination(device=device)
        self.generator = torch.Generator().manual_seed(seed)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.state, self.context = self.world.sample(self.num_envs, generator=self.generator)
        self.state = self.state.to(self.device)
        self.context = self.context.to(self.device)

    # -- Instinct-RL env surface ---------------------------------------------
    def get_algorithm_inputs(self):
        return {"imagination": self.imagination}

    def get_observations(self):
        obs = self.imagination.policy_obs(self.state, self.context, None, None)
        return obs, {"observations": self._obs_pack()}

    def _obs_pack(self):
        return {
            "policy": self.imagination.policy_obs(self.state, self.context, None, None),
            "critic": self.imagination.critic_obs(self.state, self.context, None),
            "state_pos": self.state[:, 0:2],
            "state_vel": self.state[:, 2:4],
            "joint_pos": self.state[:, 4:6],
            "friction": self.context[:, 0:1],
        }

    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, device=self.device)
        actions = actions.to(self.device).reshape(self.num_envs, NUM_ACTIONS)
        reward = self.imagination.reward(self.state, actions, torch.zeros_like(actions), None)
        next_state = self.world.step(self.state, actions, self.context)
        prob = self.imagination.termination_prob(next_state, None).squeeze(-1)  # (num_envs,)
        noise = torch.rand(self.num_envs, generator=self.generator)
        done = (noise < prob.cpu()).to(self.device)

        self.episode_length_buf += 1
        time_out = self.episode_length_buf >= self.max_episode_length
        done = done | time_out

        self.state = next_state.clone()
        final_observations = None
        if bool(done.any()):
            # like IsaacLab's ``compute_final_obs``: the observations the finished episodes ended in
            final_observations = {k: v.clone() for k, v in self._obs_pack().items()}
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            new_state, new_context = self.world.sample(len(ids), generator=self.generator)
            self.state[ids] = new_state.to(self.device)
            self.context[ids] = new_context.to(self.device)
            self.episode_length_buf[ids] = 0

        infos = {"observations": self._obs_pack(), "time_outs": time_out.to(self.device)}
        if final_observations is not None:
            infos["final_observations"] = final_observations
        return (
            self.imagination.policy_obs(self.state, self.context, None, None),
            reward,
            done.float(),
            infos,
        )
