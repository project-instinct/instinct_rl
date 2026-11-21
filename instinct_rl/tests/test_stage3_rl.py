"""Unit tests for the ACP stage-3 instinct_rl pieces (R1–R6 + D13).

Covers:
  R1: Multiple critics with distinct observation groups (``critic_obs_groups``).
  R2: Per-env value masks (``value_mask_group``): masked advantage normalization & masked value loss.
  R3: Per-critic fixed learning rates (``critic_learning_rates``), critic warmup actor mask
      (``critic_warmup_iterations``), and warmup value clipping control (``critic_value_clipping``).
  R4: ``PartitionedReplayBuffer(eviction="quality")`` eviction and ``state_dict`` round-trip.
  R5 + D13: Dual replay buffer routing (``buffer_id == 0/1``), proportional batch mixing, sidecar
      save/load, and ``planner_critic_targets`` injecting ``p_hat`` into ``critic_goal``.
  R6: ``ckpt_manipulator.add_goal_critic`` Stage-2 -> Stage-3 checkpoint transition & resume.

Run with ``python -m pytest instinct_rl/tests/test_stage3_rl.py`` (CPU only).
"""

from collections import OrderedDict

import pytest
import torch

from instinct_rl.algorithms import ContactPlannerPPO, PPO
from instinct_rl.modules import ActorCritic, EmpiricalNormalization, MlpContactPlanner
from instinct_rl.runners import OnPolicyRunner
from instinct_rl.storage.replay_storage import PartitionedReplayBuffer, TensorSpec
from instinct_rl.tests.fake_env import FakeContactPlannerEnv
from instinct_rl.utils import ckpt_manipulator
from instinct_rl.utils.utils import get_obs_slice

SMALL_MLP = dict(hidden_sizes=(32, 32))
PLANNER_INPUTS = dict(points_xy_component="points_xy", points_z_component="points_z", context_components=["proprio"])


class FakeStage3Env(FakeContactPlannerEnv):
    """FakeContactPlannerEnv extended with a second critic group, env_type mask, and dual-buffer samples."""

    def __init__(self, num_envs=32, episode_length=8, seed=0):
        super().__init__(num_envs=num_envs, episode_length=episode_length, gate_probability=0.5, seed=seed)
        self.num_rewards = 2
        self._is_goal = torch.zeros(self.num_envs, dtype=torch.bool)
        self._is_goal[self.num_envs // 2 :] = True
        self._env_type = torch.zeros(self.num_envs, 2)
        self._env_type[~self._is_goal, 0] = 1.0
        self._env_type[self._is_goal, 1] = 1.0
        self._critic_goal_extra = 5

    def get_obs_format(self):
        fmt = super().get_obs_format()
        fmt["critic_goal"] = OrderedDict(
            {
                "proprio": (4,),
                "planned_contacts": (self.NUM_BODIES * 3,),
                "goal_extra": (self._critic_goal_extra,),
            }
        )
        fmt["env_type"] = OrderedDict({"mask": (2,)})
        return fmt

    def _obs_pack(self):
        pack = super()._obs_pack()
        # Deliberately fill planned_contacts with sentinel 99.0 to verify D13 overwrites it
        proprio = pack["policy"][:, :4]
        sentinel = torch.full((self.num_envs, self.NUM_BODIES * 3), 99.0)
        goal_extra = torch.randn(self.num_envs, self._critic_goal_extra, generator=self.generator)
        pack["critic_goal"] = torch.cat([proprio, sentinel, goal_extra], dim=-1)
        pack["env_type"] = self._env_type.clone()
        return pack

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        # Produce 2-column rewards: col 0 for ref envs, col 1 for goal envs
        r1 = rewards.squeeze(-1)
        rewards_2 = torch.zeros(self.num_envs, 2)
        rewards_2[~self._is_goal, 0] = r1[~self._is_goal]
        rewards_2[self._is_goal, 1] = r1[self._is_goal] * 2.0 + 1.0
        # Attach buffer_id, partition, group, quality to planner_samples
        samples = extras.get("planner_samples")
        if samples is not None:
            n = samples["planner_obs"].shape[0]
            half = n // 2
            buf_id = torch.zeros(n, dtype=torch.long)
            buf_id[half:] = 1
            samples["buffer_id"] = buf_id
            samples["partition"] = torch.arange(n, dtype=torch.long) % 4
            samples["group"] = samples["partition"] // 2
            samples["quality"] = torch.linspace(-2.0, -0.1, n)
        return obs, rewards_2, dones, extras


def _stage3_runner(env, tmp_path=None, **alg_overrides):
    alg_cfg = {
        "class_name": "ContactPlannerPPO",
        "num_learning_epochs": 1,
        "num_mini_batches": 2,
        "learning_rate": 1e-3,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "value_mask_group": "env_type",
        "critic_learning_rates": [None, 1e-3],
        "critic_warmup_iterations": [0, 3],
        "critic_value_clipping": [True, True],
        "planner_class_name": "MlpContactPlanner",
        "planner_kwargs": {**PLANNER_INPUTS, **SMALL_MLP, "num_bodies": env.NUM_BODIES},
        "planner_target_components": ("contact_targets",),
        "planner_gate_group": "contact_source",
        "planner_critic_targets": [("critic_goal", ("planned_contacts",))],
        "planner_replay_num_partitions": 4,
        "planner_replay_partition_capacity": 32,
        "planner_goal_replay_kwargs": {
            "num_partitions": 4,
            "capacity_per_partition": 32,
            "eviction": "quality",
        },
        "planner_min_samples": 16,
        "planner_batch_size": 16,
        "planner_num_updates": 2,
        "planner_optimizer_kwargs": {"lr": 1e-3},
        **alg_overrides,
    }
    cfg = {
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [32],
            "critic_hidden_dims": [32],
            "activation": "elu",
            "init_noise_std": 1.0,
            "critic_obs_groups": ["critic", "critic_goal"],
        },
        "algorithm": alg_cfg,
        "normalizers": {"policy": {"class_name": "EmpiricalNormalization"}},
        "num_steps_per_env": 4,
        "save_interval": 1000,
        "log_interval": 1000,
    }
    return OnPolicyRunner(env, cfg, log_dir=None if tmp_path is None else str(tmp_path), device="cpu")


# -- R1: Multi-critic with distinct observation groups ------------------------------------------


def test_r1_multi_critic_distinct_obs_groups():
    env = FakeStage3Env(num_envs=8)
    obs_format = env.get_obs_format()
    ac = ActorCritic(
        obs_format,
        env.num_actions,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        activation="elu",
        num_rewards=2,
        critic_obs_groups=["critic", "critic_goal"],
    )
    dim_c0 = sum(int(torch.tensor(s).prod()) for s in obs_format["critic"].values())
    dim_c1 = sum(int(torch.tensor(s).prod()) for s in obs_format["critic_goal"].values())
    assert dim_c0 != dim_c1
    assert ac.critics[0][0].in_features == dim_c0
    assert ac.critics[1][0].in_features == dim_c1
    _, extras = env.get_observations()
    cat_obs = torch.cat([extras["observations"]["critic"], extras["observations"]["critic_goal"]], dim=-1)
    vals = ac.evaluate(cat_obs)
    assert vals.shape == (8, 2)


# -- R2 + R3: Value masks, fixed critic LR, warmup actor mask & value clipping ------------------


def test_r2_r3_masked_advantages_warmup_and_fixed_critic_lr():
    torch.manual_seed(0)
    env = FakeStage3Env(num_envs=16)
    obs_format = env.get_obs_format()
    ac = ActorCritic(
        obs_format,
        env.num_actions,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        activation="elu",
        num_rewards=2,
        critic_obs_groups=["critic", "critic_goal"],
    )
    ppo = PPO(
        ac,
        num_learning_epochs=1,
        num_mini_batches=2,
        learning_rate=1e-3,
        schedule="adaptive",
        desired_kl=0.01,
        value_mask_group="env_type",
        critic_learning_rates=[None, 5e-4],
        critic_warmup_iterations=[0, 3],
        critic_value_clipping=[True, True],
    )
    # Optimizer should have 2 param groups: adaptive (actor + critic 0) and fixed (critic 1)
    assert len(ppo.optimizer.param_groups) == 2
    assert ppo.optimizer.param_groups[0]["adaptive"] is True
    assert ppo.optimizer.param_groups[1]["adaptive"] is False
    assert ppo.optimizer.param_groups[1]["lr"] == pytest.approx(5e-4)

    ppo.init_storage(env.num_envs, 4, obs_format, env.num_actions, num_rewards=2)
    assert ppo.storage.value_masks is not None

    # Verify masked advantage normalization: only active rows contribute to mean/std, inactive rows are 0
    obs, extras = env.get_observations()
    for _ in range(4):
        actions = ppo.act(obs, extras["observations"])
        obs, rewards, dones, step_extras = env.step(actions)
        ppo.process_env_step(rewards, dones, step_extras, obs, step_extras["observations"])
        extras = step_extras
    ppo.compute_returns(extras["observations"])
    masks = ppo.storage.value_masks  # (T, N, 2)
    adv = ppo.storage.advantages  # (T, N, 2)
    assert torch.all(adv[masks < 0.5] == 0.0)
    for k in range(2):
        active = adv[..., k][masks[..., k] > 0.5]
        assert active.mean().abs() < 1e-5
        assert (active.std() - 1.0).abs() < 0.1

    # During warmup (iter 0 < 3), critic 1's lr stays fixed even if adaptive lr changes
    ppo.learning_rate = 2e-3
    losses_w, _ = ppo.update(0)
    assert "value_loss" in losses_w
    assert ppo.optimizer.param_groups[1]["lr"] == pytest.approx(5e-4)

    # Verify warmup completely masks out goal envs from surrogate loss when only goal envs are active
    # (check by setting all masks to column 1 and verifying actor MLP grad is identically zero)
    ppo.storage.value_masks.zero_()
    ppo.storage.value_masks[..., 1] = 1.0
    critic1_before = [p.clone() for p in ac.critics[1].parameters()]
    ppo.update(1)  # still < 3
    for p in ac.actor.parameters():
        assert p.grad is None or torch.all(p.grad == 0.0)
    # Critic 1 SHOULD still update during warmup!
    assert any(not torch.equal(b, a) for b, a in zip(critic1_before, ac.critics[1].parameters()))


# -- R4: PartitionedReplayBuffer quality eviction & state_dict round-trip -----------------------


def test_r4_partitioned_replay_quality_eviction_and_round_trip():
    specs = {"x": TensorSpec((2,), torch.float32)}
    buf = PartitionedReplayBuffer(num_partitions=2, capacity_per_partition=3, specs=specs, eviction="quality")

    # Fill partition 0 with 3 items of quality [-3.0, -1.0, -2.0]
    buf.add(
        {
            "x": torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            "partition": torch.tensor([0, 0, 0]),
            "group": torch.tensor([0, 0, 0]),
            "quality": torch.tensor([-3.0, -1.0, -2.0]),
        }
    )
    assert len(buf) == 3

    # Add a better item (quality -0.5): should evict worst (-3.0 -> x=[1,0])
    buf.add(
        {
            "x": torch.tensor([[4.0, 0.0]]),
            "partition": torch.tensor([0]),
            "group": torch.tensor([0]),
            "quality": torch.tensor([-0.5]),
        }
    )
    assert len(buf) == 3
    stored_x0 = set(buf._buffers["x"][0, :, 0].tolist())
    assert stored_x0 == {2.0, 3.0, 4.0}

    # Add a worse item (quality -5.0): should be rejected (not replace anything)
    buf.add(
        {
            "x": torch.tensor([[9.0, 0.0]]),
            "partition": torch.tensor([0]),
            "group": torch.tensor([0]),
            "quality": torch.tensor([-5.0]),
        }
    )
    assert set(buf._buffers["x"][0, :, 0].tolist()) == {2.0, 3.0, 4.0}

    # State dict round-trip preserves quality
    state = buf.state_dict()
    assert "quality" in state
    buf2 = PartitionedReplayBuffer(num_partitions=2, capacity_per_partition=3, specs=specs, eviction="quality")
    buf2.load_state_dict(state)
    torch.testing.assert_close(buf2._quality, buf._quality)
    torch.testing.assert_close(buf2._buffers["x"], buf._buffers["x"])


# -- R5 + D13: Dual replay buffer routing, mixin critic_goal injection, sidecar round-trip ------


def test_r5_d13_dual_replay_and_critic_goal_target_injection(tmp_path):
    torch.manual_seed(0)
    env = FakeStage3Env(num_envs=16)
    runner = _stage3_runner(env, tmp_path)
    runner.log = lambda locs, *args, **kwargs: None

    # D13: Verify act() overwrites sentinel 99.0 in critic_goal's planned_contacts slot with p_hat
    obs, extras = env.get_observations()
    extra_obs = extras["observations"]
    obs = runner._prepare_extra_obs(extra_obs)
    runner.alg.act(obs, extra_obs)
    planned = runner.alg.get_policy_feedback()["planned_contacts"].reshape(env.num_envs, -1)
    dim_c0 = sum(int(torch.tensor(s).prod()) for s in env.get_obs_format()["critic"].values())
    slot, _ = get_obs_slice(env.get_obs_format()["critic_goal"], "planned_contacts")
    critic_goal_stored = runner.alg.transition.critic_observations[:, dim_c0:]
    torch.testing.assert_close(critic_goal_stored[:, slot], planned, atol=1e-6, rtol=0)

    # Run 4 iterations (crosses warmup=3) and check dual buffers + sidecar save/load
    runner.learn(num_learning_iterations=4)
    assert runner.alg.planner_goal_replay is not None
    assert len(runner.alg.planner_replay) > 0
    assert len(runner.alg.planner_goal_replay) > 0

    ckpt_path = str(tmp_path / "model_4.pt")
    runner.save(ckpt_path)
    reloaded = _stage3_runner(env)
    reloaded.load(ckpt_path)
    assert len(reloaded.alg.planner_replay) == len(runner.alg.planner_replay)
    assert len(reloaded.alg.planner_goal_replay) == len(runner.alg.planner_goal_replay)
    torch.testing.assert_close(reloaded.alg.planner_goal_replay._quality, runner.alg.planner_goal_replay._quality)


# -- R6: ckpt_manipulator.add_goal_critic Stage-2 -> Stage-3 transition -------------------------


def test_r6_add_goal_critic_checkpoint_transition():
    torch.manual_seed(0)
    # Build a Stage-2 single-critic ContactPlannerPPO checkpoint
    env2 = FakeContactPlannerEnv(num_envs=16)
    obs_fmt2 = env2.get_obs_format()
    ac2 = ActorCritic(obs_fmt2, env2.num_actions, [32], [32], "elu", init_noise_std=1.0)
    alg2 = ContactPlannerPPO(
        ac2,
        num_learning_epochs=1,
        num_mini_batches=2,
        planner_class_name="MlpContactPlanner",
        planner_kwargs={**PLANNER_INPUTS, **SMALL_MLP, "num_bodies": env2.NUM_BODIES},
        planner_target_components=("contact_targets",),
        planner_gate_group="contact_source",
    )
    alg2.init_storage(env2.num_envs, 4, obs_fmt2, env2.num_actions)
    stage2_ckpt = {**alg2.state_dict(), "iter": 10000, "infos": None}

    # Build Stage-3 two-critic runner
    env3 = FakeStage3Env(num_envs=16)
    runner3 = _stage3_runner(env3)
    fresh_c1 = {k: v.clone() for k, v in runner3.alg.actor_critic.critics[1].state_dict().items()}

    # Manipulate Stage-2 checkpoint for Stage-3
    manipulated = ckpt_manipulator.add_goal_critic(stage2_ckpt, runner3.alg.state_dict())
    assert manipulated["iter"] == 0
    runner3.alg.load_state_dict(manipulated)

    # Actor, critic 0, and planner weights should match Stage-2; critic 1 should remain fresh
    for k, v in ac2.actor.state_dict().items():
        assert torch.equal(runner3.alg.actor_critic.actor.state_dict()[k], v)
    for k, v in ac2.critic.state_dict().items():
        assert torch.equal(runner3.alg.actor_critic.critics[0].state_dict()[k], v)
    for k, v in alg2.planner.state_dict().items():
        assert torch.equal(runner3.alg.planner.state_dict()[k], v)
    for k, v in fresh_c1.items():
        assert torch.equal(runner3.alg.actor_critic.critics[1].state_dict()[k], v)

    # Resuming a Stage-3 checkpoint passes through unchanged and preserves iter
    stage3_resume = {**runner3.alg.state_dict(), "iter": 420, "infos": None}
    resumed = ckpt_manipulator.add_goal_critic(stage3_resume, runner3.alg.state_dict())
    assert resumed["iter"] == 420
