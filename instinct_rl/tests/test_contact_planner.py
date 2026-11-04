"""Unit tests for the ACP stage-2 pieces: attention blocks, contact planners, FIFO replay and ContactPlannerPPO.

Run with ``python -m pytest instinct_rl/tests/test_contact_planner.py``; CPU only, about a minute.
"""

import os
from collections import OrderedDict

import numpy as np
import pytest
import torch

from instinct_rl.algorithms import PPO, ContactPlannerPPO
from instinct_rl.modules import (
    ActorCritic,
    AttentionContactPlanner,
    CrossAttention,
    EmpiricalNormalization,
    FiLM,
    MlpContactPlanner,
)
from instinct_rl.runners import OnPolicyRunner
from instinct_rl.storage.replay_storage import FifoReplayBuffer, TensorSpec
from instinct_rl.tests.fake_env import FakeContactPlannerEnv
from instinct_rl.utils import ckpt_manipulator
from instinct_rl.utils.utils import get_obs_slice

SMALL_ATTENTION = dict(
    d_model=16,
    num_heads=4,
    point_embedding_size=8,
    cnn_channels=(4, 8),
    cnn_kernel_size=3,
    terrain_context_sizes=(16, 16),
    encoder_hidden_sizes=(32,),
    film_hidden_sizes=(16,),
    occupancy_hidden_sizes=(32,),
)
SMALL_MLP = dict(hidden_sizes=(64, 64))
PLANNER_INPUTS = dict(points_xy_component="points_xy", points_z_component="points_z", context_components=["proprio"])


def _num_params(module):
    return sum(p.numel() for p in module.parameters())


def _planner(cls, env=None, **kwargs):
    env = env or FakeContactPlannerEnv()
    size = SMALL_ATTENTION if cls is AttentionContactPlanner else SMALL_MLP
    return cls(
        input_segments=env.get_obs_format()["planner"],
        num_bodies=env.NUM_BODIES,
        **PLANNER_INPUTS,
        **{**size, **kwargs},
    )


# -- attention blocks ----------------------------------------------------------------------------


def test_film_broadcasts_and_starts_near_identity():
    film = FiLM(cond_size=6, feature_size=5, hidden_sizes=(8,))
    x, cond = torch.randn(3, 1, 5), torch.randn(4, 6)
    assert film(x, cond).shape == (3, 4, 5)
    for p in film.generator.parameters():
        torch.nn.init.zeros_(p)
    torch.testing.assert_close(film(x, cond), x.expand(3, 4, 5))


def test_cross_attention_shapes_weights_and_parameter_layout():
    attention = CrossAttention(query_size=12, kv_size=12, d_model=12, num_heads=4)
    out, log_weights = attention(torch.randn(2, 3, 12), torch.randn(2, 7, 12))
    assert out.shape == (2, 3, 12) and log_weights.shape == (2, 4, 3, 7)
    torch.testing.assert_close(log_weights.logsumexp(-1), torch.zeros(2, 4, 3), atol=1e-6, rtol=0)
    assert _num_params(attention) == _num_params(torch.nn.MultiheadAttention(12, 4))
    with pytest.raises(ValueError):
        CrossAttention(12, 12, 10, 4)


# -- planners ------------------------------------------------------------------------------------


def _paper_segments():
    # the paper's 23-DoF planner input: 5 x 75 proprioception + 3 goal, 17 x 11 points
    return OrderedDict({"xy": (11, 17, 2), "z": (11, 17), "proprio": (375,), "goal": (3,)})


def test_planners_reproduce_the_paper_parameter_counts():
    kwargs = dict(points_xy_component="xy", points_z_component="z", context_components=["proprio", "goal"])
    assert _num_params(AttentionContactPlanner(_paper_segments(), **kwargs)) == 786_993
    assert _num_params(MlpContactPlanner(_paper_segments(), **kwargs)) == 833_322


@pytest.mark.parametrize("cls", [AttentionContactPlanner, MlpContactPlanner])
def test_planner_shapes_and_occupancy_gating(cls):
    env = FakeContactPlannerEnv(num_envs=5)
    planner = _planner(cls, env)
    obs = env.get_observations()[1]["observations"]["planner"]
    points, log_affordance, occupancy_logits = planner(obs)
    num_points = env.GRID[0] * env.GRID[1]
    assert points.shape == (5, num_points, 3)
    assert log_affordance.shape == (5, env.NUM_BODIES, num_points) and occupancy_logits.shape == (5, env.NUM_BODIES)
    torch.testing.assert_close(log_affordance.logsumexp(-1), torch.zeros(5, env.NUM_BODIES), atol=1e-5, rtol=0)
    positions, occupied = planner.predict(obs)
    assert positions.shape == (5, env.NUM_BODIES, 3) and planner.output_size == env.NUM_BODIES * 3
    assert torch.all(positions[~occupied] == 0)
    # every kept target is one of the observed points (k = 1)
    kept = positions[occupied]
    owners = occupied.nonzero()[:, 0]
    assert torch.all((points[owners] - kept.unsqueeze(1)).norm(dim=-1).min(dim=-1).values < 1e-6)


def test_top_k_decoding_and_threshold():
    planner = _planner(MlpContactPlanner, top_k=2, temperature=0.5, occupancy_threshold=0.5)
    points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])  # (1, 3, 3)
    planner.num_points = 3
    log_affordance = torch.log(torch.tensor([[[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]]]))
    positions, occupied = planner.decode(points, log_affordance, torch.tensor([[2.0, -2.0]]), top_k=2)
    w = torch.softmax(torch.log(torch.tensor([0.6, 0.3])) / 0.5, dim=0)
    torch.testing.assert_close(positions[0, 0], w[0] * points[0, 1] + w[1] * points[0, 2])
    assert occupied.tolist() == [[True, False]] and torch.all(positions[0, 1] == 0)
    top1, _ = planner.decode(points, log_affordance, torch.tensor([[2.0, 2.0]]), top_k=1)
    torch.testing.assert_close(top1[0], points[0, [1, 0]])


def test_affordance_loss_ignores_invalid_targets():
    env = FakeContactPlannerEnv(num_envs=8)
    planner = _planner(AttentionContactPlanner, env).eval()
    obs = env.get_observations()[1]["observations"]["planner"]
    target_pos, valid = env.labels(env.points_z, env.proprio)
    valid[:, 1] = False
    shifted = target_pos.clone()
    shifted[:, 1] += 3.0
    first, _ = planner.compute_loss(obs, target_pos, valid)
    second, _ = planner.compute_loss(obs, shifted, valid)
    torch.testing.assert_close(first["affordance_loss"], second["affordance_loss"])
    none_valid, _ = planner.compute_loss(obs, target_pos, torch.zeros_like(valid))
    assert float(none_valid["affordance_loss"]) == 0.0


@pytest.mark.parametrize("cls", [AttentionContactPlanner, MlpContactPlanner])
def test_planner_learns_a_synthetic_target(cls):
    torch.manual_seed(0)
    env = FakeContactPlannerEnv(num_envs=256, seed=1)
    planner = _planner(cls, env)
    optimizer = torch.optim.Adam(planner.parameters(), lr=3e-3)
    history = []
    for _ in range(300):
        env._sample_obs()
        obs = env.get_observations()[1]["observations"]["planner"]
        losses, stats = planner.compute_loss(obs, *env.labels(env.points_z, env.proprio))
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
        history.append((float(losses["loss"]), float(stats["top1_error"]), float(stats["occupancy_accuracy"])))
    first, last = np.mean(history[:10], axis=0), np.mean(history[-10:], axis=0)
    assert last[0] < 0.6 * first[0], (first, last)
    assert last[1] < 0.75 * first[1], (first, last)
    assert last[2] > 0.9, last


@pytest.mark.parametrize("cls", [AttentionContactPlanner, MlpContactPlanner])
def test_planner_onnx_matches_pytorch(cls, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    env = FakeContactPlannerEnv(num_envs=1)
    planner = _planner(cls, env)
    # non-trivial context statistics, so the exported normalizer matters
    planner.context_normalizer.update(torch.randn(64, 4) * 3.0 + 1.0)
    obs = env.get_observations()[1]["observations"]["planner"]
    path = planner.export_as_onnx(obs, str(tmp_path))
    session = ort.InferenceSession(path)
    for _ in range(5):
        env._sample_obs()
        obs = env.get_observations()[1]["observations"]["planner"]
        positions, occupancy = session.run(None, {"input": obs.numpy()})
        with torch.no_grad():
            points, log_affordance, logits = planner(obs)
            expected, _ = planner.decode(points, log_affordance, logits, planner.top_k)
        np.testing.assert_allclose(positions, expected.numpy(), atol=1e-5)
        np.testing.assert_allclose(occupancy, torch.sigmoid(logits).numpy(), atol=1e-5)


def test_planner_rejects_bad_layouts():
    segments = OrderedDict({"xy": (3, 4, 2), "z": (4, 3), "proprio": (4,)})
    with pytest.raises(ValueError):
        MlpContactPlanner(segments, "xy", "z", ["proprio"])
    with pytest.raises(ValueError):
        MlpContactPlanner(OrderedDict({"xy": (3, 4, 2), "z": (3, 4)}), "xy", "z", ["proprio"])
    with pytest.raises(TypeError):
        MlpContactPlanner(OrderedDict({"xy": (3, 4, 2), "z": (3, 4), "p": (1,)}), "xy", "z", ["p"], hidden=3)


# -- replay --------------------------------------------------------------------------------------


def _specs():
    return {"x": TensorSpec((2,)), "flag": TensorSpec((3,), torch.bool)}


def test_fifo_replay_overwrites_oldest_and_round_trips(tmp_path):
    replay = FifoReplayBuffer(5, _specs())
    assert replay.add({"x": torch.zeros(0, 2), "flag": torch.zeros(0, 3, dtype=torch.bool)}) == 0
    for start in (0, 3, 6):
        values = torch.arange(start, start + 3, dtype=torch.float)
        replay.add({"x": values[:, None].expand(-1, 2), "flag": (values[:, None] > 4).expand(-1, 3)})
    assert len(replay) == 5
    stored = sorted(replay._buffers["x"][:, 0].tolist())
    assert stored == [4.0, 5.0, 6.0, 7.0, 8.0]
    batch = replay.sample(64, generator=torch.Generator().manual_seed(0))
    assert batch["x"].shape == (64, 2) and batch["flag"].dtype == torch.bool
    assert set(batch["x"][:, 0].tolist()) <= set(stored)
    torch.save(replay.state_dict(), tmp_path / "replay.pt")
    restored = FifoReplayBuffer(5, _specs())
    restored.load_state_dict(torch.load(tmp_path / "replay.pt", weights_only=True))
    for name in ("x", "flag"):
        assert torch.equal(restored._buffers[name], replay._buffers[name])
    assert len(restored) == 5 and restored._cursor == replay._cursor
    # more samples than the capacity: the newest are kept
    replay.add({"x": torch.arange(7.0)[:, None].expand(-1, 2), "flag": torch.zeros(7, 3, dtype=torch.bool)})
    assert sorted(replay._buffers["x"][:, 0].tolist()) == [2.0, 3.0, 4.0, 5.0, 6.0]
    with pytest.raises(ValueError):
        FifoReplayBuffer(6, _specs()).load_state_dict(replay.state_dict())
    with pytest.raises(KeyError):
        replay.add({"x": torch.zeros(1, 2)})


# -- algorithm -----------------------------------------------------------------------------------


def _algorithm(env, planner_class="MlpContactPlanner", **kwargs):
    obs_format = env.get_obs_format()
    actor_critic = ActorCritic(
        obs_format=obs_format,
        num_actions=env.num_actions,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        activation="elu",
        init_noise_std=1.0,
    )
    size = SMALL_ATTENTION if planner_class == "AttentionContactPlanner" else SMALL_MLP
    options = dict(
        num_learning_epochs=1,
        num_mini_batches=2,
        planner_class_name=planner_class,
        planner_kwargs=dict(num_bodies=env.NUM_BODIES, **PLANNER_INPUTS, **size),
        planner_optimizer_kwargs={"lr": 3e-3},
        planner_batch_size=256,
        planner_accumulation_steps=4,
        planner_min_samples=64,
        planner_replay_capacity=2048,
    )
    options.update(kwargs)
    algorithm = ContactPlannerPPO(actor_critic, **options)
    return algorithm, obs_format


def _runner(env, tmp_path=None, **alg_kwargs):
    algorithm = {
        "class_name": "ContactPlannerPPO",
        "num_learning_epochs": 1,
        "num_mini_batches": 2,
        "planner_class_name": "MlpContactPlanner",
        "planner_kwargs": dict(num_bodies=env.NUM_BODIES, **PLANNER_INPUTS, **SMALL_MLP),
        "planner_optimizer_kwargs": {"lr": 3e-3},
        "planner_batch_size": 256,
        "planner_accumulation_steps": 4,
        "planner_min_samples": 64,
        "planner_replay_capacity": 2048,
        **alg_kwargs,
    }
    cfg = {
        "algorithm": algorithm,
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [32],
            "critic_hidden_dims": [32],
            "activation": "elu",
            "init_noise_std": 1.0,
        },
        "normalizers": {"policy": {"class_name": "EmpiricalNormalization"}},
        "num_steps_per_env": 8,
        "save_interval": 1000,
        "log_interval": 1000,
    }
    return OnPolicyRunner(env, cfg, log_dir=None if tmp_path is None else str(tmp_path), device="cpu")


def test_unknown_planner_option_raises():
    env = FakeContactPlannerEnv()
    with pytest.raises(TypeError, match="planner_num_update"):
        _algorithm(env, planner_num_update=3)


def test_composed_obs_is_normalized_like_the_env_targets_and_gated():
    env = FakeContactPlannerEnv(num_envs=32, gate_probability=0.5)
    algorithm, obs_format = _algorithm(env)
    normalizer = EmpiricalNormalization(10)
    normalizer.update(torch.randn(500, 10) * 2.0 + 3.0)
    algorithm.set_normalizers({"policy": normalizer})
    algorithm.init_storage(env.num_envs, 4, obs_format, env.num_actions)
    _, extras = env.get_observations()
    extra_obs = extras["observations"]
    raw_policy = extra_obs["policy"]
    obs = normalizer.normalize(raw_policy)
    algorithm.act(obs, extra_obs)
    stored = algorithm.transition.observations
    slot, _ = get_obs_slice(obs_format["policy"], "contact_targets")
    planned = algorithm.get_policy_feedback()["planned_contacts"].reshape(env.num_envs, -1)
    gate = extra_obs["contact_source"][:, 0] > 0.5
    assert 0 < int(gate.sum()) < env.num_envs
    # gated envs: the planner's raw targets, normalized with the policy normalizer's statistics of the slot;
    # inverting the normalizer gives back the planner's targets in meters
    torch.testing.assert_close(normalizer.inverse(stored)[gate][:, slot], planned[gate], atol=1e-5, rtol=0)
    torch.testing.assert_close(stored[~gate], obs[~gate])  # the rest keep the env's targets
    torch.testing.assert_close(stored[:, :4], obs[:, :4])  # other components untouched
    assert not torch.equal(stored[gate][:, slot], obs[gate][:, slot])


def test_planner_group_must_not_be_normalized():
    env = FakeContactPlannerEnv()
    algorithm, _ = _algorithm(env)
    with pytest.raises(ValueError, match="planner"):
        algorithm.set_normalizers({"planner": EmpiricalNormalization(10)})


def test_end_to_end_samples_replay_planner_and_feedback(tmp_path):
    torch.manual_seed(0)
    env = FakeContactPlannerEnv(num_envs=32, gate_probability=0.5)
    runner = _runner(env, tmp_path)
    runner.log = lambda locs, *args, **kwargs: None  # tensorboard/CUDA statistics are not under test
    results = []
    update = runner.alg.update

    def recording_update(*args, **kwargs):
        losses, stats = update(*args, **kwargs)
        results.append((dict(losses), dict(stats)))
        return losses, stats

    runner.alg.update = recording_update
    runner.learn(num_learning_iterations=12)
    assert env.iteration_history == list(range(12))
    # feedback reaches the env once per step, before it
    assert len(env.feedback_history) == 12 * 8
    assert set(env.feedback_history[-1]) == {"planned_contacts", "planned_occupancy"}
    # samples go into the replay, the planner trains once there are enough, and its loss falls
    assert results[0][1]["planner_samples_added"] == 8 * env.num_envs
    assert results[-1][1]["planner_replay_size"] == 2048
    planner_losses = [r[0]["planner_loss"] for r in results if "planner_loss" in r[0]]
    assert len(planner_losses) == 12  # 256 samples per iteration > planner_min_samples
    assert float(planner_losses[-1]) < 0.8 * float(planner_losses[0])
    assert os.path.exists(tmp_path / "model_12.pt") and os.path.exists(tmp_path / "planner_replay_latest.pt")


def test_rollout_storage_keeps_the_composed_obs():
    env = FakeContactPlannerEnv(num_envs=16, gate_probability=0.5)
    runner = _runner(env)
    runner.alg.planner.eval()
    obs, extras = env.get_observations()
    extra_obs = extras["observations"]
    obs = runner._prepare_extra_obs(extra_obs)
    gate = extra_obs["contact_source"][:, 0] > 0.5
    runner.rollout_step(obs, extra_obs)
    stored = runner.alg.storage.observations[0]
    slot, _ = get_obs_slice(env.get_obs_format()["policy"], "contact_targets")
    planned = env.feedback_history[0]["planned_contacts"].reshape(env.num_envs, -1)
    normalizer = runner.normalizers["policy"]
    torch.testing.assert_close(normalizer.inverse(stored)[gate][:, slot], planned[gate], atol=1e-5, rtol=0)
    torch.testing.assert_close(stored[~gate], obs[~gate])


def test_checkpoint_round_trip_with_replay_sidecar(tmp_path):
    env = FakeContactPlannerEnv(num_envs=32)
    runner = _runner(env, tmp_path)
    runner.log = lambda locs, *args, **kwargs: None
    runner.learn(num_learning_iterations=3)
    path = str(tmp_path / "model_3.pt")
    reloaded = _runner(env)
    reloaded.load(path)
    for a, b in zip(runner.alg.planner.state_dict().values(), reloaded.alg.planner.state_dict().values()):
        assert torch.equal(a, b)
    assert (
        reloaded.alg.planner_optimizer.state_dict()["state"].keys()
        == runner.alg.planner_optimizer.state_dict()["state"].keys()
    )
    assert len(reloaded.alg.planner_replay) == len(runner.alg.planner_replay) > 0
    assert torch.equal(
        reloaded.alg.planner_replay._buffers["planner_obs"], runner.alg.planner_replay._buffers["planner_obs"]
    )
    # a sidecar written for another checkpoint is not restored
    runner.alg._planner_samples_total += 1
    runner.alg.save_auxiliary(str(tmp_path))
    stale = _runner(env)
    stale.load(path)
    assert len(stale.alg.planner_replay) == 0


def test_stage_transition_from_a_ppo_checkpoint(tmp_path):
    env = FakeContactPlannerEnv(num_envs=16)
    obs_format = env.get_obs_format()
    actor_critic = ActorCritic(obs_format, env.num_actions, [32], [32], "elu", init_noise_std=1.0)
    ppo = PPO(actor_critic, num_learning_epochs=1, num_mini_batches=2)
    stage1 = {**ppo.state_dict(), "iter": 4000, "infos": None, "policy_normalizer_state_dict": {}}
    torch.save(stage1, tmp_path / "model_4000.pt")
    algorithm, _ = _algorithm(env)
    algorithm.init_storage(env.num_envs, 4, obs_format, env.num_actions)
    loaded = torch.load(tmp_path / "model_4000.pt", weights_only=True)
    with pytest.raises(KeyError, match="add_missing_algorithm_state"):
        algorithm.load_state_dict(loaded)
    fresh = {k: v.clone() for k, v in algorithm.planner.state_dict().items()}
    manipulated = ckpt_manipulator.add_missing_algorithm_state(loaded, algorithm.state_dict())
    assert manipulated["iter"] == 0
    algorithm.load_state_dict(manipulated)
    for k, v in algorithm.actor_critic.state_dict().items():
        assert torch.equal(v, actor_critic.state_dict()[k])
    for k, v in algorithm.planner.state_dict().items():
        assert torch.equal(v, fresh[k])
    # a checkpoint of the new stage passes unchanged (resume keeps its iteration)
    resumed = {**algorithm.state_dict(), "iter": 77, "infos": None}
    assert ckpt_manipulator.add_missing_algorithm_state(resumed, algorithm.state_dict())["iter"] == 77


def test_runner_exports_actor_and_planner(tmp_path):
    env = FakeContactPlannerEnv(num_envs=1)
    runner = _runner(env)
    obs, _ = env.get_observations()
    runner.export_as_onnx(obs, str(tmp_path))
    assert {"actor.onnx", "planner.onnx", "policy_normalizer.npz"} <= set(os.listdir(tmp_path))
