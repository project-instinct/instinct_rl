"""Unit tests for the model-based stack: world model, replay, GAE, SuperMimic.

Run with ``python -m pytest instinct_rl/tests/test_supermimic.py`` or directly with ``python``.
These tests use the toy environment in ``fake_env.py`` and run in about ten seconds on CPU.
"""

import torch

from instinct_rl.algorithms.supermimic import SuperMimic
from instinct_rl.modules import ActorCritic, WorldModel
from instinct_rl.storage.replay_storage import ReplayStorage, TensorSpec
from instinct_rl.tests.fake_env import (
    CONTEXT_SEGMENTS,
    NUM_ACTIONS,
    OBS_FORMAT,
    STATE_SEGMENTS,
    FakeVecEnv,
    ToyImagination,
)
from instinct_rl.utils.returns import compute_gae, normalize_advantages

DEVICE = "cpu"


def _build(critic=True, horizon=8, wm_horizon=4, use_critic=True, num_imagined=64, warmup=0, num_envs=16):
    env = FakeVecEnv(num_envs=num_envs, episode_length=16)
    actor_critic = ActorCritic(
        obs_format=OBS_FORMAT,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
        init_noise_std=1.0,
    )
    world_model = WorldModel(
        state_segments=STATE_SEGMENTS,
        action_size=NUM_ACTIONS,
        context_size=sum(int(torch.tensor(v).prod()) for v in CONTEXT_SEGMENTS.values()),
        hidden_sizes=(64, 64),
    )
    algo = SuperMimic(
        actor_critic=actor_critic,
        imagination=env.imagination,
        world_model=world_model,
        state_specs={k: TensorSpec(tuple(v)) for k, v in STATE_SEGMENTS.items()},
        context_specs={k: TensorSpec(tuple(v)) for k, v in CONTEXT_SEGMENTS.items()},
        replay_capacity=200,
        horizon=horizon,
        wm_horizon=wm_horizon,
        use_critic=use_critic,
        num_imagined=num_imagined,
        wm_batches_per_update=2,
        wm_batch_size=32,
        critic_epochs=1,
        critic_mini_batches=2,
        updates_per_iteration=1,
        wm_warmup_iterations=warmup,
        collection_noise_std=0.1,
        device=DEVICE,
    )
    algo.init_storage(env.num_envs, 5, OBS_FORMAT, NUM_ACTIONS, num_rewards=1)
    return env, algo


def test_generic_world_model_shapes_and_modes():
    wm = WorldModel(STATE_SEGMENTS, action_size=NUM_ACTIONS, context_size=1, hidden_sizes=(32,))
    state = torch.randn(4, 6)
    action = torch.randn(4, NUM_ACTIONS)
    context = torch.rand(4, 1)
    assert wm(state, action, context).shape == (4, 6)

    # partial delta mode keeps the layout but mixes prediction modes
    wm2 = WorldModel(STATE_SEGMENTS, action_size=NUM_ACTIONS, predict_delta=["state_pos"], hidden_sizes=(16,))
    assert wm2.predict_delta == {"state_pos"}
    assert wm2(state, action, None).shape == (4, 6)

    # absolute mode
    wm3 = WorldModel(STATE_SEGMENTS, action_size=NUM_ACTIONS, predict_delta="none", hidden_sizes=(16,))
    assert wm3.predict_delta == set()

    # loud failures
    for bad in (
        lambda: WorldModel(STATE_SEGMENTS, NUM_ACTIONS, predict_delta=["nope"]),
        lambda: WorldModel(STATE_SEGMENTS, NUM_ACTIONS, frame_of_component={"x": "world"}),
        lambda: wm(state, action, None),  # context missing
    ):
        try:
            bad()
        except (ValueError, TypeError):
            continue
        raise AssertionError("expected an error")


def test_world_model_learns_analytic_dynamics():
    torch.manual_seed(0)
    wm = WorldModel(STATE_SEGMENTS, action_size=NUM_ACTIONS, context_size=1, hidden_sizes=(128, 128))
    world = FakeVecEnv().world
    opt = torch.optim.Adam(wm.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)

    def batch(n=256):
        state, context = world.sample(n, generator=gen)
        action = torch.randn(n, NUM_ACTIONS, generator=gen) * 0.5
        return state, action, context, world.step(state, action, context)

    first = None
    for step in range(400):
        state, action, context, nxt = batch()
        losses = wm.rollout_loss(
            state.unsqueeze(0) if state.dim() == 1 else state, action.unsqueeze(1), context, nxt.unsqueeze(1)
        )
        opt.zero_grad()
        losses["loss"].backward()
        opt.step()
        if step == 0:
            first = float(losses["loss"])
    final = float(losses["loss"])
    assert final < 0.2 * first, f"world model did not learn: {first:.4f} -> {final:.4f}"


def test_gae_matches_hand_computation():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.zeros(3, 1)
    dones = torch.zeros(3, 1)
    last_values = torch.zeros(1)
    returns, adv = compute_gae(rewards, values, dones, last_values, gamma=0.5, lam=1.0)
    # geometric sum from each step: 1 + 0.5 + 0.25, then 1 + 0.5, then 1
    expected = torch.tensor([[1.75], [1.5], [1.0]])
    assert torch.allclose(returns, expected, atol=1e-6), returns
    assert torch.allclose(adv, expected, atol=1e-6)

    # a done flag stops the bootstrap
    dones2 = torch.tensor([[0.0], [1.0], [0.0]])
    returns2, _ = compute_gae(rewards, values, dones2, last_values, gamma=0.5, lam=1.0)
    # step 0 -> 1 + 0.5*(done at step 1 ends the chain) = 1.5 ; step 1 -> 1 ; step 2 -> 1
    assert torch.allclose(returns2, torch.tensor([[1.5], [1.0], [1.0]]), atol=1e-6), returns2

    norm = normalize_advantages(torch.tensor([1.0, 2.0, 3.0]))
    assert abs(float(norm.mean())) < 1e-5
    assert abs(float(norm.std()) - 1.0) < 1e-4


def test_replay_windows_are_contiguous_and_checkpointable():
    specs = {"s": TensorSpec((2,)), "episode_step": TensorSpec((), torch.int64)}
    r = ReplayStorage(num_envs=2, capacity_per_env=32, specs=specs)
    gen = torch.Generator().manual_seed(0)
    steps = torch.zeros(2, dtype=torch.int64)
    for t in range(40):
        if t == 20:
            steps[:] = 0
        r.add({"s": torch.randn(2, 2, generator=gen), "episode_step": steps.clone()})
        steps += 1
    windows, mask = r.sample_windows(4, 3, ["s", "episode_step"], generator=gen)
    assert windows["s"].shape == (4, 3, 2, 2)
    # validity is per environment: envs reset independently, so one env's window can straddle a reset
    # while another's does not
    assert mask.shape == (4, 2)

    # selecting one env per sample drops the env axis from both the windows and the mask
    sel = torch.tensor([0, 1, 0, 1])
    windows_one, mask_one = r.sample_windows(4, 3, ["s", "episode_step"], generator=gen, env_ids=sel)
    assert windows_one["s"].shape == (4, 3, 2)
    assert mask_one.shape == (4,)
    # episode_step increments inside a valid window
    assert torch.all(windows["episode_step"][0, 1] - windows["episode_step"][0, 0] == 1)

    state = r.state_dict()
    r2 = ReplayStorage(num_envs=2, capacity_per_env=32, specs=specs)
    r2.load_state_dict(state)
    assert torch.equal(r._buffers["s"], r2._buffers["s"])


def test_supermimic_runs_and_improves_imagined_return():
    torch.manual_seed(0)
    env, algo = _build()
    obs, extras = env.get_observations()

    returns = []
    for iteration in range(30):
        for _ in range(5):
            action = algo.act(obs, extras["observations"])
            obs, reward, done, infos = env.step(action)
            extra_obs = infos["observations"]
            obs = extra_obs["policy"]
            algo.process_env_step(reward, done, infos, obs, extra_obs)
        losses, stats = algo.update(iteration)
        if "imagined_reward" in losses:
            # the reward part only: ``policy_return`` also contains the critic bootstrap, which grows in
            # magnitude as the critic learns the infinite-horizon return
            returns.append(losses["imagined_reward"])

    assert algo.replay is not None and len(algo.replay) > 0
    assert "world_model_loss" in losses
    assert stats.get("world_model_trained") == 1.0
    # imagined return must improve: the toy task's optimum is to hold still
    assert len(returns) >= 20, f"too few policy updates: {len(returns)}"
    early = sum(returns[:5]) / 5
    late = sum(returns[-5:]) / 5
    assert late > early, f"imagined return did not improve: {early:.5f} -> {late:.5f}"


def test_supermimic_checkpoint_round_trip():
    torch.manual_seed(0)
    env, algo = _build()
    obs, extras = env.get_observations()
    for _ in range(5):
        action = algo.act(obs, extras["observations"])
        obs, reward, done, infos = env.step(action)
        extra_obs = infos["observations"]
        obs = extra_obs["policy"]
        algo.process_env_step(reward, done, infos, obs, extra_obs)
    algo.update(0)
    state = algo.state_dict()
    assert "replay_state_dict" in state

    _, algo2 = _build()
    algo2.load_state_dict(state)
    assert torch.equal(next(iter(algo.world_model.parameters())), next(iter(algo2.world_model.parameters())))
    # the trained policy must survive the round trip, or a checkpoint cannot be played or resumed
    for p, q in zip(algo.actor_critic.parameters(), algo2.actor_critic.parameters()):
        assert torch.equal(p, q)
    assert len(algo2.replay) == len(algo.replay)

    torch.save(state, "/tmp/instinct_rl_supermimic_test.pt")
    loaded = torch.load("/tmp/instinct_rl_supermimic_test.pt", weights_only=True)
    assert "world_model" in loaded


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"[run] {name}")
            fn()
            print(f"[ok ] {name}")
    print("\nall SuperMimic tests passed")


def test_world_model_batch_is_independent_of_num_envs():
    """The world-model fit batch must be ``wm_batch_size``, whatever the environment count.

    The sampler returns ``(B, length, num_envs, ...)`` when asked for every environment, and folding
    that env axis into the batch silently multiplied the fit batch by ``num_envs``. The same config
    then trained on different data volumes per machine, and at a few thousand envs the resulting
    tensors exhausted memory -- which surfaced as the process dying rather than as an error.
    """
    recorded = []

    for num_envs in (8, 64):
        env, algo = _build(num_envs=num_envs, warmup=0)
        original = algo.world_model.rollout_loss

        def spy(state0, actions, context, target_states, *args, **kwargs):
            recorded.append((num_envs, state0.shape[0], actions.shape[0], target_states.shape[1]))
            return original(state0, actions, context, target_states, *args, **kwargs)

        algo.world_model.rollout_loss = spy
        try:
            for _ in range(3):
                obs, extras = env.get_observations()
                for _ in range(6):
                    action = algo.act(obs, extras["observations"])
                    obs, reward, done, infos = env.step(action)
                    extra_obs = infos["observations"]
                    algo.process_env_step(reward, done, infos, extra_obs["policy"], extra_obs)
            algo._fit_world_model()
        finally:
            algo.world_model.rollout_loss = original

    assert recorded, "world-model fit never ran"
    batch_sizes = {batch for _, batch, _, _ in recorded}
    assert batch_sizes == {32}, f"world-model batch tracked num_envs: {recorded}"
    # and every batch really carried the configured window length
    assert all(k == 4 for *_, k in recorded), f"unexpected window length: {recorded}"
