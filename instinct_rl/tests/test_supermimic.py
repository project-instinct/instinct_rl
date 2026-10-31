"""Unit tests for the model-based stack: world model, replay, GAE, SuperMimic.

Run with ``python -m pytest instinct_rl/tests/test_supermimic.py`` or directly with ``python``.
These tests use the toy environment in ``fake_env.py`` and run in about ten seconds on CPU.
"""

import pytest
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


def _build(
    critic=True,
    horizon=8,
    wm_horizon=4,
    use_critic=True,
    num_imagined=64,
    warmup=0,
    num_envs=16,
    wm_batches=2,
    device=DEVICE,
    mixed_precision=None,
):
    env = FakeVecEnv(num_envs=num_envs, episode_length=16, device=device)
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
        wm_batches_per_update=wm_batches,
        wm_batch_size=32,
        critic_epochs=1,
        critic_mini_batches=2,
        updates_per_iteration=1,
        wm_warmup_iterations=warmup,
        collection_noise_std=0.1,
        mixed_precision=mixed_precision,
        device=device,
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


@pytest.mark.parametrize("use_critic,mixed_precision", [(True, None), (False, None), (True, "bfloat16")])
def test_supermimic_runs_and_improves_imagined_return(use_critic, mixed_precision):
    # The world model gets a real fitting budget and a warm start: improving an imagined return only
    # means something when the model it is imagined in describes the dynamics. ``use_critic=False`` is
    # the paper's critic ablation: the policy bootstraps with zero and no critic is fitted.
    torch.manual_seed(0)
    env, algo = _build(warmup=5, wm_batches=20, use_critic=use_critic, mixed_precision=mixed_precision)
    obs, extras = env.get_observations()

    returns = []
    for iteration in range(35):
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


def test_supermimic_checkpoint_round_trip(tmp_path):
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
    # by default the checkpoint holds no replay; only the newest copy is kept, beside it
    state = algo.state_dict()
    assert "replay_state_dict" not in state
    algo.save_auxiliary(str(tmp_path))
    assert (tmp_path / SuperMimic.REPLAY_FILE).exists()

    _, algo2 = _build()
    algo2.load_state_dict(state)
    algo2.load_auxiliary(str(tmp_path), state)
    assert torch.equal(next(iter(algo.world_model.parameters())), next(iter(algo2.world_model.parameters())))
    # the trained policy must survive the round trip, or a checkpoint cannot be played or resumed
    for p, q in zip(algo.actor_critic.parameters(), algo2.actor_critic.parameters()):
        assert torch.equal(p, q)
    assert len(algo2.replay) == len(algo.replay) > 0

    # a checkpoint older than the saved replay resumes with an empty replay rather than a newer one
    _, algo3 = _build()
    older = dict(state, iteration=state["iteration"] - 1)
    algo3.load_state_dict(older)
    algo3.load_auxiliary(str(tmp_path), older)
    assert len(algo3.replay) == 0

    # replay_checkpoint="every" keeps the replay inside the checkpoint, as before
    algo.replay_checkpoint = "every"
    assert "replay_state_dict" in algo.state_dict()
    torch.save(algo.state_dict(), tmp_path / "ckpt.pt")
    loaded = torch.load(tmp_path / "ckpt.pt", weights_only=True)
    assert "world_model" in loaded and "replay_state_dict" in loaded


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


def test_world_model_keeps_every_quaternion_unit():
    """A component holding several quaternions must come out with *each* one of unit norm.

    Normalising the flattened component as one vector scales n unit quaternions to 1/sqrt(n) each,
    which silently corrupts every downstream rotation (rewards, base-frame observations).
    """
    torch.manual_seed(0)
    model = WorldModel({"root_quat_w": (4,), "body_link_quat_w": (14, 4)}, action_size=2, hidden_sizes=(16,))
    state = torch.randn(8, 4 + 14 * 4)
    out = model(state, torch.randn(8, 2))
    norms = torch.cat([out[:, :4], out[:, 4:]], dim=-1).reshape(8, 15, 4).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), norms


def test_world_model_fit_is_autoregressive():
    """The k-step loss rolls the model on its own predictions (paper Eq. 1), not on logged states."""
    torch.manual_seed(0)
    model = WorldModel({"x": (3,)}, action_size=1, hidden_sizes=(16,), use_normalizer=False)
    state0, actions, targets = torch.randn(4, 3), torch.randn(4, 3, 1), torch.randn(4, 3, 3)
    with torch.no_grad():
        pred, own, logged = state0, [], []
        for j in range(3):
            pred = model(pred, actions[:, j])
            own.append(((pred - targets[:, j]) ** 2).mean())
            start = state0 if j == 0 else targets[:, j - 1]
            logged.append(((model(start, actions[:, j]) - targets[:, j]) ** 2).mean())
        loss = model.rollout_loss(state0, actions, None, targets, update_normalizer=False)["loss"]
    assert torch.isclose(loss, torch.stack(own).mean()), (loss, torch.stack(own).mean())
    assert not torch.isclose(loss, torch.stack(logged).mean())


def test_world_model_fit_pairs_each_transition_with_its_own_action():
    """The world model must be fit on ``(s_t, a_t) -> s_{t+1}``, the pairing imagination uses.

    The toy's joint block integrates the action exactly (``q_{t+1} = q_t + dt * a_t``), so for every
    fitted window the first transition's joint change must equal ``dt`` times its action.
    """
    torch.manual_seed(0)
    env, algo = _build()
    recorded = []
    original = algo.world_model.rollout_loss

    def spy(state0, actions, context, target_states, mask=None, **kwargs):
        recorded.append((state0.detach(), actions.detach(), target_states.detach(), mask))
        return original(state0, actions, context, target_states, mask=mask, **kwargs)

    algo.world_model.rollout_loss = spy
    obs, extras = env.get_observations()
    for _ in range(12):
        action = algo.act(obs, extras["observations"])
        obs, reward, done, infos = env.step(action)
        extra_obs = infos["observations"]
        obs = extra_obs["policy"]
        algo.process_env_step(reward, done, infos, obs, extra_obs)
    algo._fit_world_model()
    algo.world_model.rollout_loss = original

    assert recorded, "world-model fit never ran"
    state0, actions, targets, mask = recorded[0]
    valid = mask[:, 0].bool()
    assert bool(valid.any())
    joint_change = (targets[valid, 0, 4:6] - state0[valid, 4:6]) / env.world.dt
    assert torch.allclose(joint_change, actions[valid, 0], atol=1e-4), (joint_change - actions[valid, 0]).abs().max()


def test_history_window_restarts_at_the_episode_start():
    """A replayed history must look like the environment's own: frames before the reset are replaced by
    the reset frame, whose ``last_action`` is zero, exactly as the observation manager fills it."""
    h = 4
    states = torch.arange(h, dtype=torch.float32).view(1, h, 1).repeat(3, 1, 1)  # frame t holds value t
    actions = (torch.arange(h, dtype=torch.float32) + 10).view(1, h, 1).repeat(3, 1, 1)
    episode_step = torch.tensor([0, 2, 9])  # newest frame is: the reset frame / two steps in / deep in
    s, a = SuperMimic.history_since_episode_start(states, actions, episode_step)
    # reset frame newest: every slot shows it, with a zero action
    assert s[0].flatten().tolist() == [3, 3, 3, 3] and a[0].flatten().tolist() == [0, 0, 0, 0]
    # two steps after the reset (reset frame = slot 1): older slots repeat it, its action is zero
    assert s[1].flatten().tolist() == [1, 1, 2, 3] and a[1].flatten().tolist() == [0, 0, 12, 13]
    # no reset inside the window: unchanged
    assert s[2].flatten().tolist() == [0, 1, 2, 3] and a[2].flatten().tolist() == [10, 11, 12, 13]


def test_replay_observed_range_tracks_the_ring_contents():
    """The cached range must equal a brute-force min/max over the rows the ring currently holds."""
    gen = torch.Generator().manual_seed(0)
    replay = ReplayStorage(
        num_envs=3, capacity_per_env=5, specs={"x": TensorSpec((2,)), "episode_step": TensorSpec((), torch.long)}
    )
    written = []
    for step in range(12):
        x = torch.randn(3, 2, generator=gen) * (10.0 if step == 1 else 1.0)  # an outlier that ages out
        written.append(x)
        replay.add({"x": x, "episode_step": torch.full((3,), step, dtype=torch.long)})
        if step >= 2:
            lo, hi = replay.observed_range(["x"])["x"]
            held = torch.stack(written[-5:])
            assert torch.equal(lo, held.amin(dim=(0, 1))) and torch.equal(hi, held.amax(dim=(0, 1)))


def test_replay_keeps_the_state_an_episode_ended_in():
    """On a step that ends an episode the env already returns the next episode's first state; the
    replay must also hold the state the step reached, so the world model can fit the final transition."""
    torch.manual_seed(0)
    env, algo = _build()
    obs, extras = env.get_observations()
    previous = None
    checked = 0
    for _ in range(40):
        action = algo.act(obs, extras["observations"])
        obs, reward, done, infos = env.step(action)
        extra_obs = infos["observations"]
        obs = extra_obs["policy"]
        algo.process_env_step(reward, done, infos, obs, extra_obs)
        row = (algo.replay._cursor - 1) % algo.replay.capacity_per_env
        reached = algo.replay._buffers["joint_pos@final"][row]
        stored = algo.replay._buffers["joint_pos"][row]
        ended = done.bool()
        # where no episode ended the reached state is the stored one
        assert torch.equal(reached[~ended], stored[~ended])
        if previous is not None and bool(ended.any()):
            # the toy integrates the action exactly: the reached joints follow from the previous state
            expected = previous[ended] + env.world.dt * algo.replay._buffers["action"][row][ended]
            assert torch.allclose(reached[ended], expected, atol=1e-5)
            checked += int(ended.sum())
        previous = stored.clone()
    assert checked > 0, "no episode ended during the test"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("mixed_precision", [None, "bfloat16"])
def test_graphed_policy_pass_matches_eager(mixed_precision):
    """The CUDA-graph replay of a policy chunk must give the eager pass's gradients and outputs, also
    for later updates whose start states differ from the ones the graph was captured with."""
    torch.manual_seed(0)
    env, algo = _build(device="cuda", num_imagined=64, mixed_precision=mixed_precision)
    obs, extras = env.get_observations()
    for _ in range(12):
        action = algo.act(obs, extras["observations"])
        obs, reward, done, infos = env.step(action)
        extra_obs = infos["observations"]
        obs = extra_obs["policy"]
        algo.process_env_step(reward, done, infos, obs, extra_obs)
    chunks = [torch.arange(0, 32, device="cuda"), torch.arange(32, 64, device="cuda")]

    def grads():
        # parameters the return does not depend on (the action-noise ``std``) may have no gradient
        return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in algo.policy_params]

    for _ in range(3):  # capture on the first update, replay on the next ones
        algo._begin_imagination()
        algo.policy_optimizer.zero_grad(set_to_none=False)
        eager = []
        for idx in chunks:
            out = algo._imagined_return(idx, keep_trajectory=True)
            (-out[0] * 0.5).backward()
            # keep values only: a live autograd graph pins the parameters' AccumulateGrad nodes to the
            # default stream, which a later graph capture must not depend on
            eager.append((out[0].detach(), out[1], out[2], out[3]))
            del out
        expected = grads()
        algo.policy_optimizer.zero_grad(set_to_none=False)
        graphed = [algo._graphed_policy_chunk(idx, 0.5) for idx in chunks]
        for e, g in zip(expected, grads()):
            assert torch.allclose(e, g, atol=1e-5, rtol=1e-4), (e - g).abs().max()
        for e, g in zip(eager, graphed):
            assert torch.allclose(e[0].detach(), g[0], atol=1e-6)
            for k in e[3]:
                assert torch.allclose(e[3][k], g[3][k], atol=1e-5), k
        # a policy step between updates: the graph must read the new weights
        algo.policy_optimizer.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("mixed_precision", [None, "bfloat16"])
def test_graphed_world_model_fit_matches_eager(mixed_precision):
    """The CUDA-graph replay of a world-model batch gives the eager loss and gradients, on new data too."""
    torch.manual_seed(0)
    env, algo = _build(device="cuda", mixed_precision=mixed_precision)
    obs, extras = env.get_observations()
    for _ in range(12):
        action = algo.act(obs, extras["observations"])
        obs, reward, done, infos = env.step(action)
        extra_obs = infos["observations"]
        obs = extra_obs["policy"]
        algo.process_env_step(reward, done, infos, obs, extra_obs)
    wm = algo.world_model
    params = [p for p in wm.parameters() if p.requires_grad]
    for _ in range(3):
        state0 = torch.randn(32, wm.state_size, device="cuda")
        actions = torch.randn(32, 4, wm.action_size, device="cuda")
        context = torch.rand(32, wm.context_size, device="cuda")
        targets = state0.unsqueeze(1) + 0.1 * torch.randn(32, 4, wm.state_size, device="cuda")
        mask = torch.rand(32, 4, device="cuda") > 0.2
        batch = (state0, actions, context, targets, mask)
        wm._update_normalizers(*batch)
        for p in params:  # zeroed in place: the graph holds these gradient tensors
            if p.grad is not None:
                p.grad.zero_()
        with algo._autocast():
            loss = wm.rollout_loss(*batch[:4], mask=mask, update_normalizer=False)["loss"]
        loss.backward()
        eager = loss.detach().clone()
        del loss  # a live autograd graph would tie the capture to the default stream
        expected = [p.grad.detach().clone() for p in params]
        graphed = algo._graphed_world_model_loss(batch)
        assert torch.allclose(eager, graphed, rtol=1e-5, atol=1e-6)
        for e, p in zip(expected, params):
            assert torch.allclose(e, p.grad, rtol=1e-4, atol=1e-6)
        algo.world_model_optimizer.step()  # the graph must read the updated weights next time


def test_mixed_precision_keeps_the_imagined_state_float32():
    """Under bfloat16 only the network passes are reduced: the imagined trajectory the critic is fit on,
    the rewards and the world model's states stay float32."""
    torch.manual_seed(0)
    env, algo = _build(mixed_precision="bfloat16")
    obs, extras = env.get_observations()
    for _ in range(12):
        action = algo.act(obs, extras["observations"])
        obs, reward, done, infos = env.step(action)
        extra_obs = infos["observations"]
        obs = extra_obs["policy"]
        algo.process_env_step(reward, done, infos, obs, extra_obs)
    algo._begin_imagination()
    with algo._autocast():
        assert algo.world_model.net(torch.zeros(1, algo.world_model.net.model[0].in_features)).dtype == torch.bfloat16
    out = algo._imagined_return(torch.arange(16), keep_trajectory=True)
    assert out[0].dtype == torch.float32
    for name, value in out[3].items():
        assert value.dtype == torch.float32, name
    with pytest.raises(ValueError):
        _build(mixed_precision="float16")
