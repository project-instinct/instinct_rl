# Instinct RL (from Rsl-RL code base, but with lots of modificaitons)

## Warning
This codebase is released under the [Modified MIT License](LICENSE), based on the license used by Kimi K2. The license permits commercial use and adds a display requirement for commercial products or services above the user or revenue thresholds stated in the license. Third-party components and files with separate license notices remain subject to those notices.

## Contributing
See our [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md) for contribution guidelines. By contributing or submitting a pull request, you grant the project maintainers a copyright license to your contribution while retaining your ownership of it.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of acknowledged contributors.

---

## Install

This branch requires Python `>=3.12,<3.13`, PyTorch `2.11.0`, TorchVision `0.26.0`, NumPy `>=2`,
ONNX `>=1.18,<1.22`, and ONNX Script `>=0.5`.

- Clone this repository separately from the Project Instinct installation:

  ```bash
  # Option 1: HTTPS
  git clone https://github.com/project-instinct/instinct_rl.git
  # Option 2: SSH
  git clone git@github.com:project-instinct/instinct_rl.git
  ```

- Install using a Python 3.12 interpreter:

  ```bash
  python -m pip install -e instinct_rl
  ```

---

## Important New Concepts

### General Observation format

- **obs_format**: An OrderedDict specifying the format of input observations.
    key: obs_segment name, specifically for algorithm requirements, as obs_group in ManagerBasedRlEnv in IsaacLab
    value: an obs_segment object (see below)

- **obs_segment**: An OrderedDict specifying the segment of the observation.
    key: obs term name, as obs_term_name in ManagerBasedRlEnv in IsaacLab
    value: the shape of each part of the observation.

- **obs_pack**: The dictionary of observations following the definition of obs_format.
    key: obs_segment name as in observation_manager in ManagerBasedRlEnv in IsaacLab
    value: A flattened tensor/vector of each observation (required by the algorithm).

- **obs_component**: As obs_term in ManagerBasedRlEnv in IsaacLab.

---

## How to use the swappable algorithms and network modules design

In `on_policy_runner.py`, we have the following code:

```python
actor_critic = modules.build_actor_critic(
    self.policy_cfg.pop("class_name"),
    self.policy_cfg,
    obs_format,
    num_actions=env.num_actions,
    num_rewards=env.num_rewards,
).to(self.device)
```

This code is used to build the actor critic network. The `class_name` is the name of the actor critic class, and the `policy_cfg` is the configuration of the actor critic. The `obs_format` is the observation format, and the `num_actions` is the number of actions, and the `num_rewards` is the number of rewards.

The `modules.build_actor_critic` function is used to build the actor critic network. It is a factory function that builds the actor critic network based on the `class_name` and `policy_cfg`.

The `class_name` is the name of the actor critic class, which can be a class implemented in this repository, or a full import path to your own actor critic class in `module_name:class_name` pattern.

---

## Algorithms

### PPO

Standard Proximal Policy Optimization (PPO) algorithm for reinforcement learning. Implements the PPO objective with clipped surrogate loss, value function loss, and entropy regularization. Supports multiple learning epochs and mini-batch updates per environment step.

### State Estimator

Algorithm for learning state representations and estimation from partial observations. Combines reconstruction and prediction objectives to learn meaningful latent state representations that can be used for control.

### AMP (WASABI)

Implementation of WASABI, starting from the algorithm framework of AMP. Uses adversarial training with a discriminator to learn motion priors from expert demonstrations. Supports multiple discriminator architectures, including BCE loss, Wasserstein loss, and MSELoss. Includes gradient penalty and various discriminator architectures.

### Distillation and DAgger (TPPO)

Teacher-student distillation framework that extends PPO with teacher network guidance. Implements Dataset Aggregation (DAgger) and knowledge distillation techniques. Allows learning from expert demonstrations while maintaining PPO's stability guarantees. Supports various teacher action selection probabilities and distillation loss coefficients.

### VAE Distillation

Teacher-student distillation framework that extends TPPO with a VAE student network. Uses a VAE to encode and decode to generate the student's action. It generates additional latent distribution separate from the action distribution, which enables the potential of VAE-distillation combined with PPO.

### SuperMimic

World-model based policy training without PPO (SuperMimic, adapted from the [reference implementation](https://github.com/zita-ch/supermimic_vibed) by Chong Zhang, ETH Zurich, MIT License, see [licenses/dependencies/supermimic_vibed_license.txt](licenses/dependencies/supermimic_vibed_license.txt)). The policy collects transitions from the simulator into a replay storage. A floating-base world model (state dynamics and contacts) and a failure head are fitted on k-step windows. The policy is then trained by back-propagating the discounted return through imagined rollouts of the world model (BPTT), with a critic fitted by GAE on the imagined transitions. `SuperMimic` is the native setting: proprioceptive dynamics only, the policy observes no image.

### Visual SuperMimic

`VisualSuperMimic(VisualLatentAlgoMixin, SuperMimic)` is `SuperMimic` for a policy that also sees an image, structured like `WasabiPPO`: the mixin extends the hooks of `SuperMimic` (storage fields, the transition, the world-model losses, the imagination step, diagnostics and the state dict) through `super()`. The latent `z = E(image)` of the image encoder of the policy is a state of the world model: the dynamics MLP takes `z_t` and predicts its increment, in the same k-step windows, so nothing is rendered in imagination and the actor acts on the imagined latent. `E` is trained by the world model and grounded by the reconstruction of the clean image by a `LatentDecoder`. It needs a `VisualTaskModel`.

### Running SuperMimic

SuperMimic runs on the `OnPolicyRunner` like every other algorithm: `OnPolicyRunner` gives `init_storage` the env and its observation normalizers (algorithms that do not need them ignore them), from which SuperMimic builds its task model and normalizes the imagined observations. Every learning iteration the runner steps the env `num_steps_per_env` times (the update-to-data ratio is `1 / num_steps_per_env` updates per env step) and calls `SuperMimic.update` once, which does nothing until the replay storage holds `warmup_steps` env steps. The checkpoint is the one `alg.state_dict()` of `OnPolicyRunner.save`: the policy keeps the keys of `PPO` (so it is directly playable) and the world model, the critic, the failure head, their optimizers and the state of the task model are added, like the discriminator of `WasabiPPO`. Run with `inference_mode_rollout: False` in the runner config. Losses and statistics (including the held-out diagnostics every `diagnostics_interval` iterations) are logged by the runner as `Loss/...` and `Train/...`.

What the algorithm needs from the task is defined by the env, as observation groups of `obs_format` (like the `amp_policy` / `amp_reference` groups of `WasabiPPO`), and by one differentiable model of the task:

- **The `world_model` observation group** (`world_model_obs_group`, no noise, no history, no normalizer) has the components `root_pos_w`, `root_quat_w`, `root_lin_vel_w`, `root_ang_vel_w`, `joint_pos`, `joint_vel`, `last_action`, `contact`, `context` and the reference command (`command_components`, floats); `VisualSuperMimic` also needs the image of the policy and the clean image. The names are arguments of the algorithm, see the docstring of `SuperMimic`.
- **`infos["terminal_observations"]`** of `VecEnv.step`: the groups of the state right after the step, before the finished envs are reset. A real termination is `dones & ~infos["time_outs"]`.
- **`TaskModel`** (`instinct_rl.env`, `VisualTaskModel` for a policy with an image): the observations, rewards and terminations of the task as torch functions of the imagined state. It is built from `task_model_class_name` (`module:Class`) as `Class(env=env, obs_format=obs_format, device=device, **task_model_kwargs)`.

```python
train_cfg = {
    "policy": {"class_name": "EncoderActorCritic", ...},
    "normalizers": {"policy": {"class_name": "EmpiricalNormalization"}},
    "algorithm": {
        "class_name": "VisualSuperMimic",
        "task_model_class_name": "your_package.your_module:YourTaskModel",
        "task_model_kwargs": {...},
        "imagination_horizon": 32, "warmup_steps": 64, "action_noise": 0.1, ...
    },
    "num_steps_per_env": 2, "inference_mode_rollout": False,
    "save_interval": 2000, "log_interval": 50,
}
runner = OnPolicyRunner(env, train_cfg, log_dir, device)
```

A run of the former stand-alone SuperMimic trainer (`model_<it>.pt` plus the sidecar `supermimic_state_<it>.pt`) is converted on loading with `ckpt_manipulator="supermimic_sidecar"` and `ckpt_manipulator_kwargs={"sidecar_path": ".../supermimic_state_<it>.pt"}` (`instinct_rl.utils.ckpt_manipulator`).

---

## Network modules

### Actor Critic

Standard actor-critic architecture with separate policy (actor) and value (critic) networks. Supports configurable hidden dimensions, activation functions, and observation processing. Implements both continuous and discrete action spaces with proper probability distributions.

### MoE Actor Critic

Mixture of Experts (MoE) extension of the actor-critic architecture. Uses multiple expert networks with a gating mechanism to specialize different experts on different parts of the state/action space. Can improve performance on complex tasks by allowing specialization while maintaining parameter efficiency.

### VAE Actor Critic

Variational Autoencoder (VAE) extension of the actor-critic architecture. Uses a VAE to learn a latent representation of the observation, and then uses the actor to predict the action from the latent representation. Can be used for dimensionality reduction, feature learning, or as part of a larger reinforcement learning pipeline.

### Encoder related modules

Collection of encoder architectures for processing different input modalities:

- **Encoder Actor Critic**: Actor-critic networks with encoder backbones for processing high-dimensional inputs like images or complex state representations
- **All Mixer**: Multi-modal input processing that combines different observation types
- **VQ-VAE**: Vector Quantized Variational Autoencoder for discrete latent representations
- **State Estimator**: Networks for estimating hidden states from partial observations
