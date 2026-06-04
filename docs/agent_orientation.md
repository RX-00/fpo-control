# Agent Orientation Guide

Generated from the repository state on 2026-06-04. This is meant for coding
agents and maintainers who need to understand where behavior lives before
editing the repo.

## What This Repository Is

This is the FPO++ code release for "Flow Policy Gradients for Robot Control".
The repo contains two mostly independent experiment workspaces:

- `isaaclab_experiments/`: Isaac Lab locomotion and whole-body motion tracking.
- `manipulation_experiments/`: Robosuite/DexMimicGen manipulation pretraining,
  online fine-tuning, evaluation, and plotting.

Both areas implement flow-matching policies. A flow policy learns a vector field
that transports noise to actions. At inference time, the policy integrates the
vector field from `t=1` to `t=0` with Euler steps. During training, conditional
flow matching (CFM) losses are used as the policy density or likelihood proxy.
FPO-style online updates compare old and current CFM losses to form a log ratio.

The two experiment directories have separate conda environments, setup scripts,
source scripts, dependencies, and README files. Treat them as separate projects
unless a change explicitly spans both.

## Top-Level Layout

- `README.md`: repo overview, setup pointers, license and third-party summary.
- `docs/`: repo-level docs for agents and maintainers.
- `isaaclab_experiments/`: Isaac Lab extension for FPO++ locomotion/tracking.
- `manipulation_experiments/`: behavior cloning, online RL, eval, plotting, and
  large vendored simulator/policy dependencies.
- `.gitmodules`: IsaacLab and whole_body_tracking submodules used by Isaac Lab.

The repo intentionally vendors or installs large dependencies under each
workspace. Avoid broad searches through `thirdparty/` unless the task is
specifically about vendored code.

## Shared Algorithm Vocabulary

- `FPO++`: flow policy optimization using per-sample CFM loss ratios.
- `Vanilla FPO`: manipulation variant that keeps closer to original FPO design,
  notably using grouped/per-action style ratio behavior.
- `DPPO`: diffusion/denoising PPO variants in manipulation. These use the
  realized denoising trajectory and SDE noise likelihood instead of the FPO CFM
  ratio.
- `CFM loss`: conditional flow matching objective. In most configs, the model
  predicts velocity `u`.
- `zero sampling`: initialize flow integration with all zeros. Used for
  deterministic eval in several scripts.
- `random sampling`: initialize flow integration from Gaussian noise.
- `EMA`: exponential moving average of flow model weights. Used heavily for
  checkpoint quality and evaluation.
- `ASPO/SPO/PPO`: trust-region modes in FPO updates. Isaac Lab defaults to ASPO
  in the base algorithm config; manipulation usually passes these through CLI.

## Isaac Lab Workspace

Path: `isaaclab_experiments/`

Purpose: train and evaluate FPO++ policies in NVIDIA Isaac Lab for
velocity-conditioned locomotion and G1 whole-body motion tracking.

Environment:

```bash
cd isaaclab_experiments
bash setup_env.sh
source source_env.sh
```

`setup_env.sh` initializes git submodules, installs a local miniconda under
`isaaclab_experiments/thirdparty/miniconda3`, creates the `isaaclab_fpo` conda
environment, installs Isaac Sim 4.5 / Isaac Lab / whole_body_tracking, installs
`isaaclab_fpo` editable, and downloads/converts LAFAN1 tracking reference data.

### Isaac Lab Entry Points

- `isaaclab_fpo/scripts/train.py`: main FPO training entry point. It launches
  Isaac Sim through `AppLauncher`, parses Isaac/Hydra args, applies sweep
  overrides, builds the env, wraps it, creates `OnPolicyRunner`, and trains.
- `isaaclab_fpo/scripts/play.py`: Isaac Sim viewer playback.
- `isaaclab_fpo/scripts/play_with_viser.py`: browser playback with Viser and
  optional W&B checkpoint download.
- `isaaclab_fpo/scripts/play_plot.py`: playback plus action/flow diagnostics.

Common commands:

```bash
cd isaaclab_experiments
source source_env.sh

python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-Spot-v0 --headless
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-H1-v0 --headless
python isaaclab_fpo/scripts/train.py --task Isaac-Velocity-Flat-G1-v0 --headless

python isaaclab_fpo/scripts/train.py --task Tracking-Flat-G1-v0 --headless

python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
  --logger wandb --log_project_name my-project --run_name trial_01
```

Hydra-style sweep overrides are accepted as positional args:

```bash
python isaaclab_fpo/scripts/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless \
  agent.algorithm.learning_rate=3e-4 \
  agent.algorithm.n_samples_per_action=32
```

### Isaac Lab Code Map

- `isaaclab_fpo/task_cfgs.py`
  - Per-task runner configs.
  - `TASK_CONFIGS` maps gym task IDs to config classes.
  - Add or tune Isaac tasks here first.
- `isaaclab_fpo/rl_cfg.py`
  - Config dataclasses for policy, algorithm, and runner.
  - Important knobs include `sampling_steps`, `n_samples_per_action`,
    `cfm_loss_reduction`, `clip_param`, `trust_region_mode`,
    `cfm_loss_clamp`, `advantage_clamp`, and EMA settings.
- `isaaclab_fpo/cli_args.py`
  - Adds FPO CLI flags and overlays them onto task configs.
- `isaaclab_fpo/wrapper.py`
  - `FpoRslRlVecEnvWrapper` adapts Isaac Lab environments to the RSL-RL style
    `VecEnv` API expected by the runner.
  - Handles policy/critic observation groups, action clipping, reset, dones,
    and timeout bootstrapping metadata.
- `isaaclab_fpo/modules/actor_critic.py`
  - `ActorCritic` owns the flow actor MLP and critic MLP.
  - `act()` samples initial noise in training and zero noise in eval, integrates
    the learned vector field, scales actions, and optionally adds action noise.
  - `get_cfm_loss()` computes per-action-sample CFM loss used by FPO ratios.
- `isaaclab_fpo/algorithms/fpo.py`
  - `FPO` owns rollout storage, optimizer, EMA, action sampling, return
    computation, and policy/value updates.
  - During rollout, it stores actions, old CFM losses, CFM noise samples, and
    timesteps. During update, it recomputes current CFM losses and uses
    `old_loss - current_loss` as the log ratio.
- `isaaclab_fpo/storage/rollout_storage.py`
  - Stores observations, privileged observations, actions, values, rewards,
    dones, old CFM samples/losses, returns, and advantages.
- `isaaclab_fpo/runners/on_policy_runner.py`
  - Coordinates environment rollout, normalization, `FPO.update()`, logging,
    checkpoint save/load, EMA checkpointing, post-training checkpoint eval, and
    optional distributed training.
- `isaaclab_fpo/modules/normalizer.py`
  - Empirical observation normalization.
- `isaaclab_fpo/modules/ema.py`
  - Actor EMA implementation.
- `isaaclab_fpo/exporter.py`
  - TorchScript and ONNX export helpers.
- `isaaclab_fpo/patches.py`
  - Isaac Lab monkey patches for sweep/config support. `train.py` applies this
    after launching Isaac Sim.
- `isaaclab_fpo/viser/`
  - Viser scene and asset integration for browser playback.

### Isaac Lab Training Flow

1. `train.py` launches Isaac Sim and imports task registrations.
2. `parse_env_cfg()` builds the Isaac Lab env config.
3. `cli_args.parse_fpo_cfg()` looks up `TASK_CONFIGS[task]`.
4. Optional positional `agent.*` and `env.*` overrides are applied.
5. `gym.make()` creates the Isaac env, then multi-agent envs are converted to
   single-agent when needed.
6. `FpoRslRlVecEnvWrapper` exposes the env through the runner interface.
7. `OnPolicyRunner` creates `ActorCritic`, `FPO`, normalizers, storage, and
   log writers.
8. Each iteration collects `num_steps_per_env` transitions, computes GAE
   returns, updates actor/critic using CFM ratios, steps EMA, logs, and saves.
9. Final and interval checkpoints are written under
   `isaaclab_experiments/logs/isaaclab_fpo/<experiment>/<timestamp>/`.

### Isaac Lab Tasks

Current FPO task registry:

- `Isaac-Velocity-Flat-Unitree-Go2-v0`
- `Isaac-Velocity-Flat-Spot-v0`
- `Isaac-Velocity-Flat-H1-v0`
- `Isaac-Velocity-Flat-G1-v0`
- matching `*-Play-v0` variants for locomotion tasks
- `Tracking-Flat-G1-v0`
- `Isaac-Cartpole-Direct-v0` for quick debugging

Motion tracking files are expected under
`isaaclab_experiments/whole_body_tracking_reference_data/`.

## Manipulation Workspace

Path: `manipulation_experiments/`

Purpose: pretrain flow-matching base policies from LeRobot datasets, then
fine-tune them online in Robosuite/DexMimicGen tasks with FPO++, Vanilla FPO,
and DPPO variants.

Environment:

```bash
cd manipulation_experiments
bash setup_env.sh
source source_env.sh
```

`setup_env.sh` installs a local miniconda under
`manipulation_experiments/thirdparty/miniconda3`, creates the
`fpo_manipulation` conda environment, installs vendored robosuite and lerobot,
clones DexMimicGen if needed, installs ffmpeg and Python dependencies, and
installs this workspace's editable package.

Package metadata lives in `manipulation_experiments/pyproject.toml`. The package
name is `far-manipulation-fpo`, and first-party modules are under `src`.

### Manipulation Entry Points

- `pretrain_flow_bc.py`: offline behavior cloning with flow matching.
- `finetune_online_rl.py`: online fine-tuning from a pretrained base policy.
- `eval_checkpoint.py`: evaluate pretrained or fine-tuned checkpoints from W&B
  or local paths.
- `plot_results.py`: fetch W&B data and reproduce plots.
- `scripts/run_pretrain_base_policies.sh`: launch all base-policy pretraining
  commands.
- `scripts/run_main_benchmark.sh`: launch main fine-tuning benchmark.
- `scripts/run_checkpoint_ablation.sh`: launch checkpoint ablations.
- `scripts/run_fpo_ablation.sh`: launch FPO ablations.
- `scripts/eval_base_policies.sh`: evaluate base policies.
- `scripts/skypilot/`: cloud launch wrappers.
- `scripts/sweeps/`: W&B sweep YAMLs.

Most helper scripts support:

```bash
DRY_RUN=1 bash scripts/run_main_benchmark.sh
NUM_GPUS=4 bash scripts/run_main_benchmark.sh
```

### Manipulation Code Map

- `src/flow_model_config.py`
  - `FlowMatchingConfig`, registered as LeRobot policy subclass
    `flowmatching`.
  - Holds horizon, `n_action_steps`, sampling steps, flow loss modes, MLP/U-Net
    architecture settings, vision backbone settings, normalization, EMA, and
    optimizer defaults.
- `src/flow_model.py`
  - `FlowMatchingPolicy`, a LeRobot `PreTrainedPolicy`.
  - Builds normalizers, the selected flow network, EMA, action buffers, chunked
    action selection, CFM loss paths, DPPO denoising likelihood, and optional
    noise injection.
- `src/flow_net_mlp.py`
  - MLP flow network with image/state encoders.
- `src/flow_net_residual_mlp.py`
  - Residual MLP variant.
- `src/flow_net_unet.py`
  - 1D U-Net action-sequence flow network.
- `src/vit.py`
  - Vision Transformer encoder adapted from DPPO.
- `src/noise_injection_network.py`
  - Learned SDE sigma network for DPPO learned-noise runs.
- `src/dexmg_env.py`
  - Robosuite/DexMimicGen environment wrapper and vectorized env factory.
  - Maps task aliases (`Can`, `Square`, `Transport`) to robosuite names.
  - Defines robot lists, task horizons, expected camera keys, low-dimensional
    state keys, image preprocessing, rendering, and Gymnasium vector wrappers.
- `src/utils.py`
  - General manipulation helpers.

### Manipulation Pretraining Flow

1. `TrainFlowBCConfig` is parsed by Tyro in `pretrain_flow_bc.py`.
2. LeRobot dataset metadata is fetched with `LeRobotDatasetMetadata`.
3. `FlowMatchingConfig` is created from CLI settings and dataset shapes.
4. `resolve_delta_timestamps()` determines frame/action offsets.
5. `LeRobotDataset` is built with image transforms and optional episode limits.
6. `FlowMatchingPolicy` is created with dataset statistics for normalization.
7. AdamW optimizes `policy.get_optim_params()` with a lower LR for vision
   backbone params.
8. The loop trains CFM loss on batches, logs to W&B, saves `policy/` plus
   `optimizer.pt`, and optionally runs DexMimicGen rollouts.
9. Checkpoints are stored under `runs/<experiment>_<timestamp>/checkpoints/`,
   with `latest` and optionally `best`.

Example:

```bash
cd manipulation_experiments
source source_env.sh

python pretrain_flow_bc.py \
  --dataset ankile/dexmg-two-arm-threading \
  --policy flowmatching \
  --network_architecture mlp \
  --horizon 8 \
  --n_action_steps 8 \
  --sampling_steps 10 \
  --image_observation_keys "agentview_image robot0_eye_in_hand_image robot1_eye_in_hand_image" \
  --eval_env TwoArmThreading \
  --eval_num_envs 1 \
  --eval_num_episodes 5 \
  --log_freq 10 \
  --save_freq 200 \
  --rollout_freq 200 \
  --steps 6000 \
  --wandb_enable True \
  --wandb_project flow-bc
```

### Manipulation Fine-Tuning Flow

1. `FlowPPOConfig` is parsed by Tyro in `finetune_online_rl.py`.
2. A base policy is loaded from either W&B artifacts or `--base_policy_local_path`.
3. `FlowMatchingConfig` is reconstructed from `policy/config.json`; weights are
   loaded from `policy/model.safetensors`; optional EMA state is read from
   `optimizer.pt`.
4. CLI overrides are applied to policy config values such as `n_action_steps`,
   `sampling_steps`, CFM loss mode, transported clipping, SDE sigma, and
   learned-noise settings.
5. A flow actor and separate MLP critic are created. The critic input is the
   encoded observation conditioning vector.
6. `create_vectorized_env()` creates Robosuite/DexMimicGen vector envs.
7. During rollout, the actor emits action chunks. The loop stores observations,
   actions, rewards, dones, values, old CFM losses/noise/timesteps, DPPO
   denoising paths, and invalid masks for chunks that cross episode ends.
8. The update phase reshapes stored rollout data by chunks, computes GAE
   returns, recomputes current CFM losses or DPPO log-probs, forms ratios,
   applies PPO/SPO/ASPO losses, updates actor and critic, steps EMA, logs, and
   checkpoints.
9. Periodic evaluation runs both zero-sampling and random-sampling modes.

Example:

```bash
cd manipulation_experiments
source source_env.sh

torchrun --nproc_per_node=1 finetune_online_rl.py \
  --distributed True \
  --base_policy_local_path downloaded_checkpoints/95j3noe4_step_1000 \
  --load-ema True \
  --task Can \
  --eval_env Can \
  --num_envs 4 \
  --n_action_steps 4 \
  --data-collection-steps 300 \
  --wandb_enable True \
  --wandb_project flow-bc-fpo-finetuning
```

### Manipulation Tasks And Data

Common dataset/task pairs:

- `ankile/robomimic-ph-can-image` -> `PickPlaceCan` or alias `Can`
- `ankile/robomimic-ph-square-image` -> `NutAssemblySquare` or alias `Square`
- `ankile/dexmg-two-arm-box-cleanup` -> `TwoArmBoxCleanup`
- `ankile/dexmg-two-arm-lift-tray` -> `TwoArmLiftTray`
- `ankile/dexmg-two-arm-threading` -> `TwoArmThreading`

`src/dexmg_env.py` controls task-specific robot models, horizons, camera keys,
and low-dimensional state keys. When adding a manipulation task, update
`ENV_ROBOTS`, alias mapping if needed, horizon mapping, image key logic, and
low-dimensional key logic together.

## Generated And Heavy Paths

Be careful with generated or very large paths:

- `isaaclab_experiments/thirdparty/`: submodules, local miniconda, Isaac Lab.
- `isaaclab_experiments/logs/`: training logs/checkpoints.
- `isaaclab_experiments/whole_body_tracking_reference_data/*.npz`: generated or
  downloaded motion data.
- `isaaclab_experiments/isaaclab_fpo/isaaclab_fpo.egg-info/`: generated package
  metadata.
- `manipulation_experiments/thirdparty/`: vendored robosuite, vendored lerobot,
  local miniconda, cloned DexMimicGen, and requirements snapshots.
- `manipulation_experiments/runs/`: generated experiment outputs.
- `manipulation_experiments/downloaded_checkpoints/`: downloaded base policies.
- `wandb/` directories: local W&B state.

The root `.gitignore` is minimal. Check `git status --short` before and after
work so generated files do not accidentally get included in commits.

## Common Edit Targets

- Add/tune Isaac Lab task hyperparameters:
  `isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/task_cfgs.py`.
- Change Isaac Lab global FPO defaults:
  `isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/rl_cfg.py`.
- Change Isaac Lab action sampling, CFM loss, or critic behavior:
  `isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/modules/actor_critic.py`.
- Change Isaac Lab PPO/FPO update logic:
  `isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/algorithms/fpo.py`.
- Change Isaac Lab training/checkpoint/log flow:
  `isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/runners/on_policy_runner.py`.
- Add manipulation task support:
  `manipulation_experiments/src/dexmg_env.py`.
- Change manipulation policy config surface:
  `manipulation_experiments/src/flow_model_config.py`.
- Change manipulation flow policy behavior:
  `manipulation_experiments/src/flow_model.py` and the selected
  `src/flow_net_*.py`.
- Change manipulation offline BC:
  `manipulation_experiments/pretrain_flow_bc.py`.
- Change manipulation online RL:
  `manipulation_experiments/finetune_online_rl.py`.
- Change manipulation benchmark command generation:
  `manipulation_experiments/scripts/*.sh` and `scripts/sweeps/*.yaml`.

## Verification Notes

There is no first-party pytest, ruff, black, mypy, tox, pre-commit, or Makefile
configuration in the repo at the time this doc was written. Verification is
therefore task-specific.

For docs-only edits:

```bash
git status --short docs/
git diff -- docs/
```

For Python edits, prefer syntax checks on the touched first-party files inside
the relevant environment:

```bash
cd isaaclab_experiments
source source_env.sh
python -m py_compile isaaclab_fpo/isaaclab_fpo/algorithms/fpo.py

cd manipulation_experiments
source source_env.sh
python -m py_compile pretrain_flow_bc.py finetune_online_rl.py src/flow_model.py
```

For manipulation script command changes, use dry runs when available:

```bash
cd manipulation_experiments
DRY_RUN=1 bash scripts/run_pretrain_base_policies.sh
DRY_RUN=1 bash scripts/run_main_benchmark.sh
DRY_RUN=1 bash scripts/run_checkpoint_ablation.sh
DRY_RUN=1 bash scripts/run_fpo_ablation.sh
DRY_RUN=1 bash scripts/eval_base_policies.sh
```

For Isaac Lab changes, even small runs require Isaac Sim and GPU setup. The
cartpole task is the smallest registered debug task, but it still runs through
Isaac:

```bash
cd isaaclab_experiments
source source_env.sh
python isaaclab_fpo/scripts/train.py --task Isaac-Cartpole-Direct-v0 --headless --max_iterations 1
```

Do not run either `setup_env.sh` casually. They install dependencies, download
large assets, and may clone repositories.

## Practical Pitfalls

- Run commands from the correct workspace directory. Many paths are relative.
- `train.py` imports `viser` before Isaac modules to avoid a websockets version
  conflict caused by Isaac path manipulation. Preserve that ordering unless you
  are fixing that issue directly.
- Isaac Lab task configs are registered in `TASK_CONFIGS`, not in gym kwargs.
- Manipulation CLI is Tyro-based. Some flags appear in examples with hyphens
  even though dataclass fields use underscores.
- Manipulation fine-tuning assumes chunk sizes divide collection lengths:
  `data_collection_steps % n_action_steps == 0`.
- Manipulation DDP removes `ema_model` for compatibility and logs from rank 0.
- Robosuite camera names and LeRobot feature names differ. `dexmg_env.py`
  converts camera keys like `agentview_image` to
  `observation.images.agentview`.
- Successful manipulation episodes terminate early when reward equals `1.0`.
- W&B artifact checkpoint structure is expected to contain `policy/config.json`,
  `policy/model.safetensors`, and often `optimizer.pt`.
- Generated `.npz`, `.egg-info`, conda, logs, and checkpoint directories may
  appear in `git status`; inspect before committing.
