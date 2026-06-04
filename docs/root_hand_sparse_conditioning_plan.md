# Root And Hand Joint-Command Sparse Conditioning Plan

This plan describes a first sparse-conditioning ablation for the Isaac Lab G1
tracking task in this FPO++ repo.

The experiment is **inspired by** the PHC/FPO humanoid sparse-conditioning
experiment, but the current scope is deliberately narrower:

- We will mask only the current FPO++ actor's joint-space reference command.
- We will keep the current actor observation shape unchanged.
- We will keep reward, termination, critic observations, and motion data
  unchanged.
- We will not add body-space root/hand target observations in this plan.
- We will not preserve backwards compatibility with policies or `agent.pkl`
  files trained before these changes.

This means the first implementation is not a faithful reproduction of PHC/FPO's
body-space root/hand conditioning. It is a local FPO++ joint-command
sparse-conditioning ablation designed to test whether FPO++ flow policies can
express desirable tracking behavior from limited conditional input.

## Source Experiment To Adapt

Primary references:

- Paper: https://arxiv.org/html/2507.21053v2, Section 4.3, "Humanoid Control".
- PHC/FPO repo: https://github.com/akanazawa/fpo/tree/main/phc.
- PHC/FPO README: https://github.com/akanazawa/fpo/tree/main/phc#training-fpo-policies.
- Full-conditioning config: https://raw.githubusercontent.com/akanazawa/fpo/main/phc/configs/fpo.ini.
- Hand-conditioning config: https://raw.githubusercontent.com/akanazawa/fpo/main/phc/configs/fpo_hand.ini.
- Root-conditioning config: https://raw.githubusercontent.com/akanazawa/fpo/main/phc/configs/fpo_root.ini.
- Mask implementation: https://raw.githubusercontent.com/akanazawa/fpo/main/phc/puffer_phc/policy.py.
- PHC task observation implementation: https://raw.githubusercontent.com/akanazawa/fpo/main/phc/puffer_phc/humanoid_phc.py.

The PHC/FPO paper evaluates policies that receive full proprioception and sparse
goal information while still being rewarded for full-body imitation. It compares
full conditioning, root-only conditioning, and root+hands conditioning. This
plan uses that result as motivation for an expressivity ablation, not as an
exact reproduction target.

The important PHC/FPO implementation idea to reuse is shape-preserving masking:
the policy input size is not reduced. Instead, structured masks are applied to
goal-condition features. This repo should reuse that implementation pattern, but
apply it to the local joint-space command block.

## Local Repo State

Relevant local paths:

- Training: `isaaclab_experiments/isaaclab_fpo/scripts/train.py`.
- Playback with Viser: `isaaclab_experiments/isaaclab_fpo/scripts/play_with_viser.py`.
- FPO actor/critic: `isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/modules/actor_critic.py`.
- FPO config dataclasses: `isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/rl_cfg.py`.
- FPO task registry: `isaaclab_experiments/isaaclab_fpo/isaaclab_fpo/task_cfgs.py`.
- Tracking observation config: `isaaclab_experiments/thirdparty/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`.
- G1 tracking body list: `isaaclab_experiments/thirdparty/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`.
- Motion command source: `isaaclab_experiments/thirdparty/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`.
- Motion observation helpers: `isaaclab_experiments/thirdparty/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py`.

Current `Tracking-Flat-G1-v0` actor observation layout is 160-D:

```text
command                 58 = motion.joint_pos(29) + motion.joint_vel(29)
motion_anchor_pos_b      3
motion_anchor_ori_b      6
base_lin_vel             3
base_ang_vel             3
joint_pos               29
joint_vel               29
actions                 29
```

The current FPO actor MLP input for the G1 tracking config is 197-D:

```text
160 actor observation + 8 timestep embedding + 29 noised action = 197
```

The local `command` term is joint-space:

```python
MotionCommand.command = torch.cat([self.joint_pos, self.joint_vel], dim=1)
```

The torso/root-like goal information currently available to the actor is in
`motion_anchor_pos_b` and `motion_anchor_ori_b`. For G1 this anchor body is
`torso_link`. It is different from joint position commands: joint targets define
internal robot posture, while the anchor target tells the policy how the
reference torso is positioned/oriented relative to the current robot torso.

The critic observation is larger and privileged. Keep it full by default. The
sparse-conditioning change should affect only actor conditioning.

Training currently normalizes observations in
`isaaclab_fpo/runners/on_policy_runner.py` after the first environment step, but
the initial observation fetched at the start of `learn(...)` is not normalized
before the first `FPO.act(...)` call. Fix that as part of this implementation:
normalize the initial actor and critic observations before entering the rollout
loop. Playback already uses `get_inference_policy(...)`, which wraps
`policy.act_inference(obs_normalizer(obs))`. Therefore the mask should be
applied inside `ActorCritic` to the normalized actor observation tensor, before
concatenating timestep and noised action.

## Current Scope

This plan implements only shape-preserving masks over the current 160-D actor
observation. The sparse modes mean:

- `full`: keep all reference joint positions and velocities.
- `root`: mask all reference joint positions and velocities; keep the torso
  anchor pose target and proprioception.
- `root_hands`: keep torso anchor target, proprioception, and selected arm/wrist
  joint reference commands; mask all other reference joint commands.

The name `root_hands` is kept for continuity with PHC/FPO terminology, but in
this plan it means a **joint-command proxy**, not Cartesian hand-position
conditioning. If we want a more explicit name later, `root_arm_joints` or
`root_hand_joints` would be more accurate.

## Future Body-Space Conditioning

Faithful PHC-style root/hand conditioning should be added later as a separate
body-space observation extension. That future work should add explicit actor
goal terms derived from `MotionCommand.body_pos_relative_w`,
`body_quat_relative_w`, body velocities, and related robot body fields. Root and
hand masks should then keep actual root and wrist/hand body target positions.

That future body-space path is intentionally outside the scope of this plan.
Do not mix it into the first joint-command masking implementation.

## Joint-Command Mask Implementation

### Public Config Additions

Add these fields to `FpoRslRlPpoActorCriticCfg` in `rl_cfg.py`:

```python
condition_mode: Literal["full", "root", "root_hands"] = "full"
condition_drop_ratio: float = 0.0
condition_joint_names: list[str] | None = None
condition_joint_indices: list[int] | None = None
condition_include_command_vel: bool = True
condition_mask_debug: bool = False
```

Semantics:

- `condition_mode="full"`: mask is all ones. This must preserve current behavior.
- `condition_drop_ratio=0.0`: no condition masking. This must preserve current
  behavior even if `condition_mode` is not `full`.
- `condition_drop_ratio=1.0`: deterministically apply the selected sparse mask.
- `0.0 < condition_drop_ratio < 1.0`: reserved for future rollout-stable
  stochastic dropout. Do not support this in the first implementation.
- `condition_include_command_vel=True`: keep matching command velocity entries
  for selected `root_hands` joints. This intentionally makes the joint-command
  proxy stronger than PHC/FPO's body-position-only hand signal; this experiment
  is about limited-input flow-policy expressivity, not exact PHC/FPO parity.

Read these fields directly from the config. Do not add compatibility shims for
old `agent.pkl` files or checkpoints from before this sparse-conditioning
implementation.

Validate `condition_drop_ratio` in the first implementation:

```python
if condition_drop_ratio not in (0.0, 1.0):
    raise ValueError(
        "Stochastic condition dropout is not supported yet. "
        "Use condition_drop_ratio=0.0 or 1.0."
    )
```

### Why Stochastic Dropout Is Deferred

FPO++ stores old CFM losses during rollout and recomputes current CFM losses
during update:

```text
log_ratio = old_cfm_loss - current_cfm_loss
```

If condition dropout is randomly resampled inside `ActorCritic.get_cfm_loss(...)`
during rollout and update, the ratio may compare different conditioning masks for
the same transition. That adds variance and changes the meaning of the FPO
ratio.

For this first experiment, deterministic full/sparse masking is enough:

- `0.0`: always full condition.
- `1.0`: always sparse condition.

If mixed full/sparse robustness is needed later, sample the condition mask once
per rollout transition and reuse that same mask for action sampling, old CFM
loss computation, and current CFM loss recomputation. That likely requires
storing the sampled mask or masked actor observation in rollout storage.

### Mask Granularity

For the current 160-D observation, define fixed slices:

```text
command_pos             [0:29)
command_vel             [29:58)
motion_anchor_pos_b     [58:61)
motion_anchor_ori_b     [61:67)
base_lin_vel            [67:70)
base_ang_vel            [70:73)
joint_pos               [73:102)
joint_vel               [102:131)
actions                 [131:160)
```

Only the reference command block should be masked:

```text
command_pos [0:29)
command_vel [29:58)
```

Always keep:

- `motion_anchor_pos_b`
- `motion_anchor_ori_b`
- `base_lin_vel`
- `base_ang_vel`
- current `joint_pos`
- current `joint_vel`
- previous `actions`

For `full`, also keep all `command_pos` and `command_vel`.

For `root`, zero all `command_pos` and `command_vel`. The remaining target
information is the torso anchor target plus current proprioception.

For `root_hands`, keep selected arm/wrist entries inside `command_pos` and, when
`condition_include_command_vel=True`, the matching entries inside `command_vel`.
Mask all other reference joint command entries.

Recommended G1 arm/wrist joint names:

```text
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_roll_joint
left_wrist_pitch_joint
left_wrist_yaw_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
```

This is intentionally an arm-chain joint-command proxy. It gives more usable
hand-related information than wrist DOFs alone, because Cartesian hand placement
depends on shoulder and elbow joints as well as wrist joints. If a wrist-only
joint-command ablation is desired, pass only wrist joint names through
`condition_joint_names`.

Do not apply implicit defaults for `root_hands`. If
`condition_mode="root_hands"` and neither `condition_joint_names` nor
`condition_joint_indices` is provided, raise a clear `ValueError`, for example:

```python
raise ValueError(
    "condition_mode='root_hands' requires condition_joint_names or "
    "condition_joint_indices. Pass explicit arm/wrist joints for this robot."
)
```

Do not hardcode joint indices blindly. Resolve `condition_joint_indices` from
the env's robot joint order before runner construction, after the Isaac env is
created and before `OnPolicyRunner(...)` creates `ActorCritic`.

Recommended approach:

- Keep `condition_joint_names` user-facing.
- Add a small helper that reads `env.unwrapped.scene["robot"].data.joint_names`
  and fills `agent_cfg.policy.condition_joint_indices`.
- Validate that every requested joint name exists.
- If indices are provided directly, validate that every index is in range.
- If both names and indices are provided, resolve the names and verify they match
  the provided indices.
- Sanity-check command-column order. At minimum, assert that the motion command
  joint dimension matches `len(robot.data.joint_names)`. For current generated
  G1 LAFAN files, compare against the canonical `JOINT_NAMES` order in
  `whole_body_tracking_reference_data/download_lafan_data.py`; for other motion
  files, require equivalent joint-name metadata or clearly log that the command
  column order is assumed to match the robot joint order.
- Print resolved names/indices when `condition_mask_debug=True`.

### ActorCritic Changes

In `ActorCritic.__init__`:

- Store `self.condition_mode`.
- Store `self.condition_drop_ratio`.
- Store `self.condition_include_command_vel`.
- Register `self.condition_mask` as a non-persistent buffer with shape
  `[1, num_actor_obs]`:

```python
self.register_buffer("condition_mask", mask, persistent=False)
```

The buffer should be non-persistent because it is derived from
`condition_mode`, `condition_joint_indices`, and the current actor observation
layout. Persisting it risks stale masks when evaluating with different sparse
settings.

Build the mask once for `num_actor_obs == 160`. If `condition_mode != "full"` and
`num_actor_obs` is not the expected tracking observation size, raise a clear
`ValueError`.

If `condition_mode="root_hands"` and `condition_joint_indices` is missing or
empty, raise a clear `ValueError`. By the time `ActorCritic` is constructed, the
train/play script should already have resolved names to indices.

Add a helper:

```python
def _apply_condition_mask(self, observations: torch.Tensor) -> torch.Tensor:
    if self.condition_drop_ratio <= 0.0 or self.condition_mode == "full":
        return observations
    return observations * self.condition_mask
```

Apply this helper in:

- `act(...)`, before `_compiled_integrate_flow(...)`.
- `act_inference(...)`, before `_compiled_integrate_flow(...)`.
- `get_cfm_loss(...)`, before expanding actor observations.

Do not mask:

- timestep embedding.
- `x_t` noised action.
- critic observations.

Keep `_apply_condition_mask(...)` outside `_integrate_flow(...)` so the compiled
flow loop remains static and simple.

## Training Behavior

The primary comparison should be from-scratch training under the new
sparse-conditioning code. Do not treat policies trained before this
implementation as supported baselines.

Full training:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/train.py \
    --task Tracking-Flat-G1-v0 \
    --headless \
    agent.policy.condition_mode=full \
    agent.policy.condition_drop_ratio=0.0 \
    agent.run_name=full_condition \
    env.commands.motion.motion_file=whole_body_tracking_reference_data/walk1_subject1.npz
```

Root-only joint-command masking:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/train.py \
    --task Tracking-Flat-G1-v0 \
    --headless \
    agent.policy.condition_mode=root \
    agent.policy.condition_drop_ratio=1.0 \
    agent.run_name=root_joint_command_mask \
    env.commands.motion.motion_file=whole_body_tracking_reference_data/walk1_subject1.npz
```

Root+hands arm-chain joint-command masking:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/train.py \
    --task Tracking-Flat-G1-v0 \
    --headless \
    agent.policy.condition_mode=root_hands \
    agent.policy.condition_drop_ratio=1.0 \
    agent.policy.condition_joint_names='["left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"]' \
    agent.run_name=root_hands_joint_command_mask \
    env.commands.motion.motion_file=whole_body_tracking_reference_data/walk1_subject1.npz
```

For comparisons, run at least:

- full from scratch: `condition_mode=full`, `condition_drop_ratio=0.0`.
- root from scratch: `condition_mode=root`, `condition_drop_ratio=1.0`.
- root+hands from scratch: `condition_mode=root_hands`,
  `condition_drop_ratio=1.0`, and explicit `condition_joint_names` or
  `condition_joint_indices`.

Keep reward, termination, motion file, number of envs, and FPO hyperparameters
identical across those runs.

Resume/fine-tuning can still be used for diagnostics, but only from checkpoints
created with this sparse-conditioning implementation and with compatible config
fields.

## Playback Changes

Add these args to `play.py` and `play_with_viser.py`:

```text
--condition-mode full|root|root_hands
--condition-drop-ratio FLOAT
--condition-joint-names NAME [NAME ...]
--condition-joint-indices INDEX [INDEX ...]
--motion-file PATH
```

Apply them after loading `agent.pkl` and `env.pkl`, before constructing the
runner:

- If `--condition-mode` is provided, set `agent_cfg.policy.condition_mode`.
- If `--condition-drop-ratio` is provided, set
  `agent_cfg.policy.condition_drop_ratio`.
- If `--condition-joint-names` is provided, set
  `agent_cfg.policy.condition_joint_names`.
- If `--condition-joint-indices` is provided, set
  `agent_cfg.policy.condition_joint_indices`.
- If `--motion-file` is provided, set `env_cfg.commands.motion.motion_file`.
- Resolve `condition_joint_indices` after environment creation and before
  `OnPolicyRunner(...)`.

For evaluating sparse behavior, use `--condition-drop-ratio 1.0`.

Example:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/play_with_viser.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint logs/isaaclab_fpo/g1_flat_motion_tracking/<run>/<checkpoint.pt> \
    --headless \
    --viser \
    --num_envs 1 \
    --real-time \
    --condition-mode root_hands \
    --condition-drop-ratio 1.0 \
    --condition-joint-names \
        left_shoulder_pitch_joint left_shoulder_roll_joint left_shoulder_yaw_joint \
        left_elbow_joint left_wrist_roll_joint left_wrist_pitch_joint left_wrist_yaw_joint \
        right_shoulder_pitch_joint right_shoulder_roll_joint right_shoulder_yaw_joint \
        right_elbow_joint right_wrist_roll_joint right_wrist_pitch_joint right_wrist_yaw_joint \
    --motion-file whole_body_tracking_reference_data/walk1_subject1.npz
```

## Metrics And Acceptance Criteria

The joint-command masking implementation is accepted when:

- Existing full playback and training behavior is unchanged with defaults.
- `condition_mode=root` and `condition_mode=root_hands` run with the same actor
  input size and no MLP shape mismatch.
- `condition_drop_ratio=0.0` produces an all-ones effective mask.
- `condition_drop_ratio=1.0` deterministically applies the selected sparse mask
  during training and playback.
- `condition_mode=root_hands` raises a clear error unless names or indices are
  provided.
- Provided names/indices are validated against the robot joint order and command
  joint dimension.
- The initial training observation is normalized before the first action, so the
  mask is consistently applied to normalized actor observations when empirical
  normalization is enabled.
- Non-command observation fields are never masked.
- Training/play logs include selected condition mode, drop ratio, and resolved
  joint names/indices for `root_hands`.
- Viser startup still reports correct body mapping for the task.

Compare these metrics across full, root, and root+hands:

- `Mean reward`
- `Mean episode length`
- `Metrics/motion/error_anchor_pos`
- `Metrics/motion/error_anchor_rot`
- `Metrics/motion/error_body_pos`
- `Metrics/motion/error_body_rot`
- `Metrics/motion/error_joint_pos`
- `Metrics/motion/error_joint_vel`
- `Episode_Termination/anchor_pos`
- `Episode_Termination/anchor_ori`
- `Episode_Termination/ee_body_pos`

Use the same motion file for all comparisons unless explicitly testing
generalization to another motion.

## Implementation Order

1. Add config fields in `rl_cfg.py` with normal full-conditioning defaults.
2. Normalize the initial actor and critic observations in
   `OnPolicyRunner.learn(...)` before the first rollout action.
3. Add a helper to resolve `condition_joint_indices` from robot joint names in
   train/play scripts before runner construction.
4. Add validation for missing root+hands joints, index range, name/index
   consistency, and command/robot joint dimension/order.
5. Add deterministic condition-mask construction and application in
   `ActorCritic`.
6. Add playback overrides in `play.py` and `play_with_viser.py`.
7. Add debug logging for condition mode, drop ratio, and resolved joint mask.
8. Run syntax checks.
9. Run full-mode playback to confirm no regression.
10. Run root and root+hands playback with `condition_drop_ratio=1.0`.
11. Train full, root, and root+hands runs from scratch under matched settings.

Do not implement body-space root/hand target observations as part of this plan.

## Verification Commands

Syntax:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python -m py_compile \
    isaaclab_fpo/isaaclab_fpo/rl_cfg.py \
    isaaclab_fpo/isaaclab_fpo/modules/actor_critic.py \
    isaaclab_fpo/isaaclab_fpo/runners/on_policy_runner.py \
    isaaclab_fpo/scripts/play.py \
    isaaclab_fpo/scripts/play_with_viser.py
```

Full playback smoke test:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/play_with_viser.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint logs/isaaclab_fpo/g1_flat_motion_tracking/<run>/<checkpoint.pt> \
    --headless \
    --viser \
    --num_envs 1 \
    --real-time \
    --condition-mode full \
    --condition-drop-ratio 0.0
```

Root mask smoke test:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/play_with_viser.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint logs/isaaclab_fpo/g1_flat_motion_tracking/<run>/<checkpoint.pt> \
    --headless \
    --viser \
    --num_envs 1 \
    --real-time \
    --condition-mode root \
    --condition-drop-ratio 1.0
```

Root+hands joint-command mask smoke test:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/play_with_viser.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint logs/isaaclab_fpo/g1_flat_motion_tracking/<run>/<checkpoint.pt> \
    --headless \
    --viser \
    --num_envs 1 \
    --real-time \
    --condition-mode root_hands \
    --condition-drop-ratio 1.0 \
    --condition-joint-names \
        left_shoulder_pitch_joint left_shoulder_roll_joint left_shoulder_yaw_joint \
        left_elbow_joint left_wrist_roll_joint left_wrist_pitch_joint left_wrist_yaw_joint \
        right_shoulder_pitch_joint right_shoulder_roll_joint right_shoulder_yaw_joint \
        right_elbow_joint right_wrist_roll_joint right_wrist_pitch_joint right_wrist_yaw_joint
```

Stop each Viser smoke test after startup output confirms body mapping and the
first few policy steps run.

## Known Caveats

- `root_hands` is not PHC-style Cartesian hand-position conditioning. It is a
  joint-command proxy over selected arm/wrist target joints.
- `condition_include_command_vel=True` intentionally keeps matching reference
  joint velocities for selected root+hands joints. This makes the ablation
  stronger than PHC/FPO's body-position sparse signal, which is acceptable
  because this plan studies flow-policy expressivity under reduced joint-command
  conditioning rather than exact PHC/FPO reproduction.
- Keeping only wrist joint DOFs is possible but usually weak as a hand-location
  signal, because hand position depends heavily on shoulder and elbow joints.
- Policies/checkpoints created before this implementation are not a compatibility
  target.
- If the observation layout changes later, the hardcoded 160-D mask slices must
  be updated or replaced with observation-manager metadata.
- Body-space root/hand sparse conditioning remains a future extension and should
  be planned separately.
