# Whole-Body Tracking Checkpoint Workflow

This note covers the local G1 whole-body tracking run:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/train.py \
    --task Tracking-Flat-G1-v0 \
    --headless \
    env.commands.motion.motion_file=whole_body_tracking_reference_data/walk1_subject1.npz
```

The stopped checkpoint of interest is:

```text
logs/isaaclab_fpo/g1_flat_motion_tracking/2026-06-03_11-25-31/model_12000.pt
```

The saved run config in `params/env.pkl` preserves the `motion_file` override.

## Visualize In Isaac Sim

Use the regular play script when you want the Isaac viewer or a rendered video. The script loads the saved `params/env.pkl` and `params/agent.pkl` next to the checkpoint, so tracking motion overrides are preserved.

Interactive viewer:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/play.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint logs/isaaclab_fpo/g1_flat_motion_tracking/2026-06-03_11-25-31/model_12000.pt \
    --num_envs 1 \
    --real-time
```

Headless video:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/play.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint logs/isaaclab_fpo/g1_flat_motion_tracking/2026-06-03_11-25-31/model_12000.pt \
    --num_envs 1 \
    --headless \
    --video \
    --video_length 500
```

Videos are written under the checkpoint run directory:

```text
logs/isaaclab_fpo/g1_flat_motion_tracking/2026-06-03_11-25-31/videos/play/
```

## Visualize In Viser

Generate the tracking-specific Viser bundle first:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/isaac_asset_extractor.py --task Tracking-Flat-G1-v0
```

This writes:

```text
isaaclab_fpo/viser_assets/tracking_flat_g1_v0
```

Then run:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/play_with_viser.py \
    --task Tracking-Flat-G1-v0 \
    --checkpoint logs/isaaclab_fpo/g1_flat_motion_tracking/2026-06-03_11-25-31/model_12000.pt \
    --headless \
    --viser \
    --num_envs 1 \
    --real-time \
    --viser-port 8080
```

Open:

```text
http://localhost:8080
```

For the tracking task, Viser does not request a `base_velocity` command. It also overlays target body frames from the `motion` command term, so the robot mesh can be compared against the motion reference.

## Resume Training

The FPO runner loads `current_learning_iteration` from the checkpoint. Its `learn(num_learning_iterations=...)` loop treats that value as "additional iterations", not an absolute final iteration.

To resume from `model_12000.pt` and stop at absolute iteration 20000, request 8000 more iterations:

```bash
cd /home/roy/fpo-control/isaaclab_experiments
python isaaclab_fpo/scripts/train.py \
    --task Tracking-Flat-G1-v0 \
    --headless \
    --resume \
    --load_run 2026-06-03_11-25-31 \
    --checkpoint model_12000.pt \
    --max_iterations 8000 \
    env.commands.motion.motion_file=whole_body_tracking_reference_data/walk1_subject1.npz
```

This writes a new timestamped run under:

```text
logs/isaaclab_fpo/g1_flat_motion_tracking/
```

The command includes the `motion_file` override explicitly so the resumed env matches the original run.
