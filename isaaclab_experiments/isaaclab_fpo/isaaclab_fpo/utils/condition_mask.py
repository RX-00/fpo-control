from __future__ import annotations

import argparse
import os
from typing import Any


G1_LAFAN_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def add_conditioning_args(parser: argparse.ArgumentParser) -> None:
    """Add sparse condition-mask playback overrides."""
    group = parser.add_argument_group(
        "condition masking", description="Sparse actor conditioning overrides."
    )
    group.add_argument(
        "--condition-mode",
        choices=("full", "root", "root_hands"),
        default=None,
        help="Actor condition-mask mode.",
    )
    group.add_argument(
        "--condition-drop-ratio",
        type=float,
        default=None,
        help="Condition mask ratio. Use 0.0 for full conditioning or 1.0 for sparse conditioning.",
    )
    group.add_argument(
        "--condition-joint-names",
        nargs="+",
        default=None,
        help="Joint names to keep for root_hands conditioning.",
    )
    group.add_argument(
        "--condition-joint-indices",
        nargs="+",
        type=int,
        default=None,
        help="Joint indices to keep for root_hands conditioning.",
    )
    group.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Override env.commands.motion.motion_file for playback.",
    )


def apply_conditioning_cli_overrides(agent_cfg: Any, env_cfg: Any, args_cli: argparse.Namespace) -> None:
    """Apply sparse condition-mask CLI overrides after loading saved configs."""
    if getattr(args_cli, "condition_mode", None) is not None:
        agent_cfg.policy.condition_mode = args_cli.condition_mode
    if getattr(args_cli, "condition_drop_ratio", None) is not None:
        agent_cfg.policy.condition_drop_ratio = args_cli.condition_drop_ratio
    if getattr(args_cli, "condition_joint_names", None) is not None:
        agent_cfg.policy.condition_joint_names = list(args_cli.condition_joint_names)
    if getattr(args_cli, "condition_joint_indices", None) is not None:
        agent_cfg.policy.condition_joint_indices = list(args_cli.condition_joint_indices)
    if getattr(args_cli, "motion_file", None) is not None:
        if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "motion"):
            raise ValueError("--motion-file requires an environment config with commands.motion")
        env_cfg.commands.motion.motion_file = args_cli.motion_file


def resolve_condition_joint_indices(env: Any, agent_cfg: Any) -> None:
    """Resolve and validate root_hands condition joints against the live robot order."""
    policy_cfg = agent_cfg.policy
    mode = getattr(policy_cfg, "condition_mode", "full")
    drop_ratio = getattr(policy_cfg, "condition_drop_ratio", 0.0)
    debug = getattr(policy_cfg, "condition_mask_debug", False)

    if mode not in ("full", "root", "root_hands"):
        raise ValueError(f"Unknown condition_mode: {mode}")
    if drop_ratio not in (0.0, 1.0):
        raise ValueError(
            "Stochastic condition dropout is not supported yet. "
            "Use condition_drop_ratio=0.0 or 1.0."
        )
    if drop_ratio <= 0.0 and not debug:
        if mode != "full":
            print(
                "[INFO] Condition mask inactive: "
                f"mode={mode}, drop_ratio={drop_ratio}"
            )
        return

    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    command_dim = _get_motion_command_joint_dim(env)
    if command_dim is not None and command_dim != len(joint_names):
        raise ValueError(
            "Motion command joint dimension does not match robot joint order: "
            f"command_dim={command_dim}, num_robot_joints={len(joint_names)}"
        )
    _check_g1_lafan_joint_order(joint_names, env)

    names = getattr(policy_cfg, "condition_joint_names", None)
    indices = getattr(policy_cfg, "condition_joint_indices", None)
    if mode == "root_hands" and drop_ratio > 0.0 and not names and not indices:
        raise ValueError(
            "condition_mode='root_hands' requires condition_joint_names or "
            "condition_joint_indices. Pass explicit arm/wrist joints for this robot."
        )

    resolved_indices: list[int] | None = None
    resolved_names: list[str] | None = None
    if names:
        name_to_index = {name: index for index, name in enumerate(joint_names)}
        missing = [name for name in names if name not in name_to_index]
        if missing:
            raise ValueError(
                "Unknown condition_joint_names: "
                f"{missing}. Available joints: {joint_names}"
            )
        resolved_indices = [name_to_index[name] for name in names]
        resolved_names = list(names)

    if indices:
        invalid = [index for index in indices if index < 0 or index >= len(joint_names)]
        if invalid:
            raise ValueError(
                "condition_joint_indices out of range: "
                f"{invalid}. Valid range is [0, {len(joint_names) - 1}]."
            )
        index_names = [joint_names[index] for index in indices]
        if resolved_indices is not None and list(indices) != resolved_indices:
            raise ValueError(
                "condition_joint_names resolve to indices "
                f"{resolved_indices}, but condition_joint_indices is {list(indices)}."
            )
        resolved_indices = list(indices)
        resolved_names = index_names

    if resolved_indices is not None:
        policy_cfg.condition_joint_indices = resolved_indices

    if mode != "full" or debug:
        print(
            "[INFO] Condition mask config: "
            f"mode={mode}, drop_ratio={drop_ratio}, "
            f"include_command_vel={getattr(policy_cfg, 'condition_include_command_vel', True)}"
        )
    if (mode == "root_hands" or debug) and resolved_indices is not None:
        print(f"[INFO] Condition joint indices: {resolved_indices}")
        print(f"[INFO] Condition joint names: {resolved_names}")


def _get_motion_command_joint_dim(env: Any) -> int | None:
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_term"):
        return None
    try:
        motion = command_manager.get_term("motion")
    except Exception:
        return None
    if not hasattr(motion, "motion") or not hasattr(motion.motion, "joint_pos"):
        return None
    return int(motion.motion.joint_pos.shape[1])


def _check_g1_lafan_joint_order(joint_names: list[str], env: Any) -> None:
    if len(joint_names) != len(G1_LAFAN_JOINT_NAMES):
        return
    motion_file = getattr(getattr(getattr(env.unwrapped, "cfg", None), "commands", None), "motion", None)
    motion_file = getattr(motion_file, "motion_file", "")
    if set(joint_names) == set(G1_LAFAN_JOINT_NAMES):
        if joint_names != G1_LAFAN_JOINT_NAMES:
            raise ValueError(
                "Robot joint names match the G1 LAFAN set but not its canonical order. "
                "The joint-command mask assumes command columns match robot joint order."
            )
        return
    if motion_file:
        print(
            "[WARNING] Motion command joint-name metadata is unavailable for "
            f"{os.path.basename(motion_file)}; assuming command columns match robot joint order."
        )
