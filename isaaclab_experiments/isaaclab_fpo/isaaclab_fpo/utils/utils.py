# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import git
import importlib
import os
import pathlib
import torch
import types
from typing import Any, Callable


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat(
        (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0])
    )
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(
        tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list
    )
    # add at least one full length trajectory
    trajectories = trajectories + (
        torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device),
    )
    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(
        0, tensor.shape[0], device=tensor.device
    ).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def _repo_root_from_source() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "isaaclab_experiments").is_dir():
            return parent
    return pathlib.Path(__file__).resolve().parents[4]


def localize_saved_config_paths(
    cfg: Any, repo_root: str | pathlib.Path | None = None
) -> list[tuple[str, str]]:
    """Remap stale absolute paths in a copied saved config to this checkout."""

    repo_root_path = (
        _repo_root_from_source()
        if repo_root is None
        else pathlib.Path(repo_root).expanduser().resolve()
    )
    remapped_paths: list[tuple[str, str]] = []
    seen: set[int] = set()

    def _remap_path_string(value: str) -> str:
        if "://" in value:
            return value

        expanded_path = pathlib.Path(os.path.expandvars(os.path.expanduser(value)))
        if not expanded_path.is_absolute() or expanded_path.exists():
            return value

        for marker in (
            "isaaclab_experiments/",
            "manipulation_experiments/",
            "fpo-control/",
        ):
            marker_index = value.find(marker)
            if marker_index == -1:
                continue

            if marker == "fpo-control/":
                rel_path = value[marker_index + len(marker) :]
            else:
                rel_path = value[marker_index:]
            candidate = repo_root_path / rel_path
            if candidate.exists():
                remapped_value = str(candidate)
                remapped_paths.append((value, remapped_value))
                return remapped_value

        return value

    def _visit(value: Any) -> Any:
        if isinstance(value, str):
            return _remap_path_string(value)
        if isinstance(value, pathlib.Path):
            return pathlib.Path(_remap_path_string(str(value)))
        if isinstance(
            value,
            (
                bytes,
                int,
                float,
                bool,
                torch.Tensor,
                types.ModuleType,
                types.FunctionType,
                type,
                type(None),
            ),
        ) or callable(value):
            return value

        value_id = id(value)
        if value_id in seen:
            return value
        seen.add(value_id)

        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = _visit(item)
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = _visit(item)
            return value
        if isinstance(value, tuple):
            return tuple(_visit(item) for item in value)
        if isinstance(value, set):
            updated_items = {_visit(item) for item in value}
            value.clear()
            value.update(updated_items)
            return value

        if hasattr(value, "__dict__"):
            for attr_name, attr_value in vars(value).items():
                try:
                    setattr(value, attr_name, _visit(attr_value))
                except (AttributeError, TypeError):
                    pass

        return value

    _visit(cfg)
    return list(dict.fromkeys(remapped_paths))


def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
            t = repo.head.commit.tree
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            # skip if not a git repository
            continue
        # get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        # check if the diff file already exists
        if os.path.isfile(diff_file_name):
            continue
        # write the diff file
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x", encoding="utf-8") as f:
            content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            f.write(content)
        # add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths


def string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    Args:
        name (str): The function name. The format should be 'module:attribute_name'.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When unable to resolve the attribute.

    Returns:
        Callable: The function loaded from the module.
    """
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as e:
        msg = (
            "We could not interpret the entry as a callable object. The format of input should be"
            f" 'module:attribute_name'\nWhile processing input '{name}', received the error:\n {e}."
        )
        raise ValueError(msg)
