# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp.commands.path_command import PathCommand


def explore_config_space(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    update_rate_steps: float,
    command_name: str = "path_command",
) -> float:
    """Curriculum that explores the configuration space of the task.

    Args:
        env: The learning environment.
        env_ids: The environment ids.
        update_rate_steps: The number of steps after which the path and terrain are modified.

    Returns:
        The num of positive param - the num of negative param.

    """
    # extract the used quantities (to enable type-hinting)
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    unique_params, inverse_idx, unique_counts = torch.unique(
        path_cmd_generator.param_sets[env_ids, :], return_inverse=True, return_counts=True, dim=0
    )
    # unique_params: (num_unique_params, num_params = 4)
    # inverse_idx: (num_envs,)
    # unique_counts: (num_unique_params,)
    id = 0
    curriculum_params_list = []
    curriculum_speed_list = []
    score = 0
    for param_set in unique_params:
        # get the indices for envs belonging to this parameter set
        envs = env_ids[inverse_idx == id]  # (unique_counts[id],)
        assert envs.shape[0] == unique_counts[id], "The number of environments does not match the unique counts."
        # remove the abnormally terminated environments
        terminate = env.termination_manager.terminated.float()[envs]
        envs = envs[terminate == 0]
        if envs.shape[0] == 0:
            id += 1
            continue
        # get the robot's performance under this parameter set
        pos_tolerance = path_cmd_generator.pos_err_tolerance[
            envs
        ].mean()  # pos_tolerance is the same for all envs in the same parameter set
        maximum_speed = (path_cmd_generator.maximum_speed[envs])[
            0
        ]  # maximum_speed is the same for all envs in the same parameter set
        minimum_progress = torch.min(path_cmd_generator.path_progress[envs])
        maximum_track_error = torch.max(path_cmd_generator.metrics["position_error_2d"][envs])
        minimum_robot_speed = torch.min(path_cmd_generator.metrics["robot_speed"][envs])

        idx = path_cmd_generator.path_cfg_generator.param_to_grid_idx(param_set)
        idx = tuple(idx.tolist())
        # Store the data for the curriculum
        if minimum_progress > 0:
            # Initial step should not be counted since the data is invalid
            path_cmd_generator.path_cfg_generator.curriculum_counter[idx] += 1
            path_cmd_generator.path_cfg_generator.minimum_progress[idx] = torch.minimum(
                path_cmd_generator.path_cfg_generator.minimum_progress[idx], minimum_progress
            )
            path_cmd_generator.path_cfg_generator.maximum_track_error[idx] = torch.maximum(
                path_cmd_generator.path_cfg_generator.maximum_track_error[idx], maximum_track_error
            )
            path_cmd_generator.path_cfg_generator.minimum_robot_speed[idx] = torch.minimum(
                path_cmd_generator.path_cfg_generator.minimum_robot_speed[idx], minimum_robot_speed
            )
            if path_cmd_generator.cfg.convert_to_vel:
                average_speed_error = torch.mean(path_cmd_generator.metrics["speed_error"][envs])
                path_cmd_generator.path_cfg_generator.average_speed_error[idx] += average_speed_error

        # print(f"idx: {idx}, minimum_progress: {minimum_progress}, maximum_track_error: {maximum_track_error}, average_speed_error: {average_speed_error}")
        # Update the distribution based on the robot's performance
        if env.common_step_counter > path_cmd_generator.last_update_config_env_step + update_rate_steps:
            # Take the average for previous steps
            curriculum_counter = path_cmd_generator.path_cfg_generator.curriculum_counter[idx]
            minimum_progress = path_cmd_generator.path_cfg_generator.minimum_progress[idx]
            maximum_track_error = path_cmd_generator.path_cfg_generator.maximum_track_error[idx]
            minimum_robot_speed = path_cmd_generator.path_cfg_generator.minimum_robot_speed[idx]
            if path_cmd_generator.cfg.convert_to_vel:
                average_speed_error = (
                    path_cmd_generator.path_cfg_generator.average_speed_error[idx] / curriculum_counter
                )
            # print(f"num: {curriculum_counter}, total_progress: {minimum_progress}, total_track_error: {maximum_track_error}, total_speed_error: {average_speed_error}")
            # Performances are good, increase the path length and terrain difficulty
            if not path_cmd_generator.cfg.convert_to_vel and curriculum_counter > 0:
                if minimum_progress > 0.8 and maximum_track_error < pos_tolerance and minimum_robot_speed > 1.0:
                    curriculum_params_list.append(param_set.tolist())
                    curriculum_speed_list.append(minimum_robot_speed.clone())
                    score += 1

                # Performances are bad, decrease the path length and terrain difficulty
                if minimum_progress < 0.5 or maximum_track_error > pos_tolerance * 1.5 or minimum_robot_speed < 1.0:
                    curriculum_params_list.append(param_set.tolist())
                    curriculum_speed_list.append(0.0)
                    score -= 1
            elif path_cmd_generator.cfg.convert_to_vel and curriculum_counter > 0:
                if maximum_track_error < pos_tolerance and average_speed_error < 0.2:
                    curriculum_params_list.append(param_set.tolist())
                    curriculum_speed_list.append(maximum_speed + 0.2)
                    score += 1
                if maximum_track_error > pos_tolerance * 1.5 or average_speed_error > 0.3:
                    curriculum_params_list.append(param_set.tolist())
                    curriculum_speed_list.append(0.0)
                    score -= 1

            # Reset the average values and counter
            path_cmd_generator.path_cfg_generator.minimum_progress[idx] = 1.0
            path_cmd_generator.path_cfg_generator.maximum_track_error[idx] = 0.0
            path_cmd_generator.path_cfg_generator.minimum_robot_speed[idx] = 10.0
            path_cmd_generator.path_cfg_generator.curriculum_counter[idx] = 0
            if path_cmd_generator.cfg.convert_to_vel:
                path_cmd_generator.path_cfg_generator.average_speed_error[idx] = 0.0

        id += 1

    if (
        len(curriculum_params_list) > 0
    ):  # which also means env step counter > last resample env step + update rate steps
        print(f"curriculum_speed_list: {curriculum_speed_list}")
        path_cmd_generator.update_path_config(curriculum_params_list, curriculum_speed_list)

    return float(score)


def show_distribution_speed(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str = "path_command") -> float:
    """Curriculum that shows the distribution of the robot's speed."""
    # extract the used quantities (to enable type-hinting)
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    path_cfg = path_cmd_generator.path_cfg_generator
    # calculate the average speed within explored grid
    explored_indices = torch.nonzero(path_cfg.grid > 0, as_tuple=False)
    if explored_indices.numel() == 0:
        return 0.0
    speeds = path_cfg.grid[tuple(explored_indices.T)]
    average_speed = speeds.mean().item()
    return average_speed


def show_distribution_terrain(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str = "path_command") -> float:
    """Curriculum that shows the distribution of the terrain level."""
    # extract the used quantities (to enable type-hinting)
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    path_cfg = path_cmd_generator.path_cfg_generator
    # calculate the average terrain level within explored grid
    explored_indices = torch.nonzero(path_cfg.grid > 0, as_tuple=False)
    if explored_indices.numel() == 0:
        return 0.0
    terrain_levels = explored_indices[:, 3]
    average_terrain = terrain_levels.float().mean().item()
    return average_terrain


def show_path_progress(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str = "path_command") -> float:
    """Curriculum that shows the progress of the robot."""
    # extract the used quantities (to enable type-hinting)
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    # calculate the average progress of the robot
    minimum_progress = torch.mean(path_cmd_generator.path_progress[env_ids])
    return minimum_progress.item()


def show_position_error_tolerance(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str = "path_command") -> float:
    """Curriculum that shows the position error tolerance."""
    # extract the used quantities (to enable type-hinting)
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    # calculate the average progress of the robot
    average_pos_err_tolerance = torch.mean(path_cmd_generator.pos_err_tolerance[env_ids])
    return average_pos_err_tolerance.item()


def modify_reward_weight_linearly(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial_weight: float,
    final_weight: float,
    start_step: int,
    end_step: int | None = None,
) -> float:
    """Curriculum that modifies a reward weight linearly.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        initial weight: The weight of the reward term at the beginning of the curriculum.
        final weight: The weight of the reward term at the end of the curriculum.
        start_step: The step at which the curriculum starts.
        end_step: The step at which the curriculum ends. If not set, there is a single step curriculum.

    Returns:
        The new weight of the reward term.

    NOTE: env.common_step_counter = learning_iterations * num_steps_per_env)
    """
    # obtain term settings
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # compute the new weight
    if end_step is None:
        end_step = start_step + 1
    num_steps = end_step - start_step
    weight = (
        initial_weight
        + (min(end_step - start_step, max(0.0, env.common_step_counter - start_step)))
        * (final_weight - initial_weight)
        / num_steps
    )
    # update term settings
    term_cfg.weight = weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return float(weight)


def modify_reward_std_linearly(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial_value: float,
    final_value: float,
    start_step: int,
    end_step: int | None = None,
) -> float:
    """Curriculum that modifies a reward standard deviation linearly.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        initial value: The value of the reward term at the beginning of the curriculum.
        final value: The value of the reward term at the end of the curriculum.
        start_step: The step at which the curriculum starts.
        end_step: The step at which the curriculum ends. If not set, there is a single step curriculum.

    Returns:
        The new value of the reward term.

    NOTE: env.common_step_counter = learning_iterations * num_steps_per_env)
    """
    # obtain term settings
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # compute the new weight
    if end_step is None:
        end_step = start_step + 1
    num_steps = end_step - start_step
    value = (
        initial_value
        + (min(end_step - start_step, max(0.0, env.common_step_counter - start_step)))
        * (final_value - initial_value)
        / num_steps
    )
    # update term settings
    term_cfg.params["std"] = value
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return float(value)


def modify_push_velocity(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial_config: dict,
    final_config: dict,
    start_step: int,
    end_step: int | None = None,
) -> float:
    """Curriculum that modifies the push velocity linearly.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        initial_config: The initial configuration of the push velocity.
        final_config: The final configuration of the push velocity.
        start_step: The step at which the curriculum starts.
        end_step: The step at which the curriculum ends. If not set, there is a single step curriculum.

    Returns:
        The total push velocity.

    NOTE: env.common_step_counter = learning_iterations * num_steps_per_env)
    """
    # obtain term settings
    term_cfg = env.event_manager.get_term_cfg(term_name)
    # compute the new weight
    if end_step is None:
        end_step = start_step + 1
    num_steps = end_step - start_step

    # compute the new configuration
    push_vx = (
        np.array(initial_config["x"])
        + (min(end_step - start_step, max(0.0, env.common_step_counter - start_step)))
        * (np.array(final_config["x"]) - np.array(initial_config["x"]))
        / num_steps
    )
    push_vy = (
        np.array(initial_config["y"])
        + (min(end_step - start_step, max(0.0, env.common_step_counter - start_step)))
        * (np.array(final_config["y"]) - np.array(initial_config["y"]))
        / num_steps
    )
    push_vz = (
        np.array(initial_config["z"])
        + (min(end_step - start_step, max(0.0, env.common_step_counter - start_step)))
        * (np.array(final_config["z"]) - np.array(initial_config["z"]))
        / num_steps
    )

    # update term settings
    term_cfg.params["velocity_range"]["x"] = push_vx
    term_cfg.params["velocity_range"]["y"] = push_vy
    term_cfg.params["velocity_range"]["z"] = push_vz
    env.event_manager.set_term_cfg(term_name, term_cfg)
    return float(np.sqrt(push_vx[1] ** 2 + push_vy[1] ** 2 + push_vz[1] ** 2))


def set_termination_alpha(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial_alpha: float,
    final_alpha: float,
    start_step: int,
    end_step: int | None = None,
) -> float:
    """Curriculum that modifies the termination alpha linearly.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        initial_alpha: The initial termination alpha.
        final_alpha: The final termination alpha.
        start_step: The step at which the curriculum starts.
        end_step: The step at which the curriculum ends. If not set, there is a single step curriculum.

    Returns:
        The new termination alpha.
    """
    # obtain term settings
    term_cfg = env.termination_manager.get_term_cfg(term_name)
    # compute the new weight
    if end_step is None:
        end_step = start_step + 1
    num_steps = end_step - start_step
    alpha = (
        initial_alpha
        + (min(end_step - start_step, max(0.0, env.common_step_counter - start_step)))
        * (final_alpha - initial_alpha)
        / num_steps
    )
    # update term settings
    term_cfg.params["alpha"] = alpha
    env.termination_manager.set_term_cfg(term_name, term_cfg)
    return float(alpha)


def set_tolerance_res(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    update_rate_steps: float,
    start_step: int,
    end_step: int | None = None,
    decay_rate: float = 0.5,
) -> float:
    """Curriculum that modifies the tolerance resolution linearly.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        update_rate_steps: The number of steps after which the curriculum is updated.
        start_step: The step at which the curriculum starts.
        end_step: The step at which the curriculum ends. If not set, there is a single step curriculum.
        decay_rate: The rate at which the tolerance resolution decays.

    Returns:
        The new tolerance resolution.
    """
    # obtain term settings
    path_cmd_generator: PathCommand = env.command_manager.get_term(term_name)
    # compute the new weight
    if end_step is None:
        end_step = start_step + 1

    # Track the last step where the update happened
    if not hasattr(env, "last_update_step"):
        env.last_update_step = 0

    if env.common_step_counter > start_step and env.common_step_counter >= env.last_update_step + update_rate_steps:
        env.last_update_step = env.common_step_counter
        path_cmd_generator.tol_res_dynamic *= decay_rate
        path_cmd_generator.sample_paths()
    return float(path_cmd_generator.tol_res_dynamic)
