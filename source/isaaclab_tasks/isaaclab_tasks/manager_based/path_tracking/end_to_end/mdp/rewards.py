# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp.commands.path_command import PathCommand

"""
Task rewards.
"""


def tracking_speed_up(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "path_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_distance_thresh: float = 0.5,
    tol_scaled_factor: float = 2.50,
    tol_scaled_offset: float = 0.75,
    clipped_speed=None,
    use_weighted_speed: bool = False,
) -> torch.Tensor:
    """Reward for achieving high linear speed along the xy plane."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    distance_goal = path_cmd_generator.path_length_command  # (num_envs, )
    ang_diff = (path_cmd_generator.path_command_b[:, 2, 2] - path_cmd_generator.path_command_b[:, 1, 2] + np.pi) % (
        2 * np.pi
    ) - np.pi
    desired_yaw = (path_cmd_generator.path_command_b[:, 1, 2] + ang_diff / 2 + np.pi) % (
        2 * np.pi
    ) - np.pi  # desired yaw angle in robot frame
    desired_directin_vector = torch.stack([torch.cos(desired_yaw), torch.sin(desired_yaw)], dim=1)  # (num_envs, 2)
    # compute the reward
    projected_speed = torch.sum(asset.data.root_lin_vel_b[:, :2] * desired_directin_vector, dim=1)  # (num_envs, )

    # clip the speed if backward movement is disabled and the heading is is not aligned with the path
    if not path_cmd_generator.cfg.enable_backward:
        projected_speed = torch.where(
        # asset.data.root_lin_vel_b[:, 0] < -0.1, torch.zeros_like(projected_speed), projected_speed
        asset.data.root_lin_vel_b[:, 0] < -0.1, projected_speed * 0.5, projected_speed
    )

    # clip the speed if required
    if clipped_speed is not None:
        projected_speed = torch.clip(projected_speed, min=-clipped_speed, max=clipped_speed)

    # Encourage large speed for high tolerance and low speed for low tolerance
    tolerance = path_cmd_generator.pos_err_tolerance  # (num_envs, )
    if use_weighted_speed:
        projected_speed = (tol_scaled_factor * tolerance + tol_scaled_offset) * projected_speed

    # Reward high speed agent before reaching the goal while rewarding zero speed agent after reaching the goal
    reward = torch.where(
        distance_goal > goal_distance_thresh, projected_speed / std, torch.exp(-torch.square(projected_speed))
    )
    return reward


def track_pos_xy_exp(
    env: ManagerBasedRLEnv, command_name: str = "path_command", soft_factor=0.9, use_tanh: bool = False
) -> torch.Tensor:
    """Reward tracking of position-2d (xy axes) along the path using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    tolerance = path_cmd_generator.pos_err_tolerance  # (num_envs, )
    # compute the error, use tolerance as the standard deviation
    track_pos_error = path_cmd_generator.curr_pos_error  # (num_envs, )
    if not use_tanh:
        # use clipped reward
        reward = torch.where(
            track_pos_error < soft_factor * tolerance,
            torch.ones_like(track_pos_error),
            -torch.ones_like(track_pos_error),
        )
        reward = torch.where(track_pos_error > tolerance, -2.0 * torch.ones_like(track_pos_error), reward)
    else:
        # use tanh reward
        diff = track_pos_error - soft_factor * tolerance
        diff = torch.clamp(diff, min=tolerance / 2.0)
        reward = 1.5 * torch.where(diff > 0, -torch.tanh(torch.pow(diff, 0.2)), torch.tanh(torch.pow(-diff, 0.2)))
    return reward


def track_pos_xy_cons(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str = "path_command",
) -> torch.Tensor:
    """Penalize tracking position-2d (xy axes) error along the path larger than the threshold."""
    # extract the used quantities (to enable type-hinting)
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    # compute the error
    track_pos_error = torch.square(path_cmd_generator.curr_pos_error)  # (num_envs, )
    reward = torch.where(
        track_pos_error > threshold, torch.abs(track_pos_error - threshold), torch.zeros_like(track_pos_error)
    )
    return reward


def path_progress(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "path_command"
) -> torch.Tensor:
    """Reward the agent for making progress along the path."""
    # extract the used quantities (to enable type-hinting)
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    return path_cmd_generator.path_progress


def track_yaw_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "path_command") -> torch.Tensor:
    """Reward tracking of yaw along the path using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    # compute the error
    track_yaw_error = torch.square(path_cmd_generator.curr_yaw_error)  # (num_envs, )
    reward = torch.exp(-track_yaw_error / std**2)
    return reward


"""
State penalties.
"""


def near_goal_stability(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_distance_thresh: float = 0.5,
    command_name: str = "path_command",
) -> torch.Tensor:
    """Reward the agent for being stable near the goal.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        goal_distance_thresh: The distance threshold to the goal.

    Returns:
        reward: the penalty for being unstable near the goal.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    # compute the reward
    distance_goal = path_cmd_generator.path_length_command  # (num_envs, )
    square_velocity = torch.norm(asset.data.root_ang_vel_w, dim=1, p=2) ** 2  # (num_envs, )
    reward = torch.where(distance_goal < goal_distance_thresh, square_velocity, torch.zeros_like(distance_goal))
    return reward


def no_robot_movement(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_distance_thresh: float = 0.75,
    command_name: str = "path_command",
) -> torch.Tensor:
    """Penalize the agent for having abs(vy) > 0 near the goal.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        goal_distance_thresh: The distance threshold to the goal.

    Returns:
        reward: the penalty for having abs(vy) > 0 near the goal.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    # compute the reward
    distance_goal = path_cmd_generator.path_length_command  # (num_envs, )
    # penalize the agent for having vy 0 near the goal
    square_velocity_y = torch.square(asset.data.root_lin_vel_w[:, 1])  # (num_envs, )
    reward = torch.where(distance_goal < goal_distance_thresh, square_velocity_y, torch.zeros_like(distance_goal))
    return reward


def compute_terrain_normal(hit_points):
    """Compute terrain normal from ray hit points using least squares plane fitting."""
    # Create the G matrix and Z vector
    G = torch.cat([hit_points[:, :, :2], torch.ones_like(hit_points[:, :, :1])], dim=-1)  # (N, B, 3)
    Z = hit_points[:, :, 2]  # (N, B)

    # Solve for the plane parameters using least squares
    params = torch.linalg.lstsq(G, Z).solution  # (N, 3)

    # Extract the plane parameters
    a = params[:, 0]
    b = params[:, 1]

    # Compute the normal vectors
    normals = torch.stack([a, b, -torch.ones_like(a)], dim=-1)  # (N, 3)

    # Normalize the normal vectors
    normals = normals / torch.norm(normals, dim=1, keepdim=True)  # (N, 3)

    return normals


def base_height_sensor(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    margin: float,
    higher_scale: float,
    command_name: str = "path_command",
) -> torch.Tensor:
    """Penalize asset height from its target using attached sensor.

    Args:
        target_height: The target height to be maintained.
        margin: The margin to be maintained from the target height.
        higher_scale: The scale to be applied to the error when the height is higher than the target.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)

    # height scan: height = sensor_height - hit_point_z
    hit_points = sensor.data.ray_hits_w[..., :3]  # Shape: (N, B, 3)
    sensor_pos = sensor.data.pos_w[:, :3]  # Shape: (N, 3)

    # Compute terrain normal
    normals = compute_terrain_normal(hit_points)  # (N, 3)
    path_cmd_generator.terrain_normals = normals

    # Compute the true base height by projecting the sensor height onto the terrain normal
    sensor_to_hit = hit_points.mean(dim=1) - sensor_pos  # Vector from sensor to terrain
    projected_height = abs(torch.sum(sensor_to_hit * normals, dim=1, keepdim=True))

    # Calculate height error
    height_err = (projected_height.squeeze() - target_height).nan_to_num(nan=0.0, posinf=0.0, neginf=-0.0)
    height_err[height_err > 0.0] *= higher_scale
    return torch.clip(torch.abs(height_err) - margin, min=0.0)


"""
Action penalties.
"""

def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize power for each joint. Common peak values for anymal and wheel should be below 1000 per joint"""
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.sum(power, dim=1)

def applied_torque_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), activation_env_step: int = 0
) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    # Disable reward if not yet activated
    if env.common_step_counter < activation_env_step:
        out_of_limits = torch.zeros_like(out_of_limits)
    return torch.sum(out_of_limits, dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), activation_env_step: int = 0
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    # Disable reward if not yet activated
    if env.common_step_counter < activation_env_step:
        out_of_limits = torch.zeros_like(out_of_limits)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), activation_env_step: int = 0
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)

    # Disable reward if not yet activated
    if env.common_step_counter < activation_env_step:
        out_of_limits = torch.zeros_like(out_of_limits)
    return torch.sum(out_of_limits, dim=1)


"""
Actions.
"""


def action_rate_l2(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2-kernel.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    else:
        if hasattr(env.action_manager.get_term(action_name), "last_raw_actions"):
            action_rate = torch.sum(
                torch.square(
                    env.action_manager.get_term(action_name).raw_actions
                    - env.action_manager.get_term(action_name).last_raw_actions
                ),
                dim=1,
            )
        else:
            action_rate = torch.zeros(env.num_envs, device=env.device)
        # store the last action
        env.action_manager.get_term(action_name).last_raw_actions = env.action_manager.get_term(
            action_name
        ).raw_actions.clone()
        return action_rate


"""
Optional.
"""


def task_difficulty_reward(env: ManagerBasedRLEnv, command_name: str = "path_command") -> torch.Tensor:
    """Reward the agent based on the task difficulty.

    This function rewards the agent based on the task difficulty. The reward is computed based on the
    current configuration of the task. This reward servers as a way to keep the reward growing when the
    curriculum is increasing the task difficulty while maintaining the agent's performance.

    returns: The reward based on the task difficulty.
    """
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    param_bounds = path_cmd_generator.path_cfg_generator.param_bounds  # Bounds for each parameter [(min, max), ...]
    current_params = path_cmd_generator.param_sets  # (num_envs, num_params)

    # Ensure current_params is a tensor
    if not isinstance(current_params, torch.Tensor):
        current_params = torch.tensor(current_params, device=env.device)

    # Normalize parameters based on the provided bounds
    param_min = torch.tensor([b[0] for b in param_bounds], device=current_params.device)  # Shape: (num_params,)
    param_max = torch.tensor([b[1] for b in param_bounds], device=current_params.device)  # Shape: (num_params,)
    # ignore the last parameter which is the position error tolerance
    param_min = param_min[:2]
    param_max = param_max[:2]
    current_params = current_params[:, :2]
    base_difficulty = torch.scalar_tensor(1.0, device=env.device)
    normalized_params = (current_params - param_min) / (param_max - param_min + 1e-8)  # Shape: (num_envs, num_params)

    # Calculate task difficulty as the sum of normalized difficulty across all parameters
    difficulty_score = normalized_params.sum(dim=1) + base_difficulty  # Shape: (num_envs,)

    # Average performance depending on speed and path progress
    performance = path_cmd_generator.path_progress * path_cmd_generator.metrics["robot_speed"]  # Shape: (num_envs,)

    # Compute the reward as a function of task difficulty
    reward = difficulty_score * performance  # Shape: (num_envs,)

    return reward


def goal_position(env: ManagerBasedRLEnv, goal_distance_thresh: float = 0.1, command_name: str = "path_command") -> torch.Tensor:
    """Reward the agent for reaching the goal position."""
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    distance_goal = path_cmd_generator.path_length_command # (num_envs, ), the distance along the path
    goal_pos = path_cmd_generator.path_command_b[:, -1, :2]  # (num_envs, 2)
    pos_error = torch.norm(goal_pos, dim=-1, keepdim=True).reshape(path_cmd_generator.num_envs)  # (num_envs,) the direact distance to the goal
    # Use e^-(0.5x/thresh)^2 to create soft goal distance threshold.
    soft_thresh_gain = torch.exp(-torch.square(distance_goal / (2 * goal_distance_thresh))) # (num_envs, )
    reward = torch.exp(-torch.square(pos_error)) * soft_thresh_gain
    return reward

def goal_orientation(env: ManagerBasedRLEnv, goal_distance_thresh: float = 0.1, command_name: str = "path_command") -> torch.Tensor:
    """Reward the agent for reaching the goal orientation."""
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    distance_goal = path_cmd_generator.path_length_command
    yaw_error = path_cmd_generator.path_command_b[:, -1, 2] # (num_envs, )
    reward = torch.where(
        distance_goal < goal_distance_thresh, torch.exp(-torch.square(yaw_error)), torch.zeros_like(distance_goal)
    )
    return reward


def stand_still_normalization(
    env: ManagerBasedRLEnv,
    goal_distance_thresh: float = 0.1,
    goal_yaw_thresh: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "path_command",
) -> torch.Tensor:
    """Penalize lin/ang velocity + body acceleration when standing still after reaching the goal."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=-1)
    ang_vel_error = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)
    body_acceleration = (
        torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1) * 0.0001
    )
    body_rot_acceleration = (
        torch.sum(torch.norm(asset.data.body_ang_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1) * 0.0001
    )
    error = lin_vel_error + ang_vel_error + body_acceleration + body_rot_acceleration
    # get command
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    distance_goal = path_cmd_generator.path_length_command
    yaw_error = torch.abs(path_cmd_generator.curr_yaw_error)
    # Use e^-(0.5x/thresh)^2 to create soft goal distance threshold.
    soft_thresh_gain = torch.exp(-torch.square(distance_goal / (2 * goal_distance_thresh))) # (num_envs, )
    reward = error * soft_thresh_gain
    return reward


def default_pose(
    env: ManagerBasedRLEnv,
    goal_distance_thresh: float = 0.1,
    goal_yaw_thresh: float = 0.2,
    command_name: str = "path_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for being in the default pose after reaching the goal."""
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    distance_goal = path_cmd_generator.path_length_command
    yaw_error = torch.abs(path_cmd_generator.curr_yaw_error)
    joint_pos_err = (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )  # (num_envs, num_joints)
    # Use e^-(0.5x/thresh)^2 to create soft goal distance threshold.
    soft_thresh_gain = torch.exp(-torch.square(distance_goal / (2 * goal_distance_thresh))) # (num_envs, )
    reward = torch.exp(-torch.sum(torch.square(joint_pos_err), dim=1)) * soft_thresh_gain
    return reward


def feet_air_time_rw(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward


def flat_orientation_l2_thr(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_distance_thresh: float = 0.2,
    command_name: str = "path_command",
) -> torch.Tensor:
    """Penalize non-flat base orientation after reaching the goal using L2-kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.

    Args:
        env: The RL task environment.
        asset_cfg: Configuration of the scene entity.
        goal_distance_thresh: The distance threshold to the goal.

    Returns:
        A tensor representing the penalty for non-flat orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    # Compute the xy-components of the projected gravity vector
    gravity_xy = asset.data.projected_gravity_b[:, :2]
    # Compute the L2 norm of the xy-components
    gravity_xy_l2 = torch.norm(gravity_xy, dim=1)
    # Distance to the goal
    distance_goal = path_cmd_generator.path_length_command  # (num_envs, )
    # Use e^-(0.5x/thresh)^2 to create soft goal distance threshold.
    soft_thresh_gain = torch.exp(-torch.square(distance_goal / (2 * goal_distance_thresh))) # (num_envs, )
    penalty = torch.square(gravity_xy_l2) * soft_thresh_gain
    return penalty


def flat_orientation_l2_slope(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.01,
    command_name: str = "path_command",
) -> torch.Tensor:
    """Penalize non-flat base orientation using L2-kernel, allowing a small threshold for body tilt.

    This is computed by penalizing the alignment between the robot's up direction and the terrain normal.

    Args:
        env: The RL task environment.
        asset_cfg: Configuration of the scene entity.
        threshold: The allowable threshold for body tilt.

    Returns:
        A tensor representing the penalty for non-flat orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)

    # Extract the robot's up direction from its quaternion
    robot_orientations = asset.data.root_quat_w  # (num_envs, 4)
    up_direction = torch.tensor([0.0, 0.0, 1.0], device=robot_orientations.device)
    up_direction_expanded = up_direction.unsqueeze(0).expand(robot_orientations.shape[0], -1)
    robot_up_directions = math_utils.quat_rotate(robot_orientations, up_direction_expanded)

    # Get the terrain normals
    terrain_normals = path_cmd_generator.terrain_normals  # (num_envs, 3)

    # Compute the alignment between the robot's up direction and the terrain normal
    dot_product = torch.sum(robot_up_directions * terrain_normals, dim=1)
    alignment_penalty = 1.0 - torch.abs(dot_product)  # The misalignment (0: perfect alignment, 1: orthogonal)

    # Penalize only if the alignment penalty exceeds the threshold
    penalty = torch.where(alignment_penalty > threshold, alignment_penalty, torch.zeros_like(alignment_penalty))
    return penalty


def lateral_movement(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.25
) -> torch.Tensor:
    """
    Reward the agent for moving lateral using L2-Kernel, lateral movement is valid when the ratio of lateral velocity and
    forward velocity is less than a threshold.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        threshold: The threshold for the ratio of lateral velocity and forward velocity.

    Returns:
        Dense reward [0, +1] based on the lateral velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    lateral_velocity = asset.data.root_vel_b[:, 1]
    forward_velocity = asset.data.root_vel_b[:, 0]
    # make sure forward velocity is not zero
    forward_velocity += 1e-5
    ratio = torch.abs(lateral_velocity / forward_velocity)
    reward = torch.where(ratio > threshold, torch.square(lateral_velocity), torch.zeros_like(ratio))
    reward = torch.clip(reward, min=0, max=1.0)
    return reward


def backwards_movement(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.0,
    command_name: str = "path_command",
) -> torch.Tensor:
    """Reward the agent for moving backwards using L2-Kernel, backward movement is defined
    as the opposite direction of the forward waypoint list.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the backward velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)

    # get the forward direction according to the path waypoints
    ang_diff = (path_cmd_generator.path_command_b[:, 2, 2] - path_cmd_generator.path_command_b[:, 1, 2] + np.pi) % (
        2 * np.pi
    ) - np.pi
    direction_yaw = (path_cmd_generator.path_command_b[:, 1, 2] + ang_diff / 2 + np.pi) % (2 * np.pi) - np.pi
    direction_vector = torch.stack([torch.cos(direction_yaw), torch.sin(direction_yaw)], dim=1)
    forward_direction = direction_vector / (torch.norm(direction_vector, dim=1, keepdim=True) + 1e-5)

    # compute the projection length of the base velocity in the forward direction
    projection_length = torch.sum(asset.data.root_vel_b[:, :2] * forward_direction, dim=1)  # (num_envs, )

    # compute the reward
    backward_movement_idx = torch.where(
        projection_length < threshold, torch.ones_like(projection_length), torch.zeros_like(projection_length)
    )
    reward = torch.square(backward_movement_idx * projection_length)
    reward = torch.clip(reward, min=0, max=1.0)
    return reward


def body_stumble(
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize contact forces on wheels leading to a sudden brake. typical values are negative. bad values are like -1000"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    FtimesV = torch.sum(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids,:] * asset.data.root_lin_vel_w.unsqueeze(1), dim=1) #(n,3)
    return torch.clip(torch.sum(FtimesV, dim=-1), max=0.)