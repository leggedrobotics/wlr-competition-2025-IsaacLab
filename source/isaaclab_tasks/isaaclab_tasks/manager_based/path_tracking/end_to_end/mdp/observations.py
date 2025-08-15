# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from isaaclab.envs.mdp.observations import *

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.path_tracking.hierarchical.mdp.actions.vel_track_actions import TrackingSE2Action
    from isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp.commands.path_command import PathCommand

"""
Emtpy states.
"""


def empty_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Empty privileged state."""
    return torch.zeros((env.num_envs, 1), device=env.device)


"""
Privileged states.
"""


def ground_friction(env: ManagerBasedEnv) -> torch.Tensor:
    """Friction coefficient of the ground."""
    # extract the used quantities (to enable type-hinting)
    static_friction = env.scene.terrain.cfg.physics_material.static_friction
    dynamic_friction = env.scene.terrain.cfg.physics_material.dynamic_friction
    return torch.tensor([static_friction, dynamic_friction], device=env.device).repeat(env.num_envs, 1)


def feet_friction(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), mode: Literal["static", "dynamic"] = "static"
) -> torch.Tensor:
    """Friction coefficient of the feet."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    num_shapes_per_body = []
    for link_path in asset.root_physx_view.link_paths[0]:
        link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
        num_shapes_per_body.append(link_physx_view.max_shapes)

    # get the current materials of the bodies
    materials = asset.root_physx_view.get_material_properties()
    friction_coefficients = torch.zeros(
        (env.num_envs, len(asset_cfg.body_ids), 1), device=env.device
    )  # Dynamic & Static friction coefficients

    # sample material properties from the given ranges
    itr = 0
    for body_id in asset_cfg.body_ids:
        # start index of shape
        start_idx = sum(num_shapes_per_body[:body_id])
        # end index of shape
        end_idx = start_idx + num_shapes_per_body[body_id]
        if mode == "static":
            friction_coefficients[:, itr, :] = materials[:, start_idx:end_idx, 0]
        else:
            friction_coefficients[:, itr, :] = materials[:, start_idx:end_idx, 1]
        itr += 1
    return friction_coefficients.reshape(env.num_envs, -1)


def mass_disturbance(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mass disturbance in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if not hasattr(asset.data, "added_mass"):
        return torch.zeros((env.num_envs, len(asset_cfg.body_ids)), device=env.device)
    else:
        added_mass = asset.data.added_mass[:, asset_cfg.body_ids]
    return added_mass.reshape(env.num_envs, -1).to(env.device)


def feet_air_time(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Time for which the feet are in the air."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    airtime = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    return airtime.reshape(env.num_envs, -1)


def feet_contact_state(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact state of the feet."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_state = (contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids] < env.step_dt).float()
    return contact_state.reshape(env.num_envs, -1)


def penalized_contacts(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = (
        (torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold)
        .float()
        .to(env.device)
    )
    # compute the penalty
    return violation.reshape(env.num_envs, -1)


def body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the wrench
    wrench = (
        torch.cat(
            (asset._external_force_b[:, asset_cfg.body_ids], asset._external_torque_b[:, asset_cfg.body_ids]), dim=1
        )
        .reshape(env.num_envs, -1)
        .to(env.device)
    )
    return wrench


def push_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mass disturbance in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if not hasattr(asset.data, "push_velocity"):
        return torch.zeros((env.num_envs, 6), device=env.device)
    else:
        push_velocity = asset.data.push_velocity
    return push_velocity.reshape(env.num_envs, -1).to(env.device)


"""
Commands
"""


def base_command_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The base command error of the asset.
    NOTE: no performance difference in simulation observable when adding this redundant term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    state = torch.cat((asset.data.root_lin_vel_b[:, :2], asset.data.root_ang_vel_b[:, 2].unsqueeze(1)), dim=1)
    return command - state


"""
Root state.
"""


def base_pos_2d(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root 2d position (x, y) in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :2]


"""
Joint state.
"""


def joint_pos_error(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint position erros of the asset."""
    return env.action_manager.action[:, asset_cfg.joint_ids] - joint_pos_rel(env, asset_cfg)


def joint_vel_error(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint velocity erros of the asset."""
    return env.action_manager.action[:, asset_cfg.joint_ids] - joint_vel_rel(env, asset_cfg)


"""
Path tracking.
"""


def distance_to_goal(env: ManagerBasedRLEnv, command_name: str = "path_command") -> torch.Tensor:
    """The distance to the next waypoint along the path, represented as a scalar value."""
    try:
        path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
        return path_cmd_generator.path_length_command.reshape(env.num_envs, 1)
    except AttributeError:
        return torch.zeros((env.num_envs, 1), device=env.device)


def waypoints_list(env: ManagerBasedRLEnv, num_waypoints=10, command_name: str = "path_command") -> torch.Tensor:
    """The waypoints list of the path, represented as a tensor."""
    try:
        path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
        assert (
            num_waypoints == path_cmd_generator.cfg.num_waypoints
        ), "The number of waypoints should be the same as the path command generator."
        return path_cmd_generator.path_command_b.reshape(env.num_envs, -1)
        # return 0.1*torch.tensor([[0., 0., 0., 0, 1., 0., 0, 2., 0., 0, 3., 0., 0, 4., 0., 0, 5., 0., 0, 6., 0., 0, 7., 0.,
        #  0, 8., 0., 0, 9., 0.]], device='cuda:0')
    except AttributeError:
        return torch.zeros((env.num_envs, num_waypoints, 3), device=env.device)


def waypoint_interval(env: ManagerBasedRLEnv, command_name: str = "path_command") -> torch.Tensor:
    """The distance between two waypoints along the path, represented as a scalar value."""
    try:
        path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
        distance_to_goal = path_cmd_generator.path_length_command.unsqueeze(1)
        raw_interval = path_cmd_generator.intervals.reshape(env.num_envs, 1)
        return torch.where(distance_to_goal > 0.1, raw_interval, torch.zeros_like(raw_interval))
        # return 0.1 * torch.ones((env.num_envs, 1), device=env.device)
    except AttributeError:
        return torch.zeros((env.num_envs, 1), device=env.device)


def waypoints_list_fixed(env: ManagerBasedRLEnv, num_waypoints=10, command_name: str = "path_command") -> torch.Tensor:
    """The waypoints list of the path, represented as a tensor."""
    try:
        path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
        return path_cmd_generator.waypoints_fixed_b.reshape(env.num_envs, -1)
        # return 0.1*torch.tensor([[0., 0., 0., 0, 1., 0., 0, 2., 0., 0, 3., 0., 0, 4., 0., 0, 5., 0., 0, 6., 0., 0, 7., 0.,
        #  0, 8., 0., 0, 9., 0.]], device='cuda:0')
    except AttributeError:
        return torch.zeros((env.num_envs, num_waypoints, 3), device=env.device)


def pos_err_tolerance(
    env: ManagerBasedRLEnv,
    command_name: str = "path_command",
) -> torch.Tensor:
    """The position error tolerance of the path, represented as a scalar value."""
    try:
        path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
        return path_cmd_generator.pos_err_tolerance.reshape(env.num_envs, 1)
    except AttributeError:
        return torch.zeros((env.num_envs, 1), device=env.device)


def pos_err_tolerance_one_hot_vec(
    env: ManagerBasedRLEnv, scale_factor, vector_len, command_name: str = "path_command"
) -> torch.Tensor:
    """The position error tolerance of the path, represented as one hot vector.
    scale the scalar value by a factor and convert it as integer, then convert it to one hot vector.
    """
    try:
        path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
        pos_err_tolerance_float = path_cmd_generator.pos_err_tolerance.reshape(env.num_envs, 1)  # (num_envs, 1)
        pos_err_tolerance_index = (
            torch.round(pos_err_tolerance_float * scale_factor).to(torch.int64) - 1
        )  # (num_envs, 1)
        one_hot_encoding = torch.zeros(env.num_envs, vector_len, device=env.device)
        one_hot_encoding.scatter_(1, pos_err_tolerance_index, 1)
        return one_hot_encoding  # (num_envs, len)
    except AttributeError:
        one_hot_encoding = torch.zeros((env.num_envs, vector_len), device=env.device)  # (num_envs, len)
        # Default value is the last element, maximum tolerance
        one_hot_encoding[:, -1] = 1
        return one_hot_encoding


def spline_angle(env: ManagerBasedRLEnv, command_name: str = "path_command") -> torch.Tensor:
    """The spline angle for the current path configuration, represented as a scalar value."""
    try:
        path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
        return path_cmd_generator.param_sets[:, 0].reshape(env.num_envs, 1)
    except AttributeError:
        return torch.zeros((env.num_envs, 1), device=env.device)


def rotation_angle(env: ManagerBasedRLEnv, command_name: str = "path_command") -> torch.Tensor:
    """The rotation angle for the current path configuration, represented as a scalar value."""
    try:
        path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
        return path_cmd_generator.param_sets[:, 1].reshape(env.num_envs, 1)
    except AttributeError:
        return torch.zeros((env.num_envs, 1), device=env.device)


"""
Actions.
"""


def last_last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.prev_action
    else:
        raise NotImplementedError("Action term is not implemented yet.")


"""
Sensors.
"""


def foot_height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return heights.reshape(env.num_envs, -1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)


"""
For Hirearchical RL (path cmd + vel cmd)
"""


def vel_commands(env: ManagerBasedRLEnv, action_term: str) -> torch.Tensor:
    """The velocity command generated by the planner and given as input to the step function"""
    action_term: TrackingSE2Action = env.action_manager._terms[action_term]
    return action_term.processed_actions


def last_low_level_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).low_level_actions


def last_last_low_level_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).prev_low_level_actions
