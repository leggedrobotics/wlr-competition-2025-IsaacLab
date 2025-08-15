# path_command.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for path tracking."""

from __future__ import annotations

import numpy as np
import time
import torch
from typing import TYPE_CHECKING

from tabulate import tabulate

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    BLUE_DOT_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    GREEN_STRIP_MARKER_CFG,
    RED_ARROW_X_MARKER_CFG,
    RED_DOT_MARKER_CFG,
)
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from .curriculum_sampling import GridBasedDistribution
from .poly_spline import PolySplinePath
from .pp_controller import PurePursuitController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .path_command_cfg import PathCommandCfg


class PathCommand(CommandTerm):
    """Command generator for generating path commands using poly splines.

    The command generator generates paths by interpolating positions and orientations
    using poly splines. The path commands are generated in the base frame of the robot,
    and not the simulation world frame. This means that users need to handle the transformation
    from the base frame to the simulation world frame themselves.
    """

    cfg: PathCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: PathCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        self._env = env

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]

        # create buffers
        # -- commands: (x, y, yaw) in root frame
        self.path_command_b = torch.zeros(self.num_envs, cfg.num_waypoints, 3, device=self.device)
        self.waypoints_fixed_b = torch.zeros(self.num_envs, cfg.num_waypoints, 3, device=self.device)
        self.current_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.path_command_w = torch.zeros_like(self.path_command_b)
        self.waypoints_fixed_w = torch.zeros_like(self.waypoints_fixed_b)
        self.current_command_w = torch.zeros_like(self.current_command_b)
        self.closest_path_w = torch.zeros_like(self.current_command_b)
        self.goal_yaw = torch.zeros(self.num_envs, device=self.device)

        self.path_length_command = torch.zeros(self.num_envs, device=self.device)
        self.path_progress = torch.zeros(self.num_envs, device=self.device)
        self.last_closest_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.intervals = torch.full(
            (self.num_envs,), self.cfg.std_waypoint_interval * self.cfg.max_speed, device=self.device
        )
        self.fixed_interval = torch.full((self.num_envs,), self.cfg.std_waypoint_interval * 2.0, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # -- metrics
        self.metrics["position_error_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["yaw_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_position_error = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_yaw_error = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_robot_speed = torch.zeros(self.num_envs, device=self.device)
        self.robot_state_flag = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )  # 0: running, 1: goal reached
        self.curr_pos_error = torch.zeros(self.num_envs, device=self.device)
        self.curr_yaw_error = torch.zeros(self.num_envs, device=self.device)
        self.update_count = torch.zeros(self.num_envs, device=self.device)

        # -- evaluation
        self.not_updated_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prev_not_updated_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.maximum_speed = torch.full((self.num_envs,), self.cfg.max_speed, device=self.device)  # (num_envs,)
        self.terrain_normals = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            self.num_envs, 1
        )  # (num_envs, 3)
        self.pos_error_avg_list = []
        self.pos_error_max = torch.zeros(self.num_envs, device=self.device)
        self.robot_speed_avg_list = []
        self.robot_speed_max = torch.zeros(self.num_envs, device=self.device)
        self.dis_to_goal_list = []
        self.robot_pos_list = []
        self.path_pos_list = []

        self.joint_vel = torch.zeros(self.num_envs, 12, device=self.device)
        self.wheel_vel = torch.zeros(self.num_envs, 4, device=self.device)

        self.active_num_list = []
        self.finished_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.finished_num_list = []
        self.failed_envs = torch.zeros(self.num_envs, device=self.device)
        self.is_finished = False
        self.body_poses_list = torch.zeros(self.num_envs, 0, 7, device=self.device)

        # -- path sampling
        self.spline_angle_range = self.cfg.path_config["spline_angle_range"]
        self.rotate_angle_range = self.cfg.path_config["rotate_angle_range"]
        self.pos_tolerance_range = self.cfg.path_config["pos_tolerance_range"]
        self.terrain_level_range = self.cfg.path_config["terrain_level_range"]
        self.path_cfg_resolutions = self.cfg.path_config["resolution"]
        self.path_initial_params = self.cfg.path_config["initial_params"]
        self.path_cfg_generator = GridBasedDistribution(
            param_bounds=[
                self.spline_angle_range,
                self.rotate_angle_range,
                self.pos_tolerance_range,
                self.terrain_level_range,
            ],
            resolutions=self.path_cfg_resolutions,
            initial_curr_params=self.path_initial_params,
            initial_speed=self.cfg.max_speed,
            device=self.device,
        )

        self.pos_err_tolerance = torch.full((self.num_envs,), self.path_initial_params[2], device=self.device)
        self.tol_res_init = self.path_cfg_resolutions[2]
        self.tol_res_dynamic = self.path_cfg_resolutions[2]
        self.path_generator = PolySplinePath(self.path_cfg_resolutions[0], self.path_cfg_resolutions[1])
        self.path_resolution = self.path_generator.ds
        self.skip_interval = 2
        self.skip_count = 0

        # -- Velocity command related
        if self.cfg.convert_to_vel:
            self.pp_controller = PurePursuitController()
            self.metrics["speed_error"] = torch.zeros(self.num_envs, device=self.device)
            self.cumulative_speed_error = torch.zeros(self.num_envs, device=self.device)
            self.desired_speed = torch.zeros(self.num_envs, device=self.device)  # (num_envs,)

        self.last_resample_env_step = 0
        self.last_update_config_env_step = 0
        self.sample_paths()

    def __str__(self) -> str:
        msg = "PathCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.current_command_b.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired path command. Shape is (num_envs, num_waypoint, 3).

        The first three elements correspond to the position, followed by the yaw.
        """
        if self.cfg.convert_to_vel:
            command, self.desired_speed = self.pp_controller.compute_velocity_command(
                self.waypoints_fixed_b, self.intervals, self.cfg.enable_backward
            )
            return command
        return self.path_command_b.reshape(self.num_envs, -1)
        # return 0.1*torch.tensor([[0., 0., 0., 0, 1., 0., 0, 2., 0., 0, 3., 0., 0, 4., 0., 0, 5., 0., 0, 6., 0., 0, 7., 0.,
        #  0, 8., 0., 0, 9., 0.]], device='cuda:0')

    """
    Implementation specific functions.
    """

    def update_path_config(
        self,
        curriculum_params_list: list[list],  # List of parameters for the curriculum
        curriculum_speed_list: list[float],  # True for positive, False for negative
    ):
        """Update the path configuration."""
        print(f"curriculum_params_list: {curriculum_params_list}")
        print(f"curriculum_speed_list: {curriculum_speed_list}")
        self.path_cfg_generator.update_grid(curriculum_params_list, curriculum_speed_list)
        self.sample_paths()

        try:
            self.last_update_config_env_step = self._env.unwrapped.common_step_counter
        except AttributeError:
            self.last_update_config_env_step = 0

    def sample_paths(self, params_list=None, num_list=None, speeds_tensor=None):
        """Sample paths"""
        # Sample new waypoints list from poly spline
        sample_start = time.time()
        if params_list is None or num_list is None or speeds_tensor is None:
            params_list, num_list, speeds_tensor = self.path_cfg_generator.sample_params(self.num_envs)
        sample_param_time = time.time()
        self.skip_count += 1
        # Make a table for the printout
        if self.skip_count > self.skip_interval:
            table_data = [
                [i] + params + [num, speed]
                for i, (params, num, speed) in enumerate(zip(params_list, num_list, speeds_tensor.tolist()))
            ]
            headers = ["Param ID", "SplineAng", "RotateAng", "Tolerance", "Terrain Level", "Num of Path", "Avg Speed"]
            table = tabulate(table_data, headers=headers, tablefmt="pretty")
            print(
                "\n[INFO]: Sampling Poly splines list with following configuration:\n",
                f"Configuration list: {table}",
            )
            self.skip_count = 0

        # Sample paths based on the parameters
        self.paths, self.param_sets = self.path_generator.sample_paths(
            params_list,
            speeds_tensor.tolist(),
            num_list,
            self.cfg.use_rsl_path
        )
        r = torch.empty(self.num_envs, device=self.device)
        self.is_standing_env = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # Sample goal yaw angle 
        self.goal_yaw = torch.rand(self.num_envs, device=self.device) * np.pi - 0.5 * np.pi # [-pi, pi]

        """
        Paths storage:
            self.paths: List of arrays with shape (num, num_points, 4) (x, y, yaw, culmulative_length)
            self.param_sets: List of arrays with shape (num, 3) (angle, rot_angle, pos_tolerance)
        """
        self.num_points_in_paths = self.paths.shape[1]
        # Convert ndarray to tensor
        self.paths = torch.from_numpy(self.paths).to(self.device)  # (num_envs, num_points, 4)
        self.param_sets = torch.from_numpy(self.param_sets).to(self.device)  # (num_envs, 3)
        self.lengths = self.paths[:, -1, 3]  # (num_envs,)
        # Fill the self.maximum_speed based on num_list and speeds_tensor
        expanded_speeds = speeds_tensor.repeat_interleave(torch.tensor(num_list, device=self.device))  # (num_envs,)
        self.maximum_speed[:] = expanded_speeds
        # Fill the tolerance based on the param_sets
        if self.tol_res_init == self.tol_res_dynamic:
            self.pos_err_tolerance[:] = 0.5 - self.param_sets[:, 2]
        else:
            devide_num = self.tol_res_init // self.tol_res_dynamic
            offset = torch.randint(
                -int(devide_num // 2), int(devide_num // 2) + 1, (len(self.param_sets[:, 2]),), device=self.device
            )
            self.pos_err_tolerance[:] = 0.5 - self.param_sets[:, 2] + offset * self.tol_res_dynamic
            self.pos_err_tolerance = torch.clamp(self.pos_err_tolerance, 0.1, 0.5)

        # Check if the buffers already exist
        if hasattr(self, "full_path"):
            # Extend the full_path buffer
            if self.full_path.shape[1] < self.num_points_in_paths:
                last_value_path = self.full_path[:, -1, :].unsqueeze(1)  # Last element along the path dimension
                num_points_to_add = self.num_points_in_paths - self.full_path.shape[1]
                new_path_extension = last_value_path.repeat(1, num_points_to_add, 1)
                self.full_path = torch.cat((self.full_path, new_path_extension), dim=1)
        else:
            # Initialize the buffers if they don't exist
            self.full_path = torch.zeros(self.num_envs, self.num_points_in_paths, 4, device=self.device)

        sample_finish = time.time()
        # Reset last resample step (not available at first call)
        try:
            self.last_resample_env_step = self._env.unwrapped.common_step_counter
        except AttributeError:
            self.last_resample_env_step = 0
        if self.cfg.path_config["terrain_level_range"][1] > self.cfg.path_config["terrain_level_range"][0]:
            # Reset the terrain orgin for each env based on params
            try:
                terrain: TerrainImporter = self._env.scene.terrain
                terrain.set_env_terrain_levels(torch.arange(self.num_envs, device=self.device), self.param_sets[:, 3])
            except AttributeError:
                print("[INFO]: Terrain not available in the scene.\n\n\n")
        else:
            print("[INFO]: Terrain level range is 0, not setting terrain levels.\n\n\n")

        ## Record the time taken for the sampling
        print(
            f"[INFO]: Sampling hast finished.\n",
            f"Time taken for params cfg generation: {sample_param_time - sample_start:.2f} s\n",
            f"Time taken for path generation: {sample_finish - sample_param_time:.2f} s\n\n",
        )

    def _update_metrics(self):
        """Update the metrics for the command generator."""
        if self._env.unwrapped.common_step_counter <= 1:
            return  # The frist step alway has incorrect data

        robot_velocity = self.robot.data.root_lin_vel_b[:, :2]  # (num_envs, 2)
        robot_speed = torch.norm(robot_velocity, dim=1, keepdim=True)  # (num_envs, 1)

        # Update robot state flag
        self.robot_state_flag = torch.where(
            self.path_length_command <= 0.10,
            torch.ones_like(self.robot_state_flag),
            torch.zeros_like(self.robot_state_flag),
        )

        if self.cfg.testing_mode:
            # Create a mask for the active and finished environments
            self.failed_envs = self.failed_envs + self._env.termination_manager.terminated.float()  # (num_envs,)
            active_envs_mask = self.failed_envs == 0
            active_num = active_envs_mask.sum().item()

            # Power from Joint Torques (Motor Power)
            # joint_power = torch.sum(self.joint_torque * self.joint_vel, dim=1)  # (num_envs,)
            # wheel_power = torch.sum(self.wheel_torque * self.wheel_vel, dim=1)  # (num_envs,)
            # motor_power = joint_power + wheel_power # (num_envs,)

            # # Total energy consumption
            # self.total_energy = motor_power  # (num_envs,)

            # Store the evaluation information excluding the terminated environments
            if (active_envs_mask & ~self.finished_mask).any():  # Ensure there are active environments to calculate

                curr_pos_error = self.curr_pos_error[active_envs_mask & ~self.finished_mask]
                curr_pos_error_avg = curr_pos_error.mean().item()
                curr_robot_speed = robot_speed.reshape(self.num_envs)[active_envs_mask & ~self.finished_mask]
                curr_robot_speed_avg = curr_robot_speed.mean().item()
                curr_dis_to_goal = self.path_length_command[active_envs_mask & ~self.finished_mask].mean().item()

                # self.joint_torque_avg = self.joint_torque[active_envs_mask].mean().item()
                # self.wheel_torque_avg = self.wheel_torque[active_envs_mask].mean().item()
                # self.energy_avg = self.total_energy[active_envs_mask].mean().item()

                # Append the current averages to the corresponding lists
                self.pos_error_avg_list.append(curr_pos_error_avg)
                self.pos_error_max[active_envs_mask & ~self.finished_mask] = torch.maximum(
                    self.pos_error_max[active_envs_mask & ~self.finished_mask], curr_pos_error
                )
                self.robot_speed_avg_list.append(curr_robot_speed_avg)
                self.robot_speed_max[active_envs_mask & ~self.finished_mask] = torch.maximum(
                    self.robot_speed_max[active_envs_mask & ~self.finished_mask], curr_robot_speed
                )
                self.dis_to_goal_list.append(curr_dis_to_goal)
                self.robot_pos_list.append(self.robot.data.root_pos_w[0, :2].tolist())
                self.path_pos_list.append(self.closest_path_w[0, :2].tolist())
                self.active_num_list.append(active_num)

                self.finished_mask |= self.path_length_command < 0.10
                self.finished_num = self.finished_mask.sum().item()
                self.finished_num_list.append(self.finished_num)
                # self.joint_torque_avg_list.append(self.joint_torque_avg)
                # self.wheel_torque_avg_list.append(self.wheel_torque_avg)
                # self.energy_avg_list.append(self.energy_avg)

                # print(f"pos_tolerance: {self.pos_err_tolerance.mean().item()}")
                # print(f"pos_error: {curr_pos_error_avg:.2f},,robot_speed: {curr_robot_speed_avg:.2f},dis_to_goal: {curr_dis_to_goal:.2f},active_num: {active_num}")

            self.is_finished = self.finished_num >= active_num

        # Store the cumulative data for curriculum learning
        self.cumulative_position_error += self.curr_pos_error  # (num_envs,)
        self.cumulative_yaw_error += self.curr_yaw_error  # (num_envs,)
        # only update the robot speed when the robot is running
        running_mask = self.robot_state_flag == 0
        self.cumulative_robot_speed[running_mask] += robot_speed.reshape(self.num_envs)[running_mask]  # (num_envs,)
        self.update_count += 1

        # Update the metrics (average over the number of updates)
        self.metrics["position_error_2d"] = self.cumulative_position_error / self.update_count
        self.metrics["yaw_error"] = self.cumulative_yaw_error / self.update_count
        self.metrics["robot_speed"][running_mask] = (
            self.cumulative_robot_speed[running_mask] / self.update_count[running_mask]
        )
        if self.cfg.convert_to_vel:
            self.desired_speed = torch.where(
                self.path_length_command < 0.05, torch.zeros_like(self.desired_speed), self.desired_speed
            )
            desired_speed = self.desired_speed.unsqueeze(1)  # (num_envs, 1)
            sgn_speed_error = (desired_speed - robot_speed).reshape(self.num_envs)
            self.cumulative_speed_error += sgn_speed_error  # (num_envs,)
            self.metrics["speed_error"] = self.cumulative_speed_error / self.update_count

    def _find_last_unique_indices(self, path: torch.Tensor) -> torch.Tensor:
        """
        Find the last unique index for each path in path. if not repeating, return the last index.

        Args:
            path: Tensor of shape (num_envs, total_num, 3)

        Returns:
            last_unique_indices: Tensor of shape (num_envs,)
        """
        num_envs, total_num, _ = path.shape

        # Find differences between adjacent points
        diff = torch.ne(path[:, 1:], path[:, :-1]).any(dim=-1)

        # Append True to the end to consider the last point unique
        diff = torch.cat([diff, torch.ones((num_envs, 1), dtype=torch.bool, device=path.device)], dim=1)

        # Find the index of the last True value for each path
        last_unique_indices = diff.int().sum(dim=1) - 1

        return last_unique_indices

    def _resample_command(self, env_ids: list[int]):
        """sample new path targets
        Args:
            env_ids (list[int]): The list of environment IDs to resample.
        """

        # save current state of not updated environments (necessary to log correct information for evaluation)
        self.prev_not_updated_envs = self.not_updated_envs.clone()
        self.update_count[env_ids] = 0.0
        self.cumulative_position_error[env_ids] = 0.0
        self.cumulative_yaw_error[env_ids] = 0.0
        self.cumulative_robot_speed[env_ids] = 0.0
        self.robot_state_flag[env_ids] = 0
        self.curr_pos_error[env_ids] = 0.0
        if self.cfg.convert_to_vel:
            self.cumulative_speed_error[env_ids] = 0.0

        # Sample new goal yaw angle
        self.goal_yaw[env_ids] = torch.rand(len(env_ids), device=self.device) * np.pi - 0.5 * np.pi  # [-pi, pi]
        # Sample new paths for terminated environments
        sample_idx = torch.randperm(self.paths.shape[0])[: len(env_ids)]
        sample = self.paths[sample_idx].clone()  # (num_envs, num_points, 4)
        length_sample = self.lengths[sample_idx].view(-1)  # (num_envs,)

        robot_pos = self.robot.data.root_pos_w[env_ids, :]  # (num_envs, 3)
        rotate_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w[env_ids, :])[2]  # (num_envs,)

        # if self.cfg.enable_backward:
        # randomly add 180 degrees (make backward path)
        rotate_yaw += torch.randint(0, 2, (len(env_ids),), device=self.device) * np.pi
        cos_yaw = torch.cos(rotate_yaw).unsqueeze(1)
        sin_yaw = torch.sin(rotate_yaw).unsqueeze(1)
        rotation_matrix = torch.stack([cos_yaw, sin_yaw, -sin_yaw, cos_yaw], dim=-1).view(-1, 2, 2)  # (num_envs, 2, 2)

        # Update path positions using rotation matrix
        sample[:, :, :2] = torch.bmm(sample[:, :, :2], rotation_matrix) + robot_pos[:, :2].unsqueeze(
            1
        )  # (num_envs, num_points, 2)
        sample[:, :, 2] = (sample[:, :, 2] + rotate_yaw.unsqueeze(1) + np.pi) % (
            2 * np.pi
        ) - np.pi  # (num_envs, num_points)

        if self.cfg.convert_to_vel:
            self.desired_speed[env_ids] = self.maximum_speed[env_ids]
            # Quantize the desired speed to the nearest multiple of the minimum increment
            min_speed_increment = self.path_resolution / self.cfg.std_waypoint_interval  # 0.0667 m/s
            self.desired_speed[env_ids] = (
                torch.round(self.desired_speed[env_ids] / min_speed_increment) * min_speed_increment
            )
            intervals_raw = self.desired_speed[env_ids] * self.cfg.std_waypoint_interval
        else:
            # Use default interval when resampling
            intervals_raw = self.cfg.std_waypoint_interval * 2.0

        indices, self.intervals[env_ids] = self._initial_waypoints_indices(
            sample[:, :, 3], self.cfg.num_waypoints, intervals_raw
        )

        # Update command buffers in world frame
        selected_path = sample[torch.arange(len(env_ids)).unsqueeze(1), indices, :3]
        self.path_command_w[env_ids, : indices.shape[1], :] = selected_path

        # If the number of waypoints is less than required, fill the remaining with the last waypoint
        if indices.shape[1] < self.cfg.num_waypoints:
            last_waypoint = sample[:, -1, :].unsqueeze(1).repeat(1, self.cfg.num_waypoints - indices.shape[1], 1)
            self.path_command_w[env_ids, indices.shape[1] :, :] = last_waypoint

        indices_fixed, _ = self._initial_waypoints_indices(
            sample[:, :, 3], self.cfg.num_waypoints, self.fixed_interval[env_ids]
        )
        selected_path_fixed = sample[torch.arange(len(env_ids)).unsqueeze(1), indices_fixed, :3]
        self.waypoints_fixed_w[env_ids, : indices_fixed.shape[1], :] = selected_path_fixed

        # If the number of waypoints is less than required, fill the remaining with the last waypoint
        if indices_fixed.shape[1] < self.cfg.num_waypoints:
            last_waypoint = sample[:, -1, :].unsqueeze(1).repeat(1, self.cfg.num_waypoints - indices_fixed.shape[1], 1)
            self.waypoints_fixed_w[env_ids, indices_fixed.shape[1] :, :] = last_waypoint

        self.path_length_command[env_ids] = length_sample
        # Prepare the buffer for updating path
        self.full_path[env_ids, : self.num_points_in_paths] = sample

        # Check if the buffer size is larger than the desired number of points
        if self.full_path.shape[1] > self.num_points_in_paths:
            # Pad the path with the last point
            self.full_path[env_ids, self.num_points_in_paths :] = (
                sample[:, -1, :].unsqueeze(1).repeat(1, self.full_path.shape[1] - self.num_points_in_paths, 1)
            )
        self.last_closest_idx[env_ids] = 0

    def _initial_waypoints_indices(self, length, num_waypoints, interval):
        """
        Efficiently generate waypoint indices using vectorized operations.
        Args:
            length (torch.Tensor): Tensor of shape (num_envs, num_points) representing the cumulative distances for each environment.
            num_waypoints (int): Number of waypoints to be generated.
            interval (torch.Tensor): Desired interval between consecutive waypoints of shape (num_envs,).

        Returns:
            indices (torch.Tensor): Indices tensor of shape (num_envs, num_waypoints).
        """
        # adjust interval if it is larger than the total length
        lengths_exceed = length[:, -1] >= self.cfg.num_waypoints * interval
        adjusted_interval = torch.where(lengths_exceed, interval, length[:, -1] / self.cfg.num_waypoints)

        # Generate the desired distances for each environment: Shape (num_envs, num_waypoints)
        # Broadcasting interval (num_envs,) to desired_distances (num_envs, num_waypoints)
        desired_distances = torch.arange(0, num_waypoints, device=self.device).view(1, -1) * adjusted_interval.view(
            -1, 1
        )  # Shape: (num_envs, num_waypoints)

        # Perform a batched searchsorted across all environments
        search_indices = torch.searchsorted(
            length.contiguous(), desired_distances, right=True
        )  # Shape: (num_envs, num_waypoints)

        # Compute the previous indices for those that have valid `search_indices`
        previous_indices = torch.clamp(search_indices - 1, min=0)

        # Calculate distances for the left and right neighbors
        left_diff = desired_distances - torch.gather(length, 1, previous_indices)
        right_diff = torch.gather(length, 1, search_indices) - desired_distances

        # Determine the closest indices
        closest_indices = torch.where(left_diff <= right_diff, previous_indices, search_indices)

        return closest_indices, adjusted_interval

    def _update_waypoints_indices(self, lengths, start_indices, last_unique_indices, fixed_interval=False):
        """
        Update the waypoint indices based on the current position of the robot.
        Args:
            lengths (torch.Tensor): Tensor of shape (num_envs, num_points) representing the cumulative distances for each environment.
            start_indices (torch.Tensor): Tensor of shape (num_envs,) representing the starting index for each environment.
            last_unique_indices (torch.Tensor): Tensor of shape (num_envs,) representing the last unique index for each environment.
            fixed_interval (bool): Whether to use a fixed interval for updating waypoints.

        Returns:
            indices (torch.Tensor): Tensor of shape (num_envs, num_waypoints) representing the updated waypoint indices.
        """
        num_envs, num_points = lengths.shape

        # Calculate remaining length from `start_indices` to `last_unique_indices` for each environment
        remaining_lengths = (
            lengths[torch.arange(num_envs), last_unique_indices] - lengths[torch.arange(num_envs), start_indices]
        )

        if not fixed_interval:
            # Calculate the flexible interval based on remaining lengths
            self.intervals = torch.min(3.0 * remaining_lengths / (self.cfg.num_waypoints - 1), self.intervals)
            intervals = self.intervals
        else:
            intervals = torch.min(3.0 * remaining_lengths / (self.cfg.num_waypoints - 1), self.fixed_interval)

        # Calculate target distances starting from the cumulative length at `start_indices`
        # Here, target_distances should be relative to the robot's current position along the path
        waypoint_idx = torch.arange(-1, self.cfg.num_waypoints - 1, device=lengths.device)
        target_distances = (intervals.unsqueeze(1) * waypoint_idx) + lengths[
            torch.arange(num_envs), start_indices
        ].unsqueeze(
            1
        )  # Shape: (num_envs, num_waypoints)

        # Use batched searchsorted to find indices of target distances in cumulative lengths
        search_indices = torch.searchsorted(lengths.contiguous(), target_distances)  # Shape: (num_envs, num_waypoints)

        # Clamp search_indices to the range between start_indices and last_unique_indices
        search_indices = torch.max(
            search_indices, start_indices.unsqueeze(1)
        )  # Ensure no indices before `start_indices`
        search_indices = torch.min(
            search_indices, last_unique_indices.unsqueeze(1)
        )  # Ensure no indices after `last_unique_indices`

        return search_indices

    def _update_command(self):
        """Update the command based on the robot's current position."""

        # Get the current path stored in the command buffer
        path = self.full_path.clone()  # shape: (num_envs, total_num, 4)

        # Get the current position of the robot in the world frame
        root_pos = self.robot.data.root_pos_w[:, :2]  # shape: (num_envs, 2)
        root_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)[2]  # shape: (num_envs,)

        # Adjust interval length based on the current speed
        if not self.cfg.convert_to_vel:
            robot_speed = torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)  # shape: (num_envs,)
            self.intervals = torch.where(
                robot_speed < 1.0, self.cfg.std_waypoint_interval, robot_speed * self.cfg.std_waypoint_interval
            )

        # Calculate the distance between the robot and each point in the path
        distances = torch.norm(path[:, :, :2] - root_pos.unsqueeze(1), dim=2)  # shape: (num_envs, total_num)

        # Define the range around last_closest_idx to search for the new closest_idx
        search_range = 120  # This value should be chosen based on the expected robot movement per timestep

        # Define the search bounds
        lower_bound = torch.clamp(self.last_closest_idx, min=0)
        upper_bound = torch.clamp(self.last_closest_idx + search_range, max=path.size(1) - 1)

        # Create a mask to select only the distances within the search range
        mask = torch.arange(path.size(1), device=distances.device).unsqueeze(0).repeat(distances.size(0), 1)
        mask = (mask >= lower_bound.unsqueeze(1)) & (mask <= upper_bound.unsqueeze(1))

        # Set distances outside the search range to a large value (e.g., infinity)
        distances_masked = distances.clone()
        distances_masked[~mask] = float("inf")

        # Find the closest point in the path within the restricted range
        closest_idx = torch.argmin(distances_masked, dim=1)  # shape: (num_envs,)

        # Ensure the new closest index is not before the last closest index
        closest_idx = torch.max(closest_idx, self.last_closest_idx)
        self.last_closest_idx = closest_idx.clone()

        # Find the last unique index for each path in path.
        last_unique_indices = self._find_last_unique_indices(path)  # shape: (num_envs,)

        env_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1)  # shape: (num_envs, 1)
        
        # Set the goal orientation
        path[env_indices, last_unique_indices, 2] = self.full_path[env_indices, last_unique_indices, 2] + self.goal_yaw.unsqueeze(1)

        # Update the waypoint indices
        indices = self._update_waypoints_indices(path[:, :, 3], closest_idx, last_unique_indices)
        indices_fixed = self._update_waypoints_indices(
            path[:, :, 3], closest_idx, last_unique_indices, fixed_interval=True
        )

        # Standing environments
        indices[self.is_standing_env] = 0.0
        indices_fixed[self.is_standing_env] = 0.0
        self.intervals[self.is_standing_env] = 0.0

        # Update command buffers in world frame
        self.path_command_w[:, : indices.shape[1], :] = path[env_indices, indices, :3]
        self.waypoints_fixed_w[:, : indices_fixed.shape[1], :] = path[env_indices, indices_fixed, :3]

        # Update the path length to be the remaining length
        self.path_length_command = (path[env_indices, -1, 3] - path[env_indices, closest_idx.unsqueeze(1), 3]).reshape(
            self.num_envs
        )

        # Update the path progress
        self.path_progress = (path[env_indices, closest_idx.unsqueeze(1), 3] / path[env_indices, -1, 3]).reshape(
            self.num_envs
        )

        # Standing environments
        self.path_length_command[self.is_standing_env] = 0.0
        self.path_progress[self.is_standing_env] = 1.0

        # Transform the waypoint list to the robot's frame
        # Get the waypoints in the world frame
        waypoints_w = self.path_command_w[:, :, :2].clone()  # shape: (num_envs, num_waypoints, 2)
        yaws_w = self.path_command_w[:, :, 2].clone()  # shape: (num_envs, num_waypoints)
        waypoints_fixed_w = self.waypoints_fixed_w[:, :, :2].clone()  # shape: (num_envs, num_waypoints, 2)
        yaws_fixed_w = self.waypoints_fixed_w[:, :, 2].clone()  # shape: (num_envs, num_waypoints)

        # Calculate the relative positions in the robot's frame
        relative_positions = waypoints_w - root_pos.unsqueeze(1)  # shape: (num_envs, num_waypoints, 2)
        relative_positions_fixed = waypoints_fixed_w - root_pos.unsqueeze(1)  # shape: (num_envs, num_waypoints, 2)
        cos_yaw = torch.cos(-root_yaw).unsqueeze(1).unsqueeze(1)  # shape: (num_envs, 1, 1)
        sin_yaw = torch.sin(-root_yaw).unsqueeze(1).unsqueeze(1)  # shape: (num_envs, 1, 1)
        rotation_matrix = torch.cat(
            [torch.cat([cos_yaw, sin_yaw], dim=2), torch.cat([-sin_yaw, cos_yaw], dim=2)], dim=1
        )  # shape: (num_envs, 2, 2)

        relative_positions = torch.bmm(relative_positions, rotation_matrix)  # shape: (num_envs, num_waypoints, 2)
        relative_positions_fixed = torch.bmm(
            relative_positions_fixed, rotation_matrix
        )  # shape: (num_envs, num_waypoints, 2)

        # Calculate the relative yaw angles in the robot's frame
        relative_yaws = yaws_w - root_yaw.unsqueeze(1)  # shape: (num_envs, num_waypoints)
        relative_yaws = (relative_yaws + torch.pi) % (2 * torch.pi) - torch.pi
        relative_yaws_fixed = yaws_fixed_w - root_yaw.unsqueeze(1)  # shape: (num_envs, num_waypoints)
        relative_yaws_fixed = (relative_yaws_fixed + torch.pi) % (2 * torch.pi) - torch.pi

        self.path_command_b[:, :, :2] = relative_positions
        self.path_command_b[:, :, 2] = relative_yaws
        self.waypoints_fixed_b[:, :, :2] = relative_positions_fixed
        self.waypoints_fixed_b[:, :, 2] = relative_yaws_fixed

        # Update the current command to be the point in front of robots
        self.current_command_w = self.path_command_w[:, 2]
        self.closest_path_w = self.path_command_w[:, 1]
        self.current_command_b[:, :] = self.path_command_b[:, 2, :]

        # Compute the error between the closest point in the path and the robot's current position
        pos_error = self.closest_path_w[:, :2] - self.robot.data.root_pos_w[:, :2]
        yaw_error = self.closest_path_w[:, 2] - root_yaw

        # Ensure the yaw_error is in the range [-pi, pi]
        yaw_error = (yaw_error + torch.pi) % (2 * torch.pi) - torch.pi

        # Current errors for reward calculation
        self.curr_pos_error = torch.norm(pos_error, dim=-1, keepdim=True).reshape(self.num_envs)  # (num_envs,)
        self.curr_yaw_error = yaw_error.reshape(self.num_envs)  # (num_envs,)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "waypoint_visualizer"):
                # -- waypoint visualizer
                if not self.cfg.testing_mode:
                    waypoint_marker_cfg = FRAME_MARKER_CFG.copy()
                    waypoint_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                    # waypoint_marker_cfg = BLUE_DOT_MARKER_CFG.copy()
                    # waypoint_marker_cfg.markers["dot"].radius = 0.1
                else:
                    # waypoint_marker_cfg = BLUE_DOT_MARKER_CFG.copy()
                    waypoint_marker_cfg = FRAME_MARKER_CFG.copy()
                    waypoint_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

                waypoint_marker_cfg.prim_path = "/Visuals/Command/waypoints"
                self.waypoint_visualizer = VisualizationMarkers(waypoint_marker_cfg)

                # -- tolerance visualizer
                tolerance_marker_cfg = GREEN_STRIP_MARKER_CFG.copy()
                tolerance_marker_cfg.markers["strip"].radius = 0.5 - self.cfg.path_config["initial_params"][2]
                tolerance_marker_cfg.prim_path = "/Visuals/Command/tolerance"
                self.tolerance_visualizer = VisualizationMarkers(tolerance_marker_cfg)

                # -- body traj visualizer
                body_marker_cfg = RED_DOT_MARKER_CFG.copy()
                body_marker_cfg.prim_path = "/Visuals/Command/body_traj"
                self.body_traj_visualizer = VisualizationMarkers(body_marker_cfg)

                # -- desired speed
                des_speed_marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                des_speed_marker_cfg.prim_path = "/Visuals/Command/velocity_desired"
                des_speed_marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.des_speed_visualizer = VisualizationMarkers(des_speed_marker_cfg)

                # -- current speed
                body_speed_marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                body_speed_marker_cfg.prim_path = "/Visuals/Command/velocity_current"
                body_speed_marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.body_speed_visualizer = VisualizationMarkers(body_speed_marker_cfg)

                # set their visibility to true
                self.waypoint_visualizer.set_visibility(True)
                self.tolerance_visualizer.set_visibility(False)
                self.body_traj_visualizer.set_visibility(False)
                self.des_speed_visualizer.set_visibility(False)
                self.body_speed_visualizer.set_visibility(True)

            if self.cfg.convert_to_vel:
                self.des_speed_visualizer.set_visibility(True)
            if self.cfg.testing_mode and False:
                self.tolerance_visualizer.set_visibility(True)
                self.body_traj_visualizer.set_visibility(True)

        else:
            if hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer.set_visibility(False)
                self.tolerance_visualizer.set_visibility(False)
                self.body_traj_visualizer.set_visibility(False)
                self.des_speed_visualizer.set_visibility(False)
                self.body_speed_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        # -- waypoints visualizer
        if self.cfg.testing_mode and False:
            waypoint_positions = self.full_path[:, ::8, :2].reshape(-1, 2)
            waypoint_yaws = self.full_path[:, ::8, 2].reshape(-1)
        else:
            if self.cfg.convert_to_vel:
                waypoint_positions = self.waypoints_fixed_w[:, :, :2].reshape(-1, 2)
                waypoint_yaws = self.waypoints_fixed_w[:, :, 2].reshape(-1)
            else:
                waypoint_positions = self.path_command_w[:, :, :2].reshape(-1, 2)
                waypoint_yaws = self.path_command_w[:, :, 2].reshape(-1)
        waypoint_poses = torch.cat((waypoint_positions, torch.zeros_like(waypoint_positions[:, :1])), dim=1)
        waypoint_orientations = quat_from_euler_xyz(
            torch.zeros_like(waypoint_yaws), torch.zeros_like(waypoint_yaws), waypoint_yaws
        )

        waypoint_positions_2 = self.waypoints_fixed_w[:, :, :2].reshape(-1, 2)
        waypoint_yaws_2 = self.waypoints_fixed_w[:, :, 2].reshape(-1)
        waypoint_poses_2 = torch.cat((waypoint_positions_2, 0.05 * torch.ones_like(waypoint_positions_2[:, :1])), dim=1)
        waypoint_orientations_2 = quat_from_euler_xyz(
            torch.zeros_like(waypoint_yaws_2), torch.zeros_like(waypoint_yaws_2), waypoint_yaws_2
        )

        self.waypoint_visualizer.visualize(waypoint_poses_2, waypoint_orientations_2)

        if self.cfg.testing_mode:
            # -- tolerance visualizer
            self.tolerance_visualizer.visualize(waypoint_poses, waypoint_orientations)

            # -- body traj visualizer
            body_positions = self.robot.data.root_pos_w[:, :2]
            body_height = torch.ones_like(body_positions[:, :1]) * 0.25
            body_yaws = euler_xyz_from_quat(self.robot.data.root_quat_w)[2]
            body_orientations = quat_from_euler_xyz(torch.zeros_like(body_yaws), torch.zeros_like(body_yaws), body_yaws)
            body_poses = torch.cat((body_positions, body_height, body_orientations), dim=1)
            # self.body_poses_list (num_envs, -1, 7), body_poses (num_envs, 7)
            self.body_poses_list = torch.cat((self.body_poses_list, body_poses.unsqueeze(1)), dim=1)
            body_poses_list = self.body_poses_list.reshape(-1, 7)
            self.body_traj_visualizer.visualize(body_poses_list[:, :3], body_poses_list[:, 3:7])

        # -- desired speed visualizer
        if self.cfg.convert_to_vel:
            speed_pos_w = self.robot.data.root_state_w[:, :3].clone()
            speed_pos_w[:, 2] += 0.6
            ang_diff = (self.path_command_b[:, 2, 2] - self.path_command_b[:, 1, 2] + np.pi) % (2 * np.pi) - np.pi
            desired_velocity_yaw = (self.path_command_b[:, 1, 2] + ang_diff / 2 + np.pi) % (2 * np.pi) - np.pi
            des_speed = self.desired_speed
            desired_velocity = torch.stack(
                [des_speed * torch.cos(desired_velocity_yaw), des_speed * torch.sin(desired_velocity_yaw)], dim=1
            )
            vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(desired_velocity)
            self.des_speed_visualizer.visualize(speed_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)

        # -- current speed visualizer
        base_pos_w = self.robot.data.root_state_w[:, :3].clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.body_speed_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.des_speed_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 2.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
