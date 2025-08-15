# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class PurePursuitController:
    def __init__(self, lookahead_distance_k=0.25):
        self.lookahead_distance_k = lookahead_distance_k
        self.max_vx = 6.0
        self.max_vy = 1.0
        self.max_vyaw = 3.0

    def _calculate_desired_speed(self, path, intervals, interval_to_speed=0.15):
        """
        Calculate the maximum speed based on the intervals between waypoints.
        Assuming interval_to_speed --> 1m/s.
        Calculates the desired speed based on the maximum speed and the curvature of the current path.
        """
        maximum_speed = intervals / interval_to_speed
        # Compute approximate curvature for each waypoint
        length = path.shape[1]
        dx = torch.diff(path[:, : int(length), 0], dim=1)
        dy = torch.diff(path[:, : int(length), 1], dim=1)
        ddx = torch.diff(dx, dim=1)
        ddy = torch.diff(dy, dim=1)

        # Adjust for the dimension mismatch due to differentiation
        dx = dx[:, :-1]
        dy = dy[:, :-1]

        curvature = torch.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-8) ** 1.5
        curvature_max = curvature.max(dim=1).values

        # Reduce speed where curvature is high
        scale_factor = 0.2
        offset = torch.clamp(scale_factor * maximum_speed, min=0.2, max=1.0)
        curvature_factor = torch.clamp(1 / (curvature_max + offset), min=0.4, max=1.0)
        desired_speed = maximum_speed * curvature_factor

        return desired_speed  # (num_envs, )

    def find_lookahead_point(self, waypoints, desired_speed):
        """
        Find the lookahead point along the path based on the lookahead distance.
        The waypoints are in the robot's base frame.
        """

        # Calculate the differences between consecutive waypoints
        delta_waypoints = waypoints[:, 1:, :2] - waypoints[:, :-1, :2]  # shape: (num_envs, num_points-1, 2)

        # Compute Euclidean distances between consecutive waypoints
        distances_between_waypoints = torch.norm(delta_waypoints, dim=2)  # shape: (num_envs, num_points-1)

        # Calculate cumulative distances (set initial distance as 0)
        cumulative_distances = torch.cat(
            (
                torch.zeros(
                    distances_between_waypoints.shape[0], 1, device=waypoints.device
                ),  # Initial distance = 0 for all envs
                torch.cumsum(distances_between_waypoints, dim=1),
            ),  # Cumulative sum of distances
            dim=1,
        )  # shape: (num_envs, num_points)

        # Calculate the lookahead distance
        lookahead_distance = self.lookahead_distance_k * desired_speed.unsqueeze(1)  # shape: (num_envs, 1)
        lookahead_distance = torch.clamp(lookahead_distance, min=0.5)

        # Create a mask where the distances are greater than or equal to the lookahead distance
        lookahead_mask = cumulative_distances >= lookahead_distance  # shape: (num_envs, num_points)

        # If all distances are < lookahead_distance, use the last index (waypoints.shape[1] - 1)
        fallback_idx = waypoints.shape[1] - 1  # The index of the last waypoint

        # Find the first index where the lookahead distance condition is met
        valid_idx = torch.where(lookahead_mask, torch.arange(waypoints.shape[1], device=waypoints.device), fallback_idx)

        # Get the first valid index for each environment
        indices = torch.min(valid_idx, dim=1).values

        # Gather the lookahead points
        lookahead_point = torch.gather(
            waypoints, 1, indices.unsqueeze(1).unsqueeze(2).repeat(1, 1, waypoints.shape[2])
        ).squeeze(1)
        return lookahead_point

    def _calculate_velocity_command(self, lookahead_point, desired_speed, enable_backward, epsilon=1e-6, yaw_threshold=1e-2):
        """
        Calculate the velocity command based on the lookahead point.
        """
        # Handle the special case where desired_speed is zero
        if torch.all(desired_speed == 0):
            return torch.zeros(lookahead_point.shape[0], 3, device=lookahead_point.device)

        # Determine if the lookahead point is in front or behind the robot
        forward_movement = lookahead_point[:, 0] >= 0.1  # Boolean mask indicating forward movement
        backward_movement = lookahead_point[:, 0] < 0.1  # Boolean mask indicating backward movement

        direction = lookahead_point[:, :2]
        direction_norm = torch.norm(direction, dim=1, keepdim=True)

        # Avoid division by zero by adding a small epsilon
        direction_norm = torch.clamp(direction_norm, min=epsilon)

        direction_normalized = direction / direction_norm

        # Calculate linear velocities
        x_vel = direction_normalized[:, 0] * desired_speed.squeeze()
        y_vel = direction_normalized[:, 1] * desired_speed.squeeze()

        # Calculate the orientation towards the lookahead point (angle to the lookahead point)
        orientation_towards_lookahead = torch.atan2(direction[:, 1], direction[:, 0])

        # Average the tangent yaw angle (from the lookahead point) and the orientation towards the lookahead point
        yaw_angle_combined = (lookahead_point[:, 2] + orientation_towards_lookahead) / 2.0
        yaw_error = yaw_angle_combined.clone()

        # Adjust yaw error for backward movement
        if enable_backward:
            yaw_error[backward_movement] = torch.where(
                yaw_error[backward_movement] > torch.pi / 2,
                yaw_error[backward_movement] - torch.pi,
                torch.where(
                    yaw_error[backward_movement] < -torch.pi / 2,
                    yaw_error[backward_movement] + torch.pi,
                    yaw_error[backward_movement],
                ),
            )
        else:
            # Approximate the spin turn by reducing the linear velocity
            if torch.abs(yaw_error) > torch.pi / 2:
                x_vel= 0.2 * x_vel
                y_vel= 0.2 * y_vel

        # If the linear velocity is very low, reduce or zero the yaw velocity
        yaw_vel = torch.where(
            direction_norm.squeeze() < yaw_threshold, torch.zeros_like(yaw_error), 2 * yaw_error * desired_speed
        )

        # restrict vel within limits:
        x_vel = torch.clamp(x_vel, -self.max_vx, self.max_vx)
        y_vel = torch.clamp(y_vel, -self.max_vy, self.max_vy)
        yaw_vel = torch.clamp(yaw_vel, -self.max_vyaw, self.max_vyaw)
        return torch.stack([x_vel, y_vel, yaw_vel], dim=1)

    def compute_velocity_command(self, path, intervals, enable_backward):
        """
        Compute the velocity command for the robot based on the provided path.
        :param path: Tensor, shape (num_envs, num_points, 3) - path waypoints
        :param intervals: Tensor, shape (num_envs,) - intervals between waypoints
        :param enable_backward: bool - whether to enable backward movement
        :return: Tensor, shape (num_envs, 3) - velocity command (x_vel, y_vel, yaw_vel)
        """
        desired_speed = self._calculate_desired_speed(path, intervals)
        lookahead_point = self.find_lookahead_point(path, desired_speed)
        # print(f"desired_speed 1: {desired_speed}")
        velocity_command = self._calculate_velocity_command(lookahead_point, desired_speed, enable_backward)
        return velocity_command, desired_speed


# Example usage
if __name__ == "__main__":
    # Assuming you have path tensors
    num_envs = 5
    num_points = 10

    # Random example data for path
    path = torch.rand((num_envs, num_points, 3))  # path waypoints (x, y, yaw)

    controller = PurePursuitController(lookahead_distance_k=0.5)
    velocity_command = controller.compute_velocity_command(path, 0.5)
