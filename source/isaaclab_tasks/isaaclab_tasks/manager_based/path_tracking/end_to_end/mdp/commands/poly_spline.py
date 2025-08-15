# poly_spline.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility for generating poly splines."""

import numpy as np
import os
from scipy.interpolate import make_interp_spline

from .RSL_path import generate_rsl_path


class PolySplinePath:
    def __init__(self, spline_res, rotation_res, scale=0.5, delta_factor=2):
        # Initialization of path parameters
        self.scale = scale  # Scaling factor for subsequent shifts
        self.delta_factor = delta_factor  # Factor to determine the delta angle
        self.max_length = 9.0  # Maximum length of 3-sections poly spline, beyond which the path is concatenated
        self.dis = 3.0  # Distance increment for each waypoint segment
        self.ds = 0.01  # Path resolution
        self.rotation_res = rotation_res  # Rotation resolutions for the path
        self.spline_res = spline_res  # Spline resolution for the path
        self.save_dir = os.path.join(os.path.dirname(__file__), "paths")

    def _concatenate_paths(self, first_paths, second_paths):
        """Concatenate two paths by rotating and translating the second path."""
        num_path, _, _ = first_paths.shape
        num_path2, _, _ = second_paths.shape
        assert num_path == num_path2, "Number of paths must be the same"

        # Extract the last point of the first path
        last_points_first_paths = first_paths[:, -1, :2]  # shape: (num_path, 2)
        last_angles_first_paths = first_paths[:, -1, 2]  # shape: (num_path,)

        # Compute the rotation matrix for each path
        cos_angles = np.cos(last_angles_first_paths)
        sin_angles = np.sin(last_angles_first_paths)
        rotation_matrices = np.array([[cos_angles, -sin_angles], [sin_angles, cos_angles]])  # shape: (2, 2, num_path)
        rotation_matrices = np.transpose(rotation_matrices, (2, 0, 1))  # shape: (num_path, 2, 2)

        # Rotate the second path
        second_paths_rotated = np.einsum(
            "nij,nkj->nki", rotation_matrices, second_paths[:, :, :2]
        )  # shape: (num_path, num_points, 2)

        # Translate the second path
        second_paths_rotated_translated = (
            second_paths_rotated + last_points_first_paths[:, np.newaxis, :]
        )  # shape: (num_path, num_points, 2)

        # Compute the new angles for the rotated second path
        new_angles = second_paths[:, :, 2] + last_angles_first_paths[:, np.newaxis]

        # Adjust the cumulative distance for the second path
        cumulative_distance_offset = first_paths[:, -1, 3]  # Take the last cumulative distance from the first path
        new_cumulative_distances = second_paths[:, :, 3] + cumulative_distance_offset[:, np.newaxis]

        # Concatenate the four components: x, y, yaw, and cumulative distance
        second_paths_rotated_translated_with_angles_and_distances = np.dstack(
            (second_paths_rotated_translated, new_angles, new_cumulative_distances)
        )

        # Concatenate the two paths
        concatenated_paths = np.concatenate(
            (first_paths, second_paths_rotated_translated_with_angles_and_distances), axis=1
        )  # shape: (num_path, 2 * num_points, 4)

        return concatenated_paths

    def compute_curvature(self, x, y):
        """Compute the curvature of a path given its x and y coordinates."""
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5
        return curvature

    def generate_paths(self, dis, max_angle, rotate_angle, max_num, spline_order=3):
        """
        Generates paths with varying difficulty levels and initial heading angles. It will load from a local file if possible.

        Args:
            dis (float): Distance increment for each waypoint segment.
            max_angle (float): Maximum angle for the shift in waypoints.
            rotate_angle (float): Maximum rotation angle for the path.
            max_num (int): Maximum number of paths to generate.
            spline_order (int): Order of the spline interpolation (2 for quadratic, 3 for cubic, etc.)

        Returns:
            path_all (np.ndarray): All path segments, shape (num_of_path, -1, 4).
            num_of_path (int): Number of path segments.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if max_angle < self.spline_res:
            return self._generate_straight_line(dis, rotate_angle, max_num)

        # Create a unique filename based on the input configuration
        filename = f"paths_dis{dis}_maxangle{max_angle}_rotate{rotate_angle}_spline{spline_order}.npy"
        file_path = os.path.join(self.save_dir, filename)

        # Check if the file exists
        if os.path.exists(file_path):
            # print(f"Loading paths from {filename}...")
            path_all = np.load(file_path)

            # If max_num is specified and is less than the available paths, sample
            if path_all.shape[0] > max_num:
                indices = np.random.choice(path_all.shape[0], max_num, replace=False)
                path_all = path_all[indices]

            return path_all, path_all.shape[0]

        # If the file does not exist, generate the paths as usual
        delta_angle = self.spline_res / self.delta_factor  # Delta angle for shifts in waypoints
        scale_factor = self.scale
        shifts1_low = np.arange(-max_angle, -max_angle + self.spline_res + delta_angle, delta_angle)
        shifts1_high = np.arange(max_angle - self.spline_res, max_angle + delta_angle, delta_angle)
        shifts1 = np.concatenate((shifts1_low, shifts1_high))

        path_all = []

        shifts_combinations = []
        for shift1 in shifts1:
            shifts2_low = np.arange(
                -max_angle * scale_factor + shift1,
                (-max_angle + self.spline_res) * scale_factor + shift1 + delta_angle * scale_factor,
                delta_angle * scale_factor,
            )
            shifts2_high = np.arange(
                (max_angle - self.spline_res) * scale_factor + shift1,
                max_angle * scale_factor + shift1 + delta_angle * scale_factor,
                delta_angle * scale_factor,
            )
            shifts2 = np.concatenate((shifts2_low, shifts2_high))

            for shift2 in shifts2:
                shifts3_low = np.arange(
                    -max_angle * scale_factor**2 + shift2,
                    (-max_angle + self.spline_res) * scale_factor**2 + shift2 + delta_angle * scale_factor**2,
                    delta_angle * scale_factor**2,
                )
                shifts3_high = np.arange(
                    (max_angle - self.spline_res) * scale_factor**2 + shift2,
                    max_angle * scale_factor**2 + shift2 + delta_angle * scale_factor**2,
                    delta_angle * scale_factor**2,
                )
                shifts3 = np.concatenate((shifts3_low, shifts3_high))

                for shift3 in shifts3:
                    shifts_combinations.append([shift1, shift2, shift3])

        # print(f"Number of shifts combinations: {len(shifts_combinations)}")
        shifts_combinations = np.array(shifts_combinations)  # shape: (num_combinations, 3)
        waypoints = self._generate_waypoints(dis, shifts_combinations)  # shape: (num_combinations, 5, 2)
        path_all = self._generate_spline_line(
            waypoints, spline_order, rotate_angle
        )  # shape: (num_combinations, num_points, 4)

        # Save the generated paths to a file (only when all the combinations are generated)
        print(f"Saving generated paths: {file_path}...")
        np.save(file_path, path_all)

        if len(shifts_combinations) > max_num:
            np.random.shuffle(path_all)
            path_all = path_all[:max_num]

        return path_all, path_all.shape[0]

    def _generate_straight_line(self, dis, rotate_angle, num):
        paths = []
        waypts = np.array([[0, 0], [dis, 0], [2 * dis, 0], [3 * dis, 0]])
        path_r = np.arange(0, waypts[-1, 0] + self.ds, self.ds)
        path_x = path_r
        path_y = np.zeros_like(path_r)
        path_yaw = np.zeros_like(path_r)

        # Calculate the cumulative distance
        cumulative_dist = path_r.copy()  # For a straight line, cumulative distance equals path_r

        # Construct the path with distance included
        path = np.vstack((path_x, path_y, path_yaw, cumulative_dist)).T
        paths = self._apply_initial_heading(num, path, rotate_angle)

        return paths, num  # shape: (num, num_points, 4)

    def _generate_waypoints(self, dis, shifts_combinations):
        num_paths = shifts_combinations.shape[0]
        waypoints = np.zeros((num_paths, 5, 2))
        waypoints[:, 1, 0] = dis
        waypoints[:, 2, 0] = 2 * dis
        waypoints[:, 3, 0] = 3 * dis - 0.001
        waypoints[:, 4, 0] = 3 * dis
        waypoints[:, 1, 1] = shifts_combinations[:, 0]
        waypoints[:, 2, 1] = shifts_combinations[:, 1]
        waypoints[:, 3, 1] = shifts_combinations[:, 2]
        waypoints[:, 4, 1] = shifts_combinations[:, 2]
        return waypoints

    def _generate_spline_line(self, waypoints, spline_order, rotate_angle):
        """
        Generate spline paths based on input waypoints and apply initial heading rotation.

        Args:
        waypoints (np.ndarray): Input waypoints of shape (num_paths, num_waypoints, 2)
            spline_order (int): Order of the spline interpolation
            rotate_angle (float): Angle range for applying initial heading rotation

        Returns:
            np.ndarray: Rotated paths of shape (num_paths, num_points, 4)
        """
        # Generate uniform samples along the first path's x-coordinates
        path_r = np.arange(0, waypoints[0, -1, 0] + self.ds, self.ds)

        num_paths = waypoints.shape[0]
        num_points = len(path_r)
        paths = np.zeros((num_paths, num_points, 4))  # Initialize paths with 4 columns: x, y, yaw, cumulative_dist

        # Generate splines and calculate the paths for all waypoints in one go
        splines = [make_interp_spline(waypoints[i, :, 0], waypoints[i, :, 1], k=spline_order) for i in range(num_paths)]
        path_shifts = np.array([spline(path_r) for spline in splines])  # Shape: (num_paths, num_points)

        # Calculate path_x and path_y using batch operations
        path_x = path_r * np.cos(np.deg2rad(path_shifts))  # Shape: (num_paths, num_points)
        path_y = path_r * np.sin(np.deg2rad(path_shifts))  # Shape: (num_paths, num_points)

        # Calculate dx, dy using numpy's batch gradient calculation
        dx = np.gradient(path_x, axis=1)  # Shape: (num_paths, num_points)
        dy = np.gradient(path_y, axis=1)  # Shape: (num_paths, num_points)
        path_yaw = np.arctan2(dy, dx)  # Shape: (num_paths, num_points)

        # Calculate cumulative distance along each path
        dist = np.sqrt(dx**2 + dy**2)  # Shape: (num_paths, num_points)
        cumulative_dist = np.cumsum(dist, axis=1)  # Shape: (num_paths, num_points)
        cumulative_dist = np.hstack(
            (np.zeros((num_paths, 1)), cumulative_dist[:, :-1])
        )  # Add a leading zero for each path

        # Stack the generated path data (x, y, yaw, cumulative_dist) together
        paths[:, :, 0] = path_x
        paths[:, :, 1] = path_y
        paths[:, :, 2] = path_yaw
        paths[:, :, 3] = cumulative_dist

        # Apply initial heading to all paths using the vectorized function
        rotated_paths = self._apply_initial_heading(num_paths, paths, rotate_angle)

        return rotated_paths  # Shape: (num_paths, num_points, 4)

    def _apply_initial_heading(self, num, path, rotate_angle):
        # The initial heading angle is randomly selected from:
        # [-rotate_angle, -(rotate_angle - self.rotation_res)] and [rotate_angle - self.rotation_res, rotate_angle]

        # Randomly choose initial heading angles for all paths within the specified ranges
        rand_choices = np.random.rand(num) < 0.5  # Boolean array to pick between the two intervals

        # Generate random angles for the first and second intervals
        first_interval = np.random.uniform(-rotate_angle, -(rotate_angle - self.rotation_res), size=num)
        second_interval = np.random.uniform(rotate_angle - self.rotation_res, rotate_angle, size=num)

        # Select angles based on the random choices
        initial_heading_angles = np.where(rand_choices, first_interval, second_interval)  # Shape: (num,)

        # Convert the angles to radians
        initial_heading_angles_rad = initial_heading_angles * np.pi / 180.0  # Shape: (num,)

        # Create rotation matrices for each angle and reshape to (num, 2, 2)
        cos_vals = np.cos(initial_heading_angles_rad)  # Shape: (num,)
        sin_vals = np.sin(initial_heading_angles_rad)  # Shape: (num,)
        rotation_matrices = np.stack([cos_vals, -sin_vals, sin_vals, cos_vals], axis=-1).reshape(
            num, 2, 2
        )  # Shape: (num, 2, 2)

        if path.shape[1] == 4:
            if rotate_angle < self.rotation_res:
                return np.tile(path, (num, 1, 1))

            # Rotate all path points simultaneously
            path_copies = np.tile(path[:, :2], (num, 1, 1))  # Shape: (num, N, 2)
            rotated_paths = np.einsum("nij,nkj->nki", rotation_matrices, path_copies)  # Shape: (num, N, 2)

            # Apply rotation and update heading
            rotated_paths_full = np.tile(path, (num, 1, 1))  # Shape: (num, N, 4)
            rotated_paths_full[:, :, :2] = rotated_paths  # Update x, y coordinates
            rotated_paths_full[:, :, 2] += initial_heading_angles_rad[:, np.newaxis]  # Update yaw
        else:
            assert path.shape[0] == num, "Number of paths should match the first dimension of the input path."
            if rotate_angle < self.rotation_res:
                return path

            # Apply rotation and update heading using correct shape
            rotated_path = np.einsum(
                "nij,nkj->nki", rotation_matrices, path[..., :2]
            )  # Shape: (num_paths, num_points, 2)
            path[..., :2] = rotated_path
            path[..., 2] += initial_heading_angles_rad[:, np.newaxis]
            rotated_paths_full = path

        return rotated_paths_full

    def generate_path_group(self, dis: float, angle: float, rotate_angle: float, num: int):
        if dis <= self.max_length:
            paths, num_path = self.generate_paths(dis / 3.0, angle, rotate_angle, num)
            # paths, num_path = self.generate_paths(dis / 3.0, angle, rotate_angle, num, spline_order=1 + np.random.randint(3))
        else:
            num_sections = int(np.ceil(dis / self.max_length))
            paths_section, num_path = self.generate_paths(self.max_length / 3.0, angle, rotate_angle, num)
            # paths_section, num_path = self.generate_paths(self.max_length / 3.0, angle, rotate_angle, num, spline_order=1 + np.random.randint(3))
            paths = paths_section
            for _ in range(num_sections - 1):
                paths_section, num_path = self.generate_paths(self.max_length / 3.0, angle, rotate_angle, num)
                paths = self._concatenate_paths(paths, paths_section)
        return paths, num_path
    
    def generate_rsl_path_group(self, num: int):
        ## RSL path generation
        path = generate_rsl_path(resolution=0.01, scale=3.0, num_rounds = 1) # (num_points, 4)
        paths = np.repeat(path[np.newaxis, :, :], num, axis=0)  # shape: (num, num_points, 4)
        num_path = num
        return paths, num_path

    def pad_to_max_length(self, arr, max_length):
        """Pad the array to the specified max_length by repeating the last value."""
        if arr.shape[1] < max_length:
            padding = np.repeat(arr[:, -1:, :], max_length - arr.shape[1], axis=1)
            arr = np.concatenate((arr, padding), axis=1)
        return arr

    def sample_paths(
            self, params_list: list[list], speed_list: list[float],
            num_list: list[int], use_rsl_path: bool, seed: int = 1
        ):

        np.random.seed(seed)
        data = []
        param_sets = []
        max_num_points = 0
        # Generate and sample paths
        for params, num, speed in zip(params_list, num_list, speed_list):

            angle = params[0]
            r_angle = params[1]
            length_agument = 1.0 + angle / 180.0  # the total length of the path increases with the spline angle
            if speed < 1.5 * length_agument:
                dis = np.random.randint(2, 4) * self.max_length  # Randomly select the distance
            elif speed < 2.5 * length_agument:
                dis = np.random.randint(3, 5) * self.max_length
            elif speed < 3.2 * length_agument:
                dis = np.random.randint(4, 6) * self.max_length
            else:
                dis = np.random.randint(5, 7) * self.max_length
            # dis = 3 * self.max_length
            if use_rsl_path:
                paths, num_path = self.generate_rsl_path_group(num)
            else:
                paths, num_path = self.generate_path_group(dis, angle, r_angle, num)
            curr_data = []

            if num <= num_path:
                sampled_indices = np.random.choice(num_path, num, replace=False)
            else:
                # first sample all paths, then randomly sample the rest
                sampled_indices = np.arange(num_path)
                sampled_indices = np.concatenate(
                    (sampled_indices, np.random.choice(num_path, num - num_path, replace=True))
                )
            curr_data = paths[sampled_indices]  # shape: (num, num_points, 4)

            max_num_points = max(max_num_points, curr_data.shape[1])
            curr_param_set = np.repeat(np.array(params).reshape(1, -1), num, axis=0)  # shape: (num, 3)

            data.append(curr_data)
            param_sets.append(curr_param_set)

        # Pad paths to have the same number of points using the last point
        padded_data = []
        for path_group in data:
            num_paths, num_points, _ = path_group.shape
            if num_points < max_num_points:
                last_points = path_group[:, -1:, :]  # Shape: (num_paths, 1, 3)
                padding = np.repeat(last_points, max_num_points - num_points, axis=1)
                path_group = np.concatenate((path_group, padding), axis=1)
            padded_data.append(path_group)

        # Stack all sampled paths and parameters
        data = np.concatenate(padded_data, axis=0).astype(np.float32)  # shape: (sum(num_paths), max_num_points, 4)
        param_sets = np.concatenate(param_sets, axis=0).astype(np.float32)  # shape: (sum(num_paths), 3)
        return data, param_sets
