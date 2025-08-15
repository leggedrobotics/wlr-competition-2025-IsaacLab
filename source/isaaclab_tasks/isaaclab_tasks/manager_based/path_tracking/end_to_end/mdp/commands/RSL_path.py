# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np


def compute_cumulative_distances(path):
    """Compute the cumulative distance along a 2D path."""
    distances = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    return cumulative_distances


def compute_tangent_angles(path):
    """Compute the tangent angle (yaw) along the path."""
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    tangent_angles = np.arctan2(dy, dx)
    return tangent_angles


def resample_path(path, resolution):
    """Resample the path to have approximately the given resolution."""
    cumulative_distances = compute_cumulative_distances(path)
    total_length = cumulative_distances[-1]
    num_samples = int(total_length / resolution) + 1
    sampled_distances = np.linspace(0, total_length, num_samples)
    resampled_path = np.zeros((num_samples, 2))

    for i in range(2):  # Interpolate both x and y coordinates
        resampled_path[:, i] = np.interp(sampled_distances, cumulative_distances, path[:, i])

    return resampled_path


def generate_rsl_path(
    resolution: float = 0.01,
    scale: float = 5.0,
    num_rounds: int = 1
) -> np.ndarray:
    """    Generate a path in the shape of "RSL" with specified resolution and scale.
    Args:
        resolution (float): The desired resolution of the path.
        scale (float): The scaling factor for the path. Default is 5.0.
        num_rounds (int): The number of times to repeat the path.
    Returns:
        np.ndarray: A 2D array representing the path with shape (num_points, 4), 
        where each row contains [x, y, tangent_angle, cumulative_distance].
    """
    # Define the components of the RSL path using numpy arrays  
    # R - components
    R_vertical = np.array([[0, 0], [0, 2]])  # Vertical line of R
    R_horizontal = np.array([[0, 2], [1.25, 2]])  # Top horizontal line of R
    R_arc = np.array([[1.25 + 0.5 * np.sin(t), 1.5 + 0.5 * np.cos(t)] for t in np.linspace(0, np.pi, 100)])  # Arc for R
    R_short_horizontal = np.array([[1.25, 1], [0.75, 1]])  # Short horizontal line after the arc
    R_diagonal = np.array([[0.75, 1], [1.75, 0]])  # Diagonal leg of R

    # S - components
    S_horizontal_1 = np.array([[0.0, 0], [2.0, 0]])  # Bottom horizontal line of S
    S_arc_1 = np.array(
        [[2.0 + 0.5 * np.sin(t), 0.5 + 0.5 * np.cos(t)] for t in np.linspace(np.pi, 0, 100)]
    )  # Arc for S
    S_horizontal_2 = np.array([[1.5, 1], [1.0, 1]])  # Middle horizontal line of S
    S_arc_2 = np.array(
        [[1.0 - 0.5 * np.sin(t), 1.5 + 0.5 * np.cos(t)] for t in np.linspace(np.pi, 0, 100)]
    )  # Arc for S
    S_horizontal_3 = np.array([[1.0, 2], [3.0, 2]])  # Top horizontal line of S

    # L - components
    L_vertical = np.array([[0, 2], [0, 0]])  # Vertical line of L
    L_horizontal = np.array([[0, 0], [1.5, 0]])  # Horizontal line of L

    # Concatenate the paths to create a continuous shape
    R_path = np.concatenate([R_vertical, R_horizontal, R_arc, R_short_horizontal, R_diagonal])
    S_path = np.concatenate([S_horizontal_1, S_arc_1, S_horizontal_2, S_arc_2, S_horizontal_3])
    L_path = np.concatenate([L_vertical, L_horizontal])
    back_to_R_path = np.array([[6.25, 0], [6.25, -0.05], [0, -0.05]])  # Connect L to R

    # Shift the S and L to connect them smoothly to R
    S_path[:, 0] += 1.75  # Shift S to the right
    L_path[:, 0] += 4.75  # Shift L to the right

    # Combine all paths
    full_path = np.concatenate([R_path, S_path, L_path, back_to_R_path])

    # Scale the path
    full_path *= scale

    # Resample the path to have the desired resolution
    resampled_path = resample_path(full_path, resolution)

    # Compute tangent angles and cumulative distances
    tangent_angles = compute_tangent_angles(resampled_path)  # Shape: (num_points,)
    cumulative_distances = compute_cumulative_distances(resampled_path)  # Shape: (num_points,)

    # Combine into a (num_points, 4) array
    extended_path = np.hstack((resampled_path, tangent_angles[:, np.newaxis], cumulative_distances[:, np.newaxis]))

    extended_path_ori = np.copy(extended_path)
    # Repeat the path for the specified number of rounds
    for _ in range(num_rounds - 1):
        now_distance = extended_path[-1, 3]
        i_round = np.copy(extended_path_ori)
        # the cumulative distance is shifted by the total length of the path
        i_round[:, 3] += now_distance
        extended_path = np.concatenate((extended_path, i_round))
    return extended_path


if __name__ == "__main__":
    # Generate and plot the path
    path = generate_rsl_path(resolution=0.01, scale=5.0)

    plt.figure(figsize=(8, 6))
    plt.plot(path[:, 0], path[:, 1], "b-", linewidth=2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("RSL Path with Resampling and Scaling")
    plt.grid(True)

    # Print path shape and a sample of its contents
    print("Path shape:", path.shape)
    plt.show()
