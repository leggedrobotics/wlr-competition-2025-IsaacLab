# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import csv
import numpy as np
import torch
from datetime import datetime


class GridBasedDistribution:
    def __init__(
        self,
        param_bounds,
        resolutions,
        initial_curr_params,
        initial_speed,
        speed_increment=0.2,
        maximum_speed=5.0,
        weight_constant=2.0,
        device="cpu",
    ):
        """Curriculum sampler for sampling parameters based on a grid.
        Notice:
        Avoid using negative values in the param_bounds since we use torch.floor()
        to convert continuous parameters to grid indices. If u need to use negative values,
        an easy way is to first use positive values and then use linear transformation to convert
        the positive values to any negative values you want.
        Also, by using linear transformation, you can set a range [b, a] where a > b but u want
        the curriculum to start from a to b.
        Or u can also change the conversion logic. use torch.round() or torch.ceil().
        """

        self.param_bounds = param_bounds  # Bounds for each parameter [(min, max), ...]
        self.resolutions = torch.tensor(resolutions, device=device)  # Resolution for each parameter
        self.grid_shape = [(int((b[1] - b[0]) / res) + 1) for b, res in zip(param_bounds, resolutions)]
        self.grid = torch.zeros(self.grid_shape, dtype=torch.float32, device=device)

        # -- curriculum parameters --
        self.minimum_progress = torch.ones(self.grid_shape, dtype=torch.float32, device=device)
        self.maximum_track_error = torch.zeros(self.grid_shape, dtype=torch.float32, device=device)
        self.minimum_robot_speed = 10 * torch.ones(self.grid_shape, dtype=torch.float32, device=device)
        self.average_speed_error = torch.zeros(self.grid_shape, dtype=torch.float32, device=device)
        self.curriculum_counter = torch.zeros(self.grid_shape, dtype=torch.int32, device=device)

        self.initial_max_speed = initial_speed
        self.maximum_speed = maximum_speed
        self.speed_increment = speed_increment
        self.weight_constant = weight_constant
        self.device = device
        max_bounds = torch.tensor(initial_curr_params, device=device)
        self.max_bounds_idx = self.param_to_grid_idx(max_bounds).reshape(-1, len(initial_curr_params))
        self._initialize_grid(self.max_bounds_idx)

    def _initialize_grid(self, initial_idx):
        """Initialize the grid to reflect the initial curriculum idx."""
        # Create a grid mask where all values from min_bounds to initial_idx are set to default speed value
        indices = torch.cartesian_prod(*[torch.arange(0, size, device=self.device) for size in self.grid_shape])
        min_idx = torch.zeros(len(self.grid_shape), dtype=torch.int32, device=self.device)
        mask = torch.all((indices >= min_idx) & (indices <= initial_idx), dim=1)
        self.grid[indices[mask].T.tolist()] = self.initial_max_speed

    def param_to_grid_idx(self, params):
        """Convert continuous parameters to grid indices."""
        return torch.floor(
            (params - torch.tensor([b[0] for b in self.param_bounds], device=self.device)) / self.resolutions
        ).int()

    def grid_idx_to_param(self, idx):
        """Convert grid indices to continuous parameters."""
        idx = idx.float()
        return (idx * self.resolutions + torch.tensor([b[0] for b in self.param_bounds], device=self.device)).tolist()

    def check_speed_params(self, sampled_params):
        """Check the speed value (grid value) based on the sampled parameters."""
        idxs = [self.param_to_grid_idx(torch.tensor(param, device=self.device)) for param in sampled_params]
        speed_values = torch.tensor([self.grid[tuple(idx.tolist())] for idx in idxs], device=self.device)
        return speed_values

    def calculate_weights(self, valid_indices):
        """Calculate weights for valid indices based on their distance to max bounds."""
        # Normalize indices by dividing each by the maximum index in its dimension (from self.grid_shape)
        max_indices = torch.tensor(self.grid_shape, device=valid_indices.device) - 1  # -1 for zero-based indexing

        # Ignore the last dimension (tolerance) when normalizing indices
        modified_indices = valid_indices.clone()
        modified_indices[:, 2] = max_indices[2]  # Set the tolerance to the maximum value
        modified_indices[:, 3] = max_indices[3]  # Set the terrain level to the maximum value

        # Avoid division by zero
        max_indices = torch.where(max_indices == 0, torch.tensor(1, device=valid_indices.device), max_indices)
        normalized_valid_indices = modified_indices / max_indices
        normalized_max_bounds_idx = self.max_bounds_idx / max_indices
        # Calculate distances using normalized indices
        distances = torch.cdist(normalized_valid_indices.float(), normalized_max_bounds_idx.float())
        min_distances, _ = torch.min(distances, dim=1)

        # Calculate weights inversely proportional to distances to max bounds
        weights = 1 / (min_distances + self.weight_constant)

        weights = weights / torch.sum(weights)  # Normalize weights
        return weights

    def sample_params(self, num_samples=1, use_weighted_sampling=False):
        """Sample parameters from the valid areas in the grid with bias towards max bounds.

        Args:
            num_samples (int): Number of parameter sets to sample.
            use_weighted_sampling (bool): Whether to use weighted sampling based on distances to the boundaries.

        Returns:
            List of non-repeated sampled parameter sets, a list of counts for each parameter set and the corresponding speeds.
            sampled_params: List[List[float | int]]
            counts: List[int]
            speeds: Tensor of shape (len(sampled_params),)
        """
        valid_indices = torch.nonzero(self.grid, as_tuple=False).float()
        if len(valid_indices) == 0:
            raise ValueError("No valid parameter sets available.")

        # Calculate weights based on distances to max bounds
        if use_weighted_sampling:
            weights = self.calculate_weights(valid_indices)  # Tensor of shape (len(valid_indices),)
        else:
            weights = torch.ones(valid_indices.shape[0], device=self.device) / valid_indices.shape[0]

        # Create weights
        # if valid_indices.shape[0] > 1:
        #     weights = torch.ones(valid_indices.shape[0], device=self.device)  # Start with equal weights
        #     weights[0] = 0.1  # Set the first element to 10%
        #     weights[1:] = 0.9 / (valid_indices.shape[0] - 1)  # Distribute the remaining 90% equally
        #     weights /= weights.sum()

        # Sample indices based on the calculated weights
        if weights.numel() == 1:
            sampled_indices = torch.zeros(num_samples, dtype=torch.int32)
        else:
            if weights.numel() < num_samples:
                # First sample all the indices once, then sample the remaining indices based on the calculated weights
                sampled_indices1 = torch.arange(weights.numel(), device=self.device)
                sampled_indices2 = torch.multinomial(weights, num_samples - weights.numel(), replacement=True)
                sampled_indices = torch.cat((sampled_indices1, sampled_indices2))
            else:
                sampled_indices = torch.multinomial(weights, num_samples, replacement=False)

        unique_indices, counts = torch.unique(sampled_indices, return_counts=True)
        sampled_params = [self.grid_idx_to_param(valid_indices[i].int()) for i in unique_indices]
        counts = counts.tolist()
        speeds = self.check_speed_params(sampled_params)

        return sampled_params, counts, speeds

    def update_grid(self, params, curriculum_speeds):
        """Update the grid based on successful parameters."""
        if not isinstance(params[0], (list, tuple)):
            params = [params]
            curriculum_speeds = [curriculum_speeds]

        for param, robot_speed in zip(params, curriculum_speeds):
            idx = self.param_to_grid_idx(torch.tensor(param, device=self.device))
            if robot_speed > 0:
                # 1. Increase the max_speed value of the grids by a certain amount: self.speed_increment
                # Grids that have smaller index values should increase their speed values
                # mask = torch.ones(self.grid.shape, dtype=torch.bool, device=self.device)
                # for i, val in enumerate(idx):
                #     grid_indices = torch.arange(self.grid_shape[i], device=self.device).view(-1, *([1] * (len(self.grid_shape) - 1)))
                #     grid_indices = grid_indices.transpose(0, i)
                #     mask &= (grid_indices <= val)
                current_grid_value = torch.clamp(robot_speed, max=self.maximum_speed)
                # self.grid[mask] = torch.where(self.grid[mask] < current_grid_value, current_grid_value, self.grid[mask])
                # self.grid[mask] = torch.where((self.grid[mask] < current_grid_value) & (self.grid[mask] > 0),
                #                               self.grid[mask] + self.speed_increment, self.grid[mask])
                self.grid[tuple(idx.tolist())] = current_grid_value
                # 2. Find the neighbors of the grid outwards (not inwards)
                neighbors = self.get_outward_neighbors(idx)

                # 3. If the neighbors are not in the valid areas(grid value = 0), expand the areas with grid values = self.initial_max_speed
                neighbor_idxs = torch.stack(neighbors)
                mask = self.grid[neighbor_idxs.T.tolist()] == 0.0
                self.grid[neighbor_idxs[mask].T.tolist()] = self.initial_max_speed

                # 4. Update the max_bounds_idx with all the neighbors
                new_max_bounds_idx = []
                for neighbor in neighbor_idxs[mask]:
                    # Only add the neighbor if it extends beyond any current boundary point
                    if not any(torch.all(neighbor <= bound) for bound in self.max_bounds_idx):
                        new_max_bounds_idx.append(neighbor)

                new_max_bounds_idx = (
                    torch.stack(new_max_bounds_idx)
                    if new_max_bounds_idx
                    else torch.empty((0, len(self.grid_shape)), dtype=torch.int32, device=self.device)
                )

                # Combine and filter the new and old max bounds to maintain only the furthest boundary points
                combined_idxs = torch.cat((self.max_bounds_idx, new_max_bounds_idx), dim=0)
                unique_max_bounds_idx = []
                for i in range(combined_idxs.shape[0]):
                    if not any(
                        torch.all(combined_idxs[i] <= combined_idxs[j]) and i != j
                        for j in range(combined_idxs.shape[0])
                    ):
                        unique_max_bounds_idx.append(combined_idxs[i])

                self.max_bounds_idx = torch.stack(unique_max_bounds_idx)

            else:
                # Decrease the max_speed value of the grid by a certain amount: self.speed_increment
                self.grid[tuple(idx.tolist())] = self.initial_max_speed

    def get_outward_neighbors(self, idx):
        """Get outward neighbors of the given grid index."""
        neighbors = []
        for i in range(len(idx)):
            neighbors.append(idx + torch.eye(len(idx), dtype=torch.int32, device=self.device)[i])
            neighbors.append(idx - torch.eye(len(idx), dtype=torch.int32, device=self.device)[i])

        # Filter out neighbors that are out of bounds
        neighbors = [
            neighbor
            for neighbor in neighbors
            if all(neighbor >= 0) and all(neighbor < torch.tensor(self.grid_shape, device=self.device))
        ]
        return neighbors

    def save_grid_to_file(self, file_path):
        """Save the current grid configuration to a CSV file."""
        grid_cpu = self.grid.cpu().numpy()  # Move grid to CPU and convert to numpy for saving
        indices = np.argwhere(grid_cpu > 0)  # Get indices of all non-zero elements
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write parameter bounds
            writer.writerow(["Param Bounds:"] + [f"[{b[0]}, {b[1]}]" for b in self.param_bounds])
            # Write resolutions
            writer.writerow(["Resolutions:"] + [self.resolutions.cpu().numpy().tolist()])
            # Write additional attributes
            writer.writerow(["Initial Max Speed:", self.initial_max_speed])
            writer.writerow(["Maximum Speed:", self.maximum_speed])
            writer.writerow(["Speed Increment:", self.speed_increment])
            # Write max bounds indices
            writer.writerow(["Max Bounds Indices:"])
            for idx in self.max_bounds_idx.cpu().numpy():
                writer.writerow([idx.tolist()])
            # Write header for the CSV file
            header = ["Spline_angle", "Rotate_angle", "Tolerance", "Terrain_level", "Speed"]
            writer.writerow(header)
            # Iterate over all non-zero grid indices
            for idx in indices:
                speed = grid_cpu[tuple(idx)]
                writer.writerow([idx.tolist(), speed])

    def load_grid_from_file(self, file_path):
        """Load the grid configuration from a CSV file."""
        import csv

        with open(file_path) as file:
            reader = csv.reader(file)
            rows = list(reader)

            # Read parameter bounds
            param_bounds = [tuple(eval(pb)) for pb in rows[0][1:]]  # Convert string to tuples
            self.param_bounds = param_bounds

            # Read resolutions
            self.resolutions = torch.tensor(eval(rows[1][1]), device=self.device)  # Convert string to list

            # Read additional attributes
            self.initial_max_speed = float(rows[2][1])
            self.maximum_speed = float(rows[3][1])
            self.speed_increment = float(rows[4][1])

            # Read max bounds indices dynamically until "Length" header
            max_bounds_indices = []
            for i, row in enumerate(rows[6:], start=6):  # Start reading after the first 5 rows
                if row[0].startswith("Length"):
                    header_index = i
                    break
                idx = eval(rows[i][0])  # Convert string to list
                max_bounds_indices.append(idx)
            self.max_bounds_idx = torch.tensor(max_bounds_indices, dtype=torch.int32, device=self.device)

            # Initialize grid with zeros
            self.grid = torch.zeros(self.grid_shape, dtype=torch.float32, device=self.device)

            # Read grid configuration data
            for row in rows[header_index + 1 :]:
                if not row:
                    continue
                idx = eval(row[0])
                speed = float(row[1])
                self.grid[tuple(idx)] = speed


if __name__ == "__main__":
    # Example usage:
    param_bounds = [(9.0, 25.0), (9, 25.0), (0.0, 0.4)]
    resolutions = [1.0, 1.0, 0.1]
    initial_curr_params = [9.0, 9.0, 0.0]
    sampler = GridBasedDistribution(param_bounds, resolutions, initial_curr_params, initial_speed=2.0, device="cuda")
    print(f"Initial Grid Shape:\n{sampler.grid_shape}")

    # Sample some parameters from initial distribution
    # sampled_params = sampler.sample_params(num_samples=20)
    # print(f"Sampled Params: {sampled_params}")

    # From other code ... some training process
    # From other code ... evaluate the performance of robots under sampled_params each fixed number of iterations
    # From other code ... decide the gird-params in which the performance is good or bad

    # There are two stpes to update the grid:
    # 1. Update the grid values, which represent the maximum speed for each parameter set. can be increased or decreased
    # 2. Update the grid areas, which represent the valid areas for sampling. can be expanded or contracted, 4 parameters, 4 dimensions

    # Curriculum params can be either a single set of params or a list of params
    curriculum_params_single = [9.0, 9.0, 0.1]
    curriculum_flags_single = torch.scalar_tensor(2.33)

    curriculum_params_list = [[9.0, 9.0, 0.0], [9.0, 9.0, 0.0]]
    curriculum_flags_list = [torch.scalar_tensor(3.33), torch.scalar_tensor(4.33)]

    sampler.update_grid(curriculum_params_single, curriculum_flags_single)
    # sampler.update_grid(curriculum_params_list, curriculum_flags_list)
    sampled_params, counts, speeds = sampler.sample_params(num_samples=500)
    print(f"Sampled Params: {sampled_params}", f"\nCounts: {counts}")
    print(f"speed: {speeds.shape}\n{speeds}")

    # Save the grid to a CSV file
    # file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-") + "grid_config.csv"
    # sampler.save_grid_to_file(file_name)
    # print(f"Grid saved to {file_name}")

    # # Load grid from file
    # new_sampler = GridBasedDistribution(param_bounds, resolutions, initial_curr_params, initial_speed=2.0, device='cuda')
    # new_sampler.load_grid_from_file(file_name)
    # print(f"Loaded Grid Shape:\n{new_sampler.grid_shape}")

    # # Check if the grid was loaded correctly
    # print("Grid equal:", torch.equal(sampler.grid, new_sampler.grid))
    # print("Param bounds equal:", sampler.param_bounds == new_sampler.param_bounds)
    # print("Resolutions equal:", torch.equal(sampler.resolutions, new_sampler.resolutions))
    # print("Max bounds indices equal:", torch.equal(sampler.max_bounds_idx, new_sampler.max_bounds_idx))
    # print("Initial max speed equal:", sampler.initial_max_speed == new_sampler.initial_max_speed)
    # print("Maximum speed equal:", sampler.maximum_speed == new_sampler.maximum_speed)
    # print("Speed increment equal:", sampler.speed_increment == new_sampler.speed_increment)
