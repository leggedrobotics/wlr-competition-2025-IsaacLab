import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def generate_random_path(
    num_paths=1,
    num_points=100,
    sharp_turns=False,
    oscillations=False,
    varying_curvature=False,
    noise_intensity=0.0,
    sharp_turn_intensity=0.5,
    oscillation_frequency=5,
    curvature_variation_intensity=0.3,
    random_seed=None,
    scale=1.0
):
    """
    Generate a random path with optional challenges: sharp turns, oscillations, varying curvature, and noise.
    
    Args:
        num_paths (int): Number of random paths to generate.
        num_points (int): Number of waypoints along each path.
        sharp_turns (bool): If True, introduces sharp turns into the path.
        oscillations (bool): If True, introduces oscillations into the path.
        varying_curvature (bool): If True, introduces varying curvature into the path.
        noise_intensity (float): Intensity of noise to add to the path (0 means no noise).
        sharp_turn_intensity (float): Intensity of sharp turns if sharp_turns is True.
        oscillation_frequency (int): Number of oscillations along the path if oscillations is True.
        curvature_variation_intensity (float): Intensity of varying curvature if varying_curvature is True.
        random_seed (int): Random seed for reproducibility.
        scale (float): Approximate length of the entire path.
    
    Returns:
        np.ndarray: Array of shape (num_paths, num_points, 4) representing the (x, y, tangent_angle, cumulative_distance) of the path.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    paths = []
    
    for _ in range(num_paths):
        # Start with a linear path from (0, 0) to (1, 1)
        x = np.linspace(0, 1, num_points)
        y = 0.5 * np.sin(2 * np.pi * np.linspace(0, 1, num_points))

        # Apply challenges based on input flags
        if sharp_turns:
            sharp_turn_indices = np.random.choice(np.arange(1, num_points-1), size=int(num_points * sharp_turn_intensity), replace=False)
            y[sharp_turn_indices] += np.random.uniform(-0.5, 0.5, size=len(sharp_turn_indices))

        if oscillations:
            y += 0.1 * np.sin(oscillation_frequency * np.pi * x)


        if noise_intensity > 0:
            x += np.random.normal(0, noise_intensity, size=num_points)
            y += np.random.normal(0, noise_intensity, size=num_points)

        # Ensure x is strictly increasing by sorting and removing duplicates
        x, unique_indices = np.unique(x, return_index=True)
        y = y[unique_indices]  # Ensure y matches the sorted, unique x values

        # Interpolate the path to make it smooth
        path_spline = CubicSpline(x, y)
        x_smooth = np.linspace(0, 1, num_points * 5 *int(scale))  # High resolution for smoothness
        y_smooth = path_spline(x_smooth)

        # Apply the scale to both x and y coordinates to change the total path length
        x_smooth *= scale
        y_smooth *= scale

        # Calculate the tangent angle (yaw) at each point
        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        tangent_angle = np.arctan2(dy, dx)

        # Calculate the cumulative distance along the path
        dist = np.sqrt(dx**2 + dy**2)
        cumulative_distance = np.cumsum(dist)
        cumulative_distance[0] = 0  # Start with 0 distance at the first point

        # Create a path array with shape (num_points, 4)
        path = np.vstack((x_smooth, y_smooth, tangent_angle, cumulative_distance)).T
        paths.append(path)

    # Stack paths to have shape (num_paths, num_points, 4)
    paths = np.stack(paths)
    
    return paths, paths.shape[0]

# Example usage
path, _ = generate_random_path(
    num_points=100,
    sharp_turns=True,
    oscillations=True,
    varying_curvature=True,
    noise_intensity=0.002,
    sharp_turn_intensity=0.1,
    oscillation_frequency=8,
    curvature_variation_intensity=0.1,
    random_seed=42,
    scale=10.0  # This sets the total path length to be approximately 10 units
)

# Plot the path
plt.plot(path[:, 0], path[:, 1], label="Random Path")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Random Path with Sharp Turns, Oscillations, Curvature, and Noise")
plt.legend()
plt.show()
