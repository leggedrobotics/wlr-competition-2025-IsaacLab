# path_command_cfg.py

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for path command generator."""

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .path_command import PathCommand


@configclass
class PathCommandCfg(CommandTermCfg):
    """Configuration for path command generator."""

    class_type: type = PathCommand
    """Type of the command generator class."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    path_config: dict = {
        "spline_angle_range": (0, 60.0),
        "rotate_angle_range": (0, 90.0),
        "pos_tolerance_range": (0.1, 0.5),
        "terrain_level_range": (0, 0),
        "resolution": [1.0, 1.0, 0.1, 1],
        "initial_params": [0.0, 0.0, 0, 0],
    }
    """Configuration for the path. Contains list of different path configurations."""

    num_waypoints: int = 10
    """Number of waypoints to be maintained in the path."""

    std_waypoint_interval: float = 0.15
    """Standard distance between two waypoint along the path [m] correspondent to v = 1m/s."""

    max_speed: float = 2.0
    """Maximum speed of the robot [m/s]."""

    convert_to_vel: bool = False
    """Whether to convert the path to velocity commands."""

    rel_standing_envs: float = 0.0
    """Probability threshold for environments where the robots that are standing still."""

    enable_backward: bool = True
    """Whether to enable backward movement. If False, the robot will not be encouraged to move backward."""

    testing_mode: bool = False
    """Whether to enable testing mode. In testing mode, the visualization markers are abundant but the performance is slow."""

    use_rsl_path: bool = False
    """Whether to use the 'rsl' path. If true, the polynomial spline won't be used."""