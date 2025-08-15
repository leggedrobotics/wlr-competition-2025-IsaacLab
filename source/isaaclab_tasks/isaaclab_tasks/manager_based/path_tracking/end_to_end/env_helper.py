# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp as mdp

# HELPER FUNCTIONS


def add_base_train_configuration(self):
    """Base configuration for training"""

    # number of environments
    self.viewer.origin_type = "asset_root"
    self.scene.num_envs = 4000

    # Set the number of terrains to train
    if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.num_rows = 6
        self.scene.terrain.terrain_generator.num_cols = 8


def add_base_play_configuration(self):
    """Base configuration for play"""

    self.viewer.origin_type = "asset_root"
    self.viewer.eye = (3.0, 3.0, 3.0)

    # make a smaller scene for play
    self.scene.num_envs = 50
    self.scene.env_spacing = 6.0

    # contact viz
    self.scene.contact_forces.debug_vis = True
    # self.scene.base_height_scanner.debug_vis = True

    # spawn the robot randomly in the grid (instead of their terrain levels)
    self.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2
        self.scene.terrain.terrain_generator.curriculum = False

    # remove random pushing event
    self.events.base_external_force_torque = None
    self.events.add_base_mass = None
    self.events.physics_material = None

    # disable curriculum for play
    self.curriculum.show_distribution_speed = None
    self.curriculum.show_distribution_terrain = None
    self.curriculum.show_path_progress = None

    # Adjust path settings
    self.commands.path_command.max_speed = 2.0
    self.commands.path_command.path_config = {
        "spline_angle_range": (180.0, 180.0),
        "rotate_angle_range": (60.0, 60.0),
        "pos_tolerance_range": (0.2, 0.2),
        "terrain_level_range": (0.0, 0.0),
        "resolution": [5.0, 5.0, 0.1, 1],
        "initial_params": [180.0, 60.0, 0.2, 0],
    }


def add_base_dev_configuration(self):
    """Base configuration for dev"""

    # number of environments
    self.scene.num_envs = 10

    # Height scans viz
    if hasattr(self.scene, "height_scanner") and self.scene.height_scanner is not None:
        self.scene.height_scanner.debug_vis = True

    # Commands viz
    if hasattr(self.commands, "path_command") and self.commands.path_command is not None:
        self.commands.path_command.debug_vis = True

    # spawn the robot randomly in the grid (instead of their terrain levels)
    self.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 1
        # self.scene.terrain.terrain_generator.curriculum = False

    # disable curriculum for play
    self.curriculum.path_length = None

    # Adjust path settings
    # self.commands.path_command.path_config = {
    #         "spline_angle_range": (0, 60.0),
    #         "rotate_angle_range": (0, 90.0),
    #         "pos_tolerance_range": (0.0, 0.4),
    #         "resolution": [1.0, 1.0, 0.1],
    #         "initial_params": [0.0, 0.0, 0.0],
    #     }
