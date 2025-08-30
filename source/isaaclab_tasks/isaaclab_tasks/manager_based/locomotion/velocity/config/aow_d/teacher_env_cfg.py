# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from isaaclab_assets.robots.aow import ANYMAL_D_ON_WHEELS_CFG

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.path_tracking.end_to_end.env_helper import (
    add_base_dev_configuration,
    add_base_play_configuration,
    add_base_train_configuration,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


"""
Vel-Tracking env (End-to-End)
"""

@configclass
class AoWDVelEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to aow-d
        self.scene.robot = ANYMAL_D_ON_WHEELS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (25.0, 25.0, 25.0)

        # scene settings
        # self.scene.height_scanner = None

        # Observation settings
        # self.observations.exte = None


class AoWDVelEnvCfg_DEV(AoWDVelEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # add base dev configuration
        add_base_dev_configuration(self)


@configclass
class AoWDVelEnvCfg_PLAY(AoWDVelEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # add base play configuration
        add_base_play_configuration(self)


@configclass
class AoWDVelEnvCfg_TRAIN(AoWDVelEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # add base train configuration
        add_base_train_configuration(self)
