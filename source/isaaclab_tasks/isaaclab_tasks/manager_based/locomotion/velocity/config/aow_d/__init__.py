# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, teacher_env_cfg

####################
# TEACHER ENVIRONMENTS
####################

##
# Register Roll environments.
##
gym.register(
    id="Isaac-Vel-Teacher-Anymal-D-Dev-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": teacher_env_cfg.AoWDVelEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AoWDVelPPORunnerCfg_DEV,
    },
)

gym.register(
    id="Isaac-Vel-Teacher-Anymal-D-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": teacher_env_cfg.AoWDVelEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AoWDVelPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Vel-Teacher-Anymal-D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": teacher_env_cfg.AoWDVelEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AoWDVelPPORunnerCfg,
    },
)