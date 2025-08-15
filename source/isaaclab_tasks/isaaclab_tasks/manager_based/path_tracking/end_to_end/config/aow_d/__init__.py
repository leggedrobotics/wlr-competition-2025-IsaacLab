# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, student_env_cfg, teacher_env_cfg

####################
# TEACHER ENVIRONMENTS
####################

##
# Register Roll environments.
##
gym.register(
    id="Isaac-Path-Teacher-Anymal-D-Dev-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": teacher_env_cfg.AoWDPathEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AoWDPathPPORunnerCfg_DEV,
    },
)

gym.register(
    id="Isaac-Path-Teacher-Anymal-D-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": teacher_env_cfg.AoWDPathEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AoWDPathPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Path-Teacher-Anymal-D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": teacher_env_cfg.AoWDPathEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.AoWDPathPPORunnerCfg,
    },
)


####################
# STUDENT ENVIRONMENTS
####################
##
# Register Roll environments.
##
gym.register(
    id="Isaac-Path-Student-Anymal-D-Dev-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": student_env_cfg.AoWDPathEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StudentPathRecurrentCfg_DEV,
    },
)

gym.register(
    id="Isaac-Path-Student-Anymal-D-Train-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": student_env_cfg.AoWDPathEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StudentPathRecurrentCfg,
    },
)

gym.register(
    id="Isaac-Path-Student-Anymal-D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": student_env_cfg.AoWDPathEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StudentPathRecurrentCfg,
    },
)