# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_teacher_env_cfg import ObservationsCfg as TeacherObservationsCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_teacher_env_cfg import LocomotionVelocityRoughEnvCfg

LEG_JOINT_NAMES = [".*HAA", ".*HFE", ".*KFE"]

##
# MDP settings
##


@configclass
class ObservationsCfg(TeacherObservationsCfg):
    """Observation specifications for the MDP."""
    
    @configclass
    class PoliStudentObservationsCfgcyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # 3
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # 3
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # 3
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)}
        )  #12
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))  # 16
        actions = ObsTerm(func=mdp.last_action)

    policy: PoliStudentObservationsCfgcyCfg = PoliStudentObservationsCfgcyCfg()
    teacher: TeacherObservationsCfg.PolicyCfg = TeacherObservationsCfg().PolicyCfg()

##
# Environment configuration
##


@configclass
class StudentVelocityTrackingEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
