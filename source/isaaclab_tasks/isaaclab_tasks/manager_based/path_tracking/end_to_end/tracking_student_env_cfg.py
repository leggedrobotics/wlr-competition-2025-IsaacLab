# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_tasks.manager_based.path_tracking.end_to_end.tracking_teacher_env_cfg import CurriculumCfg as TeacherCurriculumCfg
from isaaclab_tasks.manager_based.path_tracking.end_to_end.tracking_teacher_env_cfg import ObservationsCfg as TeacherObservationsCfg
from isaaclab_tasks.manager_based.path_tracking.end_to_end.tracking_teacher_env_cfg import TeacherPathTrackingEnvCfg

##
# MDP settings
##


@configclass
class ObservationsCfg(TeacherObservationsCfg):
    """Observation specifications for the MDP."""
    
    @configclass
    class StudentObservationsCfg(TeacherObservationsCfg.PolicyCfg):
        """Observations for the student policy."""

        def __post_init__(self):
            ###############################################
            # 1. proprioceptive group
            ###############################################
            self.history_joint_pos_error = None
            self.history_joint_vel = None
            self.history_root_lin_vel = None
            self.history_root_ang_vel = None
            self.history_root_pos = None

            ###############################################
            # 2. exteroceptive group
            ###############################################
            self.height_scan = None

            ###############################################
            # 3. tracking group
            ###############################################
            self.pos_err_tolerance_vec = None
            self.wp_interval = None
            self.waypoints_list_time = None

            ###############################################
            # 3. privileged group
            ###############################################
            self.feet_contact_state = None
            self.feet_friction_static = None
            self.feet_friction_dynamic = None
            self.ground_friction = None
            self.penalized_contact_state = None
            self.external_wrench = None
            self.air_time = None
            self.mass_disturbance = None
            self.push_velocity = None

    policy: StudentObservationsCfg = StudentObservationsCfg()
    teacher: TeacherObservationsCfg.PolicyCfg = TeacherObservationsCfg().PolicyCfg()


@configclass
class CurriculumCfg(TeacherCurriculumCfg):
    """Curriculum specifications for the MDP."""

    set_tolerance_res = CurrTerm(
        func=mdp.set_tolerance_res,
        params={
            "term_name": "path_command",
            "update_rate_steps": 500 * 48,
            "start_step": 3500 * 48,
            "end_step": 4500 * 48,
            "decay_rate": 0.5,
        },
    )


##
# Environment configuration
##


@configclass
class StudentPathTrackingEnvCfg(TeacherPathTrackingEnvCfg):
    """Configuration for the locomotion path-tracking environment."""

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
