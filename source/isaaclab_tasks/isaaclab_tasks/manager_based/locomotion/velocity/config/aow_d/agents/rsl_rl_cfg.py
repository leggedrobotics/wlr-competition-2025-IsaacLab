# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from isaaclab.utils import configclass

# from isaaclab_rl.rsl_rl import (
#     OnPolicyRunnerCfg, OnPolicyDistillationRunnerCfg,
#     RslRlActorCriticSeparateCfg, RslRlActorCriticSharedRecurrentCfg,
#     RslRlPPOAlgorithmCfg, RslRlDistillationAlgorithmCfg,
#     TrackingArchitectureCfg, StudentTrackingArchitectureCfg,
#     ActionDistributionCfg, RslRlSymmetryCfg
# )
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg, RslRlSymmetryCfg,
    RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherRecurrentCfg
)
# from isaaclab_tasks.manager_based.locomotion.velocity.data_augmentation_vel import (
#     get_symmetric_states,
#     get_symmetric_states_scan,
# )

##
# Teacher Policy
##

# Train vel-tracking high level control env
@configclass
class AoWDVelPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name = "Vel_Tracking_AoW_D"
    run_name = "Teacher_PPO"
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 500
    empirical_normalization = True

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # symmetry_cfg=RslRlSymmetryCfg(
        #     use_data_augmentation=True,
        #     data_augmentation_func=get_symmetric_states,
        # ),
    )


@configclass
class AoWDVelPPORunnerCfg_DEV(AoWDVelPPORunnerCfg):
    logger = "tensorboard"
    run_name = "Dev_" + AoWDVelPPORunnerCfg().run_name
    experiment_name = "Dev_" + AoWDVelPPORunnerCfg().experiment_name