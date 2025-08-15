# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp as mdp
from isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp import ObservationHistoryTermCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import AOWD_TERRAINS_CFG  # isort: skip


LEG_JOINT_NAMES = [".*HAA", ".*HFE", ".*KFE"]
LEG_BODY_NAMES = [".*HIP", ".*THIGH", ".*SHANK"]
WHEEL_JOINT_NAMES = [".*WHEEL"]
WHEEL_BODY_NAMES = [".*WHEEL_L"]

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=AOWD_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING

    # basic sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.3, 0.3]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
    )
    # normal scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(3.0, 2.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    path_command = mdp.PathCommandCfg(
        asset_name="robot",
        resampling_time_range=(1000000.0, 1000000.0),
        debug_vis=True,
        num_waypoints=10,
        path_config={
            "spline_angle_range": (0.0, 120.0),
            "rotate_angle_range": (0.0, 150.0),
            "pos_tolerance_range": (0.2, 0.2),
            "terrain_level_range": (0, 0),
            "resolution": [10.0, 10.0, 0.2, 1],
            "initial_params": [30.0, 40.0, 0.2, 0],
        },
        max_speed=1.0,
        rel_standing_envs=0.0,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # legs
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINT_NAMES, scale=0.5, use_default_offset=True
    )
    # wheels
    wheel_joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=WHEEL_JOINT_NAMES, scale=5.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        ###############################################
        # 1. proprioceptive group
        ###############################################
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 12
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))  # 16
        last_actions = ObsTerm(func=mdp.last_action)  # 16
        last_last_actions = ObsTerm(func=mdp.last_last_action)  # 16
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # 3
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # 3
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # 3
        """These two obs update at sim update period (200Hz). For performance reasons, they should be kept minimal."""
        history_joint_pos_error = ObservationHistoryTermCfg(
            func=mdp.ObservationHistory,
            params={"method": "joint_pos_error"},
            history_indices=[-1, -3, -5], 
            asset_cfg=SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES,
                                     joint_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), # have to manually specify joint_ids here
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 3*12
        history_joint_vel = ObservationHistoryTermCfg(
            func=mdp.ObservationHistory,
            params={"method": "joint_vel"},
            history_indices=[-3, -5], 
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )  # 2*16
        history_root_lin_vel = ObservationHistoryTermCfg(
            func=mdp.ObservationHistory,
            params={"method": "root_lin_vel_history"},
            history_indices=[-20, -30, -40],
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 3*3
        history_root_ang_vel = ObservationHistoryTermCfg(
            func=mdp.ObservationHistory,
            params={ "method": "root_ang_vel_history"},
            history_indices=[-20, -30, -40],
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 3*3
        history_root_pos = ObservationHistoryTermCfg(
            func=mdp.ObservationHistory,
            params={"method": "root_pos_history"},
            history_indices=[-20, -30, -40, -50, -60],
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 5*2

        ###############################################
        # 2. exteroceptive group
        ###############################################
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.65,
            },  # default 0.5 -> +0.15m for wheel radius
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )  # 30

        ###############################################
        # 3. tracking group
        ###############################################
        wp_interval = ObsTerm(func=mdp.waypoint_interval, noise=Unoise(n_min=-0.01, n_max=0.01))  # 1
        # pos_err_tolerance = ObsTerm(func=mdp.pos_err_tolerance) # 1
        pos_err_tolerance_vec = ObsTerm(
            func=mdp.pos_err_tolerance_one_hot_vec, params={"scale_factor": 10, "vector_len": 5}
        )  # 5
        waypoints_list_time = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "path_command"}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )  # num_waypoints*3
        waypoints_list_fixed = ObsTerm(
            func=mdp.waypoints_list_fixed, noise=Unoise(n_min=-0.01, n_max=0.01)
        )  # num_waypoints*3

        ###############################################
        # 3. privileged group
        ###############################################
        feet_contact_state = ObsTerm(
            func=mdp.feet_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES)},
        )  # 4
        feet_friction_static = ObsTerm(
            func=mdp.feet_friction,
            params={"mode": "static", "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES)},
        )  # 4
        feet_friction_dynamic = ObsTerm(
            func=mdp.feet_friction,
            params={"mode": "dynamic", "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES)},
        )  # 4
        ground_friction = ObsTerm(func=mdp.ground_friction)  # 2
        penalized_contact_state = ObsTerm(
            func=mdp.penalized_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=LEG_BODY_NAMES), "threshold": 1.0},
        )  # 12
        external_wrench = ObsTerm(
            func=mdp.body_incoming_wrench, params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])}
        )  # 6
        air_time = ObsTerm(
            func=mdp.feet_air_time, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES)}
        )  # 4
        mass_disturbance = ObsTerm(
            func=mdp.mass_disturbance, params={"asset_cfg": SceneEntityCfg("robot", body_names="base")}
        )  # 1
        push_velocity = ObsTerm(func=mdp.push_velocity)  # 6

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # share the same observation for actor and critic
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "mass_distribution_params": (-5.0, 5.0), "operation": "add"},
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-10, 10),  # N
            "torque_range": (-2.0, 2.0),  # Nm
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity_thr,
    #     mode="interval",
    #     interval_range_s=(2.0, 2.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "goal_distance_thresh": 0.1},
    # )

    # resample_wp_interval = EventTerm(
    #     func=mdp.resample_wp_interval,
    #     mode="interval",
    #     interval_range_s=(5.0, 15.0),
    #     params={"random_rotation": False},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    #### Tracking rewards ############################################################################
    track_pos_error = RewTerm(func=mdp.track_pos_xy_exp, weight=2.0, params={"use_tanh": False})
    track_speed_up = RewTerm(func=mdp.tracking_speed_up, weight=1.0, params={"goal_distance_thresh": 0.1, "std": 1.0})

    #### Behavior rewards ############################################################################

    # 1. On the way to the goal
    base_height = RewTerm(
        func=mdp.base_height_sensor,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
            "target_height": 0.55,
            "margin": 0.05,
            "higher_scale": 0.99,
        },
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)

    # 2. Near the goal
    goal_position = RewTerm(func=mdp.goal_position, weight=0.5, params={"goal_distance_thresh": 0.5})
    goal_orientation = RewTerm(func=mdp.goal_orientation, weight=0.0, params={"goal_distance_thresh": 0.2})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2_thr, weight=-1.0, params={"goal_distance_thresh": 0.2})

    # 3. At the goal
    default_pose = RewTerm(
        func=mdp.default_pose,
        weight=1.0,
        params={"goal_distance_thresh": 0.1, "goal_yaw_thresh": 100,
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    stand_still_normalization = RewTerm(
        func=mdp.stand_still_normalization, weight=-0.5,
        params={"goal_distance_thresh": 0.1, "goal_yaw_thresh": 100,}
    )

    #### Torque Penalties ############################################################################
    joint_torques_l2_legs = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_power_l1 = RewTerm(func=mdp.joint_power_l1, weight=-1.0e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES+WHEEL_JOINT_NAMES)})

    ### Velocity Penalties ###########################################################################
    joint_vel_l2_legs = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_vel_l2_wheels = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )

    ### Acceleration Penalties ########################################################################
    joint_acc_l2_legs = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_acc_l2_wheels = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-9,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )

    #### Action Rate Penalties ########################################################################
    action_rate_l2_legs = RewTerm(func=mdp.action_rate_l2, weight=-0.01, params={"action_name": "leg_joint_pos"})
    action_rate_l2_wheels = RewTerm(func=mdp.action_rate_l2, weight=-0.01, params={"action_name": "wheel_joint_vel"})

    #### Termination Penalty ##########################################################################
    episode_termination = RewTerm(
        func=mdp.is_terminated,
        weight=-500.0,  # Sparse Reward of {-20.0, 0.0} --> Max Episode Penalty: -20.0
    )

    #### Contact Penalties ############################################################################
    undesired_contacts_legs = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH", ".*SHANK"]), "threshold": 1.0},
    )
    joint_pos_limits_legs = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-100,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_vel_limits_legs = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES), "soft_ratio": 0.9},
    )
    joint_vel_limits_wheels = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES), "soft_ratio": 0.9},
    )
    joint_torque_limits_legs = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_torque_limits_wheels = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
    )

    #### Optional Rewards ############################################################################
    # body_stumble = RewTerm(
    #     func=mdp.body_stumble, weight= 2e-3,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES), "threshold": 2.0, 
    #             "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES) },
    # )
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_rw,
    #     weight=-1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES),
    #         "threshold": 0.2,
    #     },
    # )
    # flat_orientation_l2_slope = RewTerm(
    #     func=mdp.flat_orientation_l2_slope,
    #     weight=-1.0,
    # )
    # lateral_movement = RewTerm(
    #     func=mdp.lateral_movement,
    #     weight=-0.0,
    # )
    # backward_movement = RewTerm(
    #     func=mdp.backwards_movement,
    #     params={"threshold": -0.1}, # speed threshold
    #     weight=-5.0,
    # )
    # no_robot_movement = RewTerm(
    #     func=mdp.no_robot_movement,
    #     weight=-1.0,
    #     params={"goal_distance_thresh": 0.1},
    # )
    # near_goal_stability = RewTerm(
    #     func=mdp.near_goal_stability,
    #     weight=-1.0,
    #     params={"goal_distance_thresh": 0.1},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "extended_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*THIGH", ".*HIP", ".*SHANK"]),
            "threshold": 1.0,
            "extended_step": 1000 * 48,
        },
        time_out=False,
    )
    deviate_from_path = DoneTerm(
        func=mdp.robot_away_from_path,
        params={"alpha": 1.0},
        time_out=False,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    path_terrain_difficulty = CurrTerm(
        func=mdp.explore_config_space,
        params={
            "update_rate_steps": 2 * 48,
        },
    )

    show_distribution_speed = CurrTerm(
        func=mdp.show_distribution_speed,
    )

    show_distribution_terrain = CurrTerm(
        func=mdp.show_distribution_terrain,
    )

    show_position_error_tolerance = CurrTerm(
        func=mdp.show_position_error_tolerance,
    )

    show_path_progress = CurrTerm(
        func=mdp.show_path_progress,
    )

    set_termination_alpha = CurrTerm(
        func=mdp.set_termination_alpha,
        params={
            "term_name": "deviate_from_path",
            "initial_alpha": 0.0,
            "final_alpha": 1.0,
            "start_step": 1000 * 48,
            "end_step": 3000 * 48,
        },
    )

    # reward_base_height = CurrTerm(
    #     func=mdp.modify_reward_weight_linearly,
    #     params={"term_name": "base_height",
    #             "initial_weight": RewardsCfg().base_height.weight,
    #             "final_weight": RewardsCfg().base_height.weight * 2,
    #             "start_step": 500 * 48,
    #             "end_step": 1500 * 48}
    # )

    # push_velocity = CurrTerm(
    #     func = mdp.modify_push_velocity,
    #     params={"term_name": "push_robot",
    #             "initial_config": EventCfg().push_robot.params["velocity_range"],
    #             "final_config": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-0.5, 0.5)},
    #             "start_step": 1000 * 48,
    #             "end_step": 2000 * 48}
    # )


##
# Environment configuration
##
@configclass
class TeacherPathTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion path-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    # costs: CostsCfg = CostsCfg() # CMDP
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if hasattr(self.scene, "base_height_scanner") and self.scene.base_height_scanner is not None:
            self.scene.base_height_scanner.update_period = self.decimation * self.sim.dt
        if hasattr(self.scene, "height_scanner") and self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if hasattr(self.scene, "contact_forces") and self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "path_terrain_difficulty", None) is not None:
            if self.scene.terrain.terrain_generator is not None and self.commands.path_command.path_config[
                "terrain_level_range"
            ] != (0, 0):
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
