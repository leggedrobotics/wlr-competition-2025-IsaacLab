# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp import ObservationHistoryTermCfg, ObservationHistory
from isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp.observations import (
    feet_contact_state, feet_friction, ground_friction, last_last_action, feet_air_time,
    mass_disturbance, push_velocity, penalized_contacts, body_incoming_wrench
)

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

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.5,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-4.0, 4.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-3.0, 3.0), heading=(-math.pi, math.pi)
        ),
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
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 12
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))  # 16
        last_actions = ObsTerm(func=mdp.last_action)  # 16
        last_last_actions = ObsTerm(func=last_last_action)  # 16
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))  # 3
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # 3
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))  # 3

        ###############################################
        # 2. history group
        ###############################################
        history_joint_pos_error = ObservationHistoryTermCfg(
            func=ObservationHistory,
            params={"method": "joint_pos_error"},
            history_indices=[-1, -3, -5], 
            asset_cfg=SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES,
                                     joint_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), # have to manually specify joint_ids here
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 3*12
        history_joint_vel = ObservationHistoryTermCfg(
            func=ObservationHistory,
            params={"method": "joint_vel"},
            history_indices=[-3, -5], 
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )  # 2*16
        history_root_lin_vel = ObservationHistoryTermCfg(
            func=ObservationHistory,
            params={"method": "root_lin_vel_history"},
            history_indices=[-20, -30, -40],
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 3*3
        history_root_ang_vel = ObservationHistoryTermCfg(
            func=ObservationHistory,
            params={ "method": "root_ang_vel_history"},
            history_indices=[-20, -30, -40],
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 3*3
        history_root_pos = ObservationHistoryTermCfg(
            func=ObservationHistory,
            params={"method": "root_pos_history"},
            history_indices=[-20, -30, -40, -50, -60],
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )  # 5*2

        ###############################################
        # 3. exteroceptive group
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
        # 4. privileged group
        ###############################################
        feet_contact_state = ObsTerm(
            func=feet_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES)},
        )  # 4
        feet_friction_static = ObsTerm(
            func=feet_friction,
            params={"mode": "static", "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES)},
        )  # 4
        feet_friction_dynamic = ObsTerm(
            func=feet_friction,
            params={"mode": "dynamic", "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES)},
        )  # 4
        ground_friction = ObsTerm(func=ground_friction)  # 2
        penalized_contact_state = ObsTerm(
            func=penalized_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=LEG_BODY_NAMES), "threshold": 1.0},
        )  # 12
        external_wrench = ObsTerm(
            func=body_incoming_wrench, params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])}
        )  # 6
        air_time = ObsTerm(
            func=feet_air_time, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=WHEEL_BODY_NAMES)}
        )  # 4
        mass_disturbance = ObsTerm(
            func=mass_disturbance, params={"asset_cfg": SceneEntityCfg("robot", body_names="base")}
        )  # 1
        push_velocity = ObsTerm(func=push_velocity)  # 6

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
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
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-2.0, 2.0),
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
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
            "target_height": 0.55,
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.4,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    episode_termination = RewTerm(
        func=mdp.is_terminated,
        weight=-500.0,  # Sparse Reward of {-20.0, 0.0} --> Max Episode Penalty: -20.0
    )


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


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
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
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
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
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
