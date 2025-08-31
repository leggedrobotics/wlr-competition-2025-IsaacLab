# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

### Data Augmentation ####

# ORBIT: ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA', 'LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE', 'LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE', 'LF_WHEEL', 'LH_WHEEL', 'RF_WHEEL', 'RH_WHEEL']
IDX_LEGS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
IDX_WHEELS = [12, 13, 14, 15]


def get_symmetric_states(obs, actions, scan=False, **kwargs):
    obs_aug, actions_aug = _symmetry_observations(obs, actions, scan)
    return obs_aug, actions_aug


def get_symmetric_states_scan(obs, actions, scan=True, **kwargs):
    obs_aug, actions_aug = _symmetry_observations(obs, actions, scan)
    return obs_aug, actions_aug


def _symmetry_observations(obs, actions, scan):
    obs_aug, actions_aug = None, None
    num_symmetries = 4

    if obs is not None:
        num_envs = obs.shape[0]
        num_obs = obs.shape[1]
        obs_aug = torch.zeros(num_envs * num_symmetries, num_obs, device=obs.device)
        # -- original
        obs_aug[:num_envs] = obs[:]
        # -- left-right
        obs_aug[num_envs : 2 * num_envs] = _transform_obs_left_right(obs, scan)
        # -- front-back
        obs_aug[2 * num_envs : 3 * num_envs] = _transform_obs_front_back(obs, scan)
        # -- diagonal
        obs_aug[3 * num_envs :] = _transform_obs_front_back(obs_aug[num_envs : 2 * num_envs], scan)

    if actions is not None:
        num_envs = actions.shape[0]
        num_actions = actions.shape[1]
        actions_aug = torch.zeros(num_envs * num_symmetries, num_actions, device=actions.device)
        # -- original
        actions_aug[:num_envs] = actions[:]
        # -- left-right
        actions_aug[num_envs : 2 * num_envs] = _transform_actions_left_right(actions)
        # -- front-back
        actions_aug[2 * num_envs : 3 * num_envs] = _transform_actions_front_back(actions)
        # -- diagonal
        actions_aug[3 * num_envs :] = _transform_actions_front_back(actions_aug[num_envs : 2 * num_envs])

    return obs_aug, actions_aug


"""
JOINTS
"""


def _switch_joints_lr(dof, fixed_values=False):
    dof_switched = torch.zeros_like(dof, device=dof.device)
    assert dof.shape[-1] == 12 or dof.shape[-1] == 16, "Only 12, 16 DoF are supported"

    # legs
    dof_switched[..., IDX_LEGS] = _switch_legs_lr(dof[..., IDX_LEGS], fixed_values=fixed_values)

    # wheels
    if dof.shape[-1] == 16:
        dof_switched[..., IDX_WHEELS] = _switch_wheels_lr(dof[..., IDX_WHEELS], fixed_values=fixed_values)

    return dof_switched


def _switch_joints_fb(dof, fixed_values=False):
    dof_switched = torch.zeros_like(dof, device=dof.device)
    assert dof.shape[-1] == 12 or dof.shape[-1] == 16, "Only 12, 16 DoF are supported"

    # legs
    dof_switched[..., IDX_LEGS] = _switch_legs_fb(dof[..., IDX_LEGS], fixed_values=fixed_values)

    # wheels
    if dof.shape[-1] == 16:
        dof_switched[..., IDX_WHEELS] = _switch_wheels_fb(dof[..., IDX_WHEELS], fixed_values=fixed_values)

    return dof_switched


"""
LEGS
"""


def _switch_legs_lr(dof, fixed_values=False):
    dof_switched = torch.zeros_like(dof, device=dof.device)
    assert dof.shape[-1] == 12, "Only 12 DoF are supported"
    IDX_LEG_LF = [0, 4, 8]
    IDX_LEG_LH = [1, 5, 9]
    IDX_LEG_RF = [2, 6, 10]
    IDX_LEG_RH = [3, 7, 11]
    # front left <-> right
    dof_switched[..., IDX_LEG_LF] = dof[..., IDX_LEG_RF]
    dof_switched[..., IDX_LEG_RF] = dof[..., IDX_LEG_LF]
    # hind left <...> right
    dof_switched[..., IDX_LEG_LH] = dof[..., IDX_LEG_RH]
    dof_switched[..., IDX_LEG_RH] = dof[..., IDX_LEG_LH]

    if not fixed_values:
        dof_switched[..., IDX_LEG_LF] *= torch.tensor([-1, 1, 1], device=dof.device)
        dof_switched[..., IDX_LEG_RF] *= torch.tensor([-1, 1, 1], device=dof.device)
        dof_switched[..., IDX_LEG_LH] *= torch.tensor([-1, 1, 1], device=dof.device)
        dof_switched[..., IDX_LEG_RH] *= torch.tensor([-1, 1, 1], device=dof.device)

    return dof_switched


def _switch_legs_fb(dof, fixed_values=False):
    dof_switched = torch.zeros_like(dof, device=dof.device)
    assert dof.shape[-1] == 12, "Only 12 DoF are supported"
    IDX_LEG_LF = [0, 4, 8]
    IDX_LEG_LH = [1, 5, 9]
    IDX_LEG_RF = [2, 6, 10]
    IDX_LEG_RH = [3, 7, 11]
    # front left <-> hind left
    dof_switched[..., IDX_LEG_LF] = dof[..., IDX_LEG_LH]
    dof_switched[..., IDX_LEG_LH] = dof[..., IDX_LEG_LF]
    # front right <...> hind right
    dof_switched[..., IDX_LEG_RF] = dof[..., IDX_LEG_RH]
    dof_switched[..., IDX_LEG_RH] = dof[..., IDX_LEG_RF]

    if not fixed_values:
        dof_switched[..., IDX_LEG_LF] *= torch.tensor([1, -1, -1], device=dof.device)
        dof_switched[..., IDX_LEG_LH] *= torch.tensor([1, -1, -1], device=dof.device)
        dof_switched[..., IDX_LEG_RF] *= torch.tensor([1, -1, -1], device=dof.device)
        dof_switched[..., IDX_LEG_RH] *= torch.tensor([1, -1, -1], device=dof.device)

    return dof_switched


"""
WHEELS
"""


def _switch_wheels_lr(dof, fixed_values=False):
    dof_switched = torch.zeros_like(dof, device=dof.device)
    assert dof.shape[-1] == 4, "Only 4 wheels are supported"
    IDX_WHEEL_LF = 0
    IDX_WHEEL_LH = 1
    IDX_WHEEL_RF = 2
    IDX_WHEEL_RH = 3
    # front left <-> right
    dof_switched[..., IDX_WHEEL_LF] = dof[..., IDX_WHEEL_RF]
    dof_switched[..., IDX_WHEEL_RF] = dof[..., IDX_WHEEL_LF]
    # hind left <...> right
    dof_switched[..., IDX_WHEEL_LH] = dof[..., IDX_WHEEL_RH]
    dof_switched[..., IDX_WHEEL_RH] = dof[..., IDX_WHEEL_LH]

    if not fixed_values:
        dof_switched[..., IDX_WHEEL_LF] *= 1
        dof_switched[..., IDX_WHEEL_RF] *= 1
        dof_switched[..., IDX_WHEEL_LH] *= 1
        dof_switched[..., IDX_WHEEL_RH] *= 1

    return dof_switched


def _switch_wheels_fb(dof, fixed_values=False):
    dof_switched = torch.zeros_like(dof, device=dof.device)
    assert dof.shape[-1] == 4, "Only 4 wheels are supported"
    IDX_WHEEL_LF = 0
    IDX_WHEEL_LH = 1
    IDX_WHEEL_RF = 2
    IDX_WHEEL_RH = 3
    # front left <-> hind left
    dof_switched[..., IDX_WHEEL_LF] = dof[..., IDX_WHEEL_LH]
    dof_switched[..., IDX_WHEEL_LH] = dof[..., IDX_WHEEL_LF]
    # front right <...> hind right
    dof_switched[..., IDX_WHEEL_RF] = dof[..., IDX_WHEEL_RH]
    dof_switched[..., IDX_WHEEL_RH] = dof[..., IDX_WHEEL_RF]

    if not fixed_values:
        dof_switched[..., IDX_WHEEL_LF] *= -1
        dof_switched[..., IDX_WHEEL_LH] *= -1
        dof_switched[..., IDX_WHEEL_RF] *= -1
        dof_switched[..., IDX_WHEEL_RH] *= -1

    return dof_switched


"""
OBSERVATIONS
"""


def _transform_obs_left_right(obs, scan: bool):
    obs = obs.clone()
    device = obs.device
    num_obs = obs.shape[1]
    idx = 0

    ## Proprioceptive
    # joint pos rel
    dim = 12
    obs[:, idx : idx + dim] = _switch_joints_lr(obs[:, idx : idx + dim])
    idx += dim
    # joint vel rel
    dim = 16
    obs[:, idx : idx + dim] = _switch_joints_lr(obs[:, idx : idx + dim])
    idx += dim
    # last actions
    dim = 16
    obs[:, idx : idx + dim] = _transform_actions_left_right(obs[:, idx : idx + dim])
    idx += dim
    # last last actions
    dim = 16
    obs[:, idx : idx + dim] = _transform_actions_left_right(obs[:, idx : idx + dim])
    idx += dim
    # projected gravity
    dim = 3
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, 1], device=device)
    idx += dim
    # base lin vel
    dim = 3
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, 1], device=device)
    idx += dim
    # base ang vel
    dim = 3
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, -1], device=device)
    idx += dim
    # history joint position error 3 steps
    dim = 12
    obs[:, idx : idx + dim] = _switch_joints_lr(obs[:, idx : idx + dim])
    idx += dim
    obs[:, idx : idx + dim] = _switch_joints_lr(obs[:, idx : idx + dim])
    idx += dim
    obs[:, idx : idx + dim] = _switch_joints_lr(obs[:, idx : idx + dim])
    idx += dim
    # history joint vel rel 2 steps
    dim = 16
    obs[:, idx : idx + dim] = _switch_joints_lr(obs[:, idx : idx + dim])
    idx += dim
    obs[:, idx : idx + dim] = _switch_joints_lr(obs[:, idx : idx + dim])
    idx += dim
    # history base lin vel 3 steps
    dim = 3
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, 1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, 1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, 1], device=device)
    idx += dim
    # history base ang vel 3 steps
    dim = 3
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, -1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, -1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, -1], device=device)
    idx += dim
    # history base pos 2d, 5 steps
    dim = 2
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1], device=device)
    idx += dim

    ## Exteroception
    if scan:
        dim = 176
        obs[:, idx : idx + dim] = obs[:, idx : idx + dim].view(-1, 11, 16).flip(dims=[1]).view(-1, 11 * 16)
        idx += dim

    ## Privileged
    # feet contact state
    dim = 4
    obs[:, idx : idx + dim] = _switch_wheels_lr(obs[:, idx : idx + dim], fixed_values=True)
    idx += dim
    # feet friction static
    dim = 4
    obs[:, idx : idx + dim] = _switch_wheels_lr(obs[:, idx : idx + dim], fixed_values=True)
    idx += dim
    # feet friction dynamic
    dim = 4
    obs[:, idx : idx + dim] = _switch_wheels_lr(obs[:, idx : idx + dim], fixed_values=True)
    idx += dim
    # ground friction
    dim = 2
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim]
    idx += dim
    # penalized contact state
    dim = 12
    obs[:, idx : idx + dim] = _switch_joints_lr(obs[:, idx : idx + dim], fixed_values=True)
    idx += dim
    # external wrench
    dim = 6
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, 1, -1, 1, -1], device=device)
    idx += dim
    # air time
    dim = 4
    obs[:, idx : idx + dim] = _switch_wheels_lr(obs[:, idx : idx + dim], fixed_values=True)
    idx += dim
    # mass disturbance
    dim = 1
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim]
    idx += dim
    # push velocity
    dim = 6
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, 1, -1, 1, -1], device=device)
    idx += dim

    assert idx == num_obs, f"Expected {num_obs} but got {idx}"
    return obs


def _transform_obs_front_back(obs, scan: bool):
    obs = obs.clone()
    device = obs.device
    num_obs = obs.shape[1]
    idx = 0

    ## Proprioceptive
    # joint pos
    dim = 12
    obs[:, idx : idx + dim] = _switch_joints_fb(obs[:, idx : idx + dim])
    idx += dim
    # joint vel
    dim = 16
    obs[:, idx : idx + dim] = _switch_joints_fb(obs[:, idx : idx + dim])
    idx += dim
    # last actions
    dim = 16
    obs[:, idx : idx + dim] = _transform_actions_front_back(obs[:, idx : idx + dim])
    idx += dim
    # last last actions
    dim = 16
    obs[:, idx : idx + dim] = _transform_actions_front_back(obs[:, idx : idx + dim])
    idx += dim
    # projected gravity
    dim = 3
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, 1], device=device)
    idx += dim
    # base lin vel
    dim = 3
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, 1], device=device)
    idx += dim
    # base ang vel
    dim = 3
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, -1], device=device)
    idx += dim
    # history joint position error 3 steps
    dim = 12
    obs[:, idx : idx + dim] = _switch_joints_fb(obs[:, idx : idx + dim])
    idx += dim
    obs[:, idx : idx + dim] = _switch_joints_fb(obs[:, idx : idx + dim])
    idx += dim
    obs[:, idx : idx + dim] = _switch_joints_fb(obs[:, idx : idx + dim])
    idx += dim
    # history joint velocities 2 steps
    dim = 16
    obs[:, idx : idx + dim] = _switch_joints_fb(obs[:, idx : idx + dim])
    idx += dim
    obs[:, idx : idx + dim] = _switch_joints_fb(obs[:, idx : idx + dim])
    idx += dim
    # history base line vel 3 steps
    dim = 3
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, 1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, 1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, 1], device=device)
    idx += dim
    # history base ang vel 3 steps
    dim = 3
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, -1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, -1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([1, -1, -1], device=device)
    idx += dim
    # history base pos 2d, 5 steps
    dim = 2
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1], device=device)
    idx += dim
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1], device=device)
    idx += dim

    ## Exteroception
    if scan:
        dim = 176
        obs[:, idx : idx + dim] = obs[:, idx : idx + dim].view(-1, 11, 16).flip(dims=[2]).view(-1, 11 * 16)
        idx += dim

    ## Privileged
    # feet contact state
    dim = 4
    obs[:, idx : idx + dim] = _switch_wheels_fb(obs[:, idx : idx + dim], fixed_values=True)
    idx += dim
    # feet friction static
    dim = 4
    obs[:, idx : idx + dim] = _switch_wheels_fb(obs[:, idx : idx + dim], fixed_values=True)
    idx += dim
    # feet friction dynamic
    dim = 4
    obs[:, idx : idx + dim] = _switch_wheels_fb(obs[:, idx : idx + dim], fixed_values=True)
    idx += dim
    # ground friction
    dim = 2
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim]
    idx += dim
    # penalized contact state
    dim = 12
    obs[:, idx : idx + dim] = _switch_joints_fb(obs[:, idx : idx + dim], fixed_values=True)
    idx += dim
    # external wrench
    dim = 6
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, 1, 1, -1, -1], device=device)
    idx += dim
    # air time
    dim = 4
    obs[:, idx : idx + dim] = _switch_wheels_fb(obs[:, idx : idx + dim], fixed_values=True)
    idx += dim
    # mass disturbance
    dim = 1
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim]
    idx += dim
    # push velocity
    dim = 6
    obs[:, idx : idx + dim] = obs[:, idx : idx + dim] * torch.tensor([-1, 1, 1, 1, -1, -1], device=device)
    idx += dim

    assert idx == num_obs, f"Expected {num_obs} but got {idx}"
    return obs


"""
ACTIONS
"""


def _transform_actions_left_right(actions):
    actions = actions.clone()
    # legs
    actions[..., 0:12] = _switch_legs_lr(actions[..., 0:12])
    # wheels
    actions[..., 12:16] = _switch_wheels_lr(actions[..., 12:16])

    return actions


def _transform_actions_front_back(actions):
    actions = actions.clone()
    # legs
    actions[..., 0:12] = _switch_legs_fb(actions[..., 0:12])
    # wheels
    actions[..., 12:16] = _switch_wheels_fb(actions[..., 12:16])

    return actions
