# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from isaaclab_tasks.manager_based.path_tracking.end_to_end.mdp.commands.path_command import PathCommand


"""
MDP terminations.
"""

def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Args:
        quat (torch.Tensor): Shape (N, 4) quaternion in (w, x, y, z) format.

    Returns:
        torch.Tensor: Shape (N, 3, 3) rotation matrices.
    """
    assert quat.shape[-1] == 4, "Quaternion must have shape (*, 4) in (w, x, y, z) format."

    w, x, y, z = quat.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Convert list elements into tensors before stacking
    mat = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    ], dim=-2)  # Change stacking axis to ensure correct shape

    return mat.view(*quat.shape[:-1], 3, 3)  # Shape: (..., 3, 3)


def robot_away_from_path(env: ManagerBasedRLEnv, command_name: str = "path_command", alpha: float = 1.0) -> torch.Tensor:
    """Check if object has gone far from the robot.

    The object is considered to be out-of-reach if the distance between the robot and the path is greater
    than the threshold.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot and the closest point in the path.
    """
    # extract useful elements
    path_cmd_generator: PathCommand = env.command_manager.get_term(command_name)
    pos_2d_avg = path_cmd_generator.metrics["position_error_2d"]  # Shape: (num_envs,)
    pos_2d_curr = path_cmd_generator.curr_pos_error  # Shape: (num_envs,)
    pos_2d = pos_2d_curr * alpha + pos_2d_avg * (1.0 - alpha)
    threshold = path_cmd_generator.pos_err_tolerance
    return pos_2d > threshold
