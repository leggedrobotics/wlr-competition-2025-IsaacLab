# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import NoiseCfg
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg, ObservationTermCfg
from isaaclab.managers import SceneEntityCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.managers.action_manager import ActionTerm
    from isaaclab.managers.command_manager import CommandTerm
    from isaaclab.managers.manager_base import ManagerTermBase
    from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg

@configclass
class ObservationHistoryTermCfg(ObservationTermCfg):
    """Configuration for an observation term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called."""

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """The asset configuration object."""

    history_indices: list = [-1]
    """The indices of the history to be considered."""

    action_terms: list = []
    """The action terms to be considered."""

    is_low_level: bool = False
    """Whether the observation is from a low-level action term."""