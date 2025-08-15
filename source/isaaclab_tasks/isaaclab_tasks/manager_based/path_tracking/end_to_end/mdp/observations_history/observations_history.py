from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.managers import ObservationTermCfg

from isaaclab.envs.mdp import *
from .observations_history_cfg import ObservationHistoryTermCfg
from ..observations import joint_pos_error



class ObservationHistory(ManagerTermBase):

    def __init__(self, cfg: ObservationHistoryTermCfg, env: ManagerBasedRLEnv):
        """Initialize the observation history term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # call the base class constructor
        super().__init__(cfg, env)

        self._cfg = cfg
        self._env = env
        
        """Initialize the buffers for the history of observations.
        NOTE: the histroy includes past information excluding the latest one!
        NOTE: The buffer is updated with step frequency (e.g. 50Hz) - only history group is updated with 200Hz!
        """
        self.buffers = {}  # buffer container data and last update step

    def __call__(self, *args, method = None) -> torch.Any:

        """Dynamically calls the requested method with flexible arguments.

        Args:
            method (str): The name of the method to call.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            Any: The result of the method call.
        """
        if method is None:
            raise ValueError("Method name must be specified using the 'method' keyword argument.")

        method_ref = getattr(self, method, None)  # Get method reference

        if method_ref is None or not callable(method_ref):
            raise ValueError(f"Method '{method}' not found in the class.")
        
        return method_ref(*args, self._cfg.asset_cfg, self._cfg.history_indices)

    def _reset(self, env: ManagerBasedRLEnv, buffer_names: list = None):
        """Reset the buffers for terminated episodes.

        Args:
            env: The environment object.
        """
        # Initialize & find terminated episodes
        try:
            terminated_mask = env.termination_manager.dones
        except AttributeError:
            terminated_mask = torch.zeros((env.num_envs), dtype=int).to(env.device)
        for key in buffer_names:
            # Initialize buffer if empty
            if self.buffers[key]["data"].shape[0] == 0:
                self.buffers[key]["data"] = torch.zeros((env.num_envs, *list(self.buffers[key].shape[1:]))).to(
                    env.device
                )
            # Reset buffer for terminated episodes
            self.buffers[key]["data"][terminated_mask, :, :] = 0.0

    def _process(
        self, env: ManagerBasedEnv, buffer_name: str, history_indices: list[int], data: torch.Tensor, asset_cfg: SceneEntityCfg
    ):
        """Update the bufers and return new buffer.
        Args:
            env: The environment object.
            buffer_name: The name of the buffer.
            history_indices: The history indices. -1 is the most recent entry in the history.
            data: The data to be stored in the buffer.
        """
        # Extract the history of the data
        history_idx = torch.tensor(history_indices).to(env.device) - 1  # history_idx-1 to not include the current step

        # Extract env step
        try:
            env_step = env.common_step_counter
        except AttributeError:
            env_step = 0

        # Add new buffer if fist call
        if buffer_name not in self.buffers:
            buffer_length = max(abs(index) for index in history_indices) + 2  # +1 for the current step
            self.buffers[buffer_name] = {}
            self.buffers[buffer_name]["data"] = torch.zeros((env.num_envs, buffer_length, data.shape[1])).to(env.device)
            self.buffers[buffer_name]["last_update_step"] = env_step

        # Check if buffer is already updated
        if not self.buffers[buffer_name]["last_update_step"] == env_step:
            # Reset buffer for terminated episodes
            self._reset(env, [buffer_name])
            # Updates buffer
            self.buffers[buffer_name]["data"] = self.buffers[buffer_name]["data"].roll(shifts=-1, dims=1)
            self.buffers[buffer_name]["data"][:, -1, :] = data
            self.buffers[buffer_name]["last_update_step"] = env_step

        # Extract the history of the data
        obs_history = self.buffers[buffer_name]["data"][:, history_idx, :]
        return obs_history[:, :, asset_cfg.joint_ids].reshape(env.num_envs, -1)

    def joint_pos(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]
    ):
        """Get the history of joint positions.

        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "joint_pos"
        data = joint_pos(env, SceneEntityCfg("robot"))
        return self._process(env, name, history_indices, data, asset_cfg)

    def joint_vel(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]
    ):
        """Get the history of joint velocites.

        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "joint_vel"
        data = joint_vel_rel(env, SceneEntityCfg("robot"))
        return self._process(env, name, history_indices, data, asset_cfg)

    def joint_pos_error(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]
    ):
        """Get the history of joint position errors.

        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        # extract the used quantities (to enable type-hinting)
        name = "joint_pos_error"
        data = joint_pos_error(env, SceneEntityCfg("robot"))
        return self._process(env, name, history_indices, data, asset_cfg)

    def root_pos_history(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]
    ):
        """Get the history of root positions 2d.

        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        name = "root_pos"
        asset: Articulation = env.scene[asset_cfg.name]
        data = asset.data.root_pos_w[:, :2]
        return self._process(env, name, history_indices, data, asset_cfg)

    def root_lin_vel_history(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]
    ):
        """Get the history of root linear velocity.

        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        name = "root_lin_vel"
        asset: Articulation = env.scene[asset_cfg.name]
        data = asset.data.root_lin_vel_b
        return self._process(env, name, history_indices, data, asset_cfg)
        
    def root_ang_vel_history(
        self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), history_indices: list = [-1]
    ):
        """Get the history of root angular velocity.

        Args:
            history_indices: The history indices. -1 is the most recent entry in the history.
        """
        name = "root_quat"
        asset: Articulation = env.scene[asset_cfg.name]
        data = asset.data.root_ang_vel_b
        return self._process(env, name, history_indices, data, asset_cfg)