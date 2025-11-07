from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions import JointAction


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import actions_cfg

from isaaclab.utils import configclass    

class DeltaJointVelocityAction(JointAction):
    """Joint action term for delta joint velocity using one-step integration (OI∆)."""
    
    cfg: actions_cfg.DeltaJointVelocityActionCfg
    
    def __init__(self, cfg: actions_cfg.DeltaJointVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.c_scale = cfg.c_scale
        self._dt = env.physics_dt * env.cfg.decimation
        
        print(f"[INFO] DeltaJointVelocityAction initialized:")
        print(f"  Num envs: {self.num_envs}")
        print(f"  c_scale: {self.c_scale}")
        print(f"  control dt: {self._dt}")
        print(f"  Joint names: {self._joint_names}")
    
    def process_actions(self, actions: torch.Tensor):
        # Store raw actions
        self._raw_actions[:] = actions
        
        # Apply standard scaling and offset (but not clipping yet)
        scaled_actions = self._raw_actions * self._scale + self._offset
        
        # Get current joint velocities (feedback 'v')
        current_vel = self._asset.data.joint_vel[:, self._joint_ids]
        
        # Apply OI∆ formula: v_d ← v + c · a · dt
        target_velocity = current_vel + self.c_scale * scaled_actions * self._dt
        
        # Clip to velocity limits if specified
        if self.cfg.clip is not None:
            target_velocity = torch.clamp(
                target_velocity, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        
        # Store as processed actions
        self._processed_actions[:] = target_velocity
    
    def apply_actions(self):
        # Set velocity targets
        self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Reset raw actions properly
        if env_ids is None:
            self._raw_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            
        
# class DeltaJointVelocityAction(JointAction):
#     """Joint action term for delta joint velocity using one-step integration (OI∆)."""
    
#     cfg: actions_cfg.DeltaJointVelocityActionCfg
    
#     def __init__(self, cfg: actions_cfg.DeltaJointVelocityActionCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)
        
#         self.c_scale = cfg.c_scale # c (scaling constant)
#         self._dt = env.physics_dt * env.cfg.decimation
        
#         # Initialize desired joint positions
#         self._desired_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        
#         print(f"[INFO] DeltaJointVelocityAction initialized:")
#         print(f"  Num envs: {self.num_envs}")
#         print(f"  c_scale: {self.c_scale}")
#         print(f"  control dt: {self._dt}")
#         print(f"  Joint names: {self._joint_names}")
    
#     def process_actions(self, actions: torch.Tensor):
#         # Apply base processing (scale, offset, clip)
#         super().process_actions(actions)
        
#         # Get current joint velocities v (current feedback)
#         current_vel = self._asset.data.joint_vel[:, self._joint_ids]
        
#         # Apply delta integration (v_d ← v + c · a · dt)
#         # self.processed_actions = a (dt from action scaling)
#         target_velocity = current_vel + self.c_scale * self.processed_actions
        
#         # Update processed_actions to be the target velocity
#         self._processed_actions = target_velocity
        
#         # Integrate velocity to position
#         # Maximum velocity change per step = c * dt * |a_max| where |a_max| = 1.0
#         self._desired_joint_pos += target_velocity * self._dt
    
#     def apply_actions(self):
#         # Apply both position and velocity targets
#         self._asset.set_joint_position_target(self._desired_joint_pos, joint_ids=self._joint_ids)
#         self._asset.set_joint_velocity_target(self.processed_actions, joint_ids=self._joint_ids)
    
#     def reset(self, env_ids: Sequence[int] | None = None) -> None:
#         # Call parent reset
#         super().reset(env_ids)
        
#         # Reset desired positions to current positions
#         if env_ids is None:
#             self._desired_joint_pos[:] = self._asset.data.joint_pos[:, self._joint_ids]
#         else:
#             # Index environments first, then joints
#             self._desired_joint_pos[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]