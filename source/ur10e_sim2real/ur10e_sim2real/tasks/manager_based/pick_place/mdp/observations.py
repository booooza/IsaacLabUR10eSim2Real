"""Observation functions for pick-and-place task."""

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def distance_to_object(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Compute Euclidean distance from end-effector to object.
    
    Args:
        env: The environment instance.
        object_cfg: Configuration for the object entity.
        ee_frame_cfg: Configuration for the end-effector frame.
    
    Returns:
        Distance tensor shaped (num_envs, 1).
    """
    object_entity = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    object_pos = object_entity.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    
    distance = torch.norm(object_pos - ee_pos, p=2, dim=-1, keepdim=True)
    
    return distance

def ee_to_object_vector(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Vector from end-effector to object (relative position).
    
    Args:
        env: The environment instance.
        object_cfg: Configuration for the object entity.
        ee_frame_cfg: Configuration for the end-effector frame.
    
    Returns:
        Relative position vector shaped (num_envs, 3).
    """
    object_entity = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    object_pos = object_entity.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    
    return object_pos - ee_pos


def object_to_target_vector(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Vector from object to target location (relative position).
    
    Args:
        env: The environment instance.
        object_cfg: Configuration for the object entity.
        target_cfg: Configuration for the target entity.
    
    Returns:
        Relative position vector shaped (num_envs, 3).
    """
    object_entity = env.scene[object_cfg.name]
    target_entity = env.scene[target_cfg.name]
    
    object_pos = object_entity.data.root_pos_w[:, :3]
    target_pos = target_entity.data.root_pos_w[:, :3]
    
    return target_pos - object_pos

def ee_pose(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector pose (position + orientation quaternion).
    
    Args:
        env: The environment instance.
        ee_frame_cfg: Configuration for the end-effector frame.
    
    Returns:
        End-effector pose tensor shaped (num_envs, 7).
        First 3 elements are position [x, y, z].
        Last 4 elements are orientation quaternion [w, x, y, z].
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get position and orientation
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Concatenate [pos, quat]
    ee_pose_full = torch.cat([ee_pos, ee_quat], dim=-1)
    
    return ee_pose_full


def ee_position(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position only.
    
    Args:
        env: The environment instance.
        ee_frame_cfg: Configuration for the end-effector frame.
    
    Returns:
        End-effector position tensor shaped (num_envs, 3).
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :3]


def ee_velocity(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector linear and angular velocity.
    
    Args:
        env: The environment instance.
        ee_frame_cfg: Configuration for the end-effector frame.
    
    Returns:
        End-effector velocity tensor shaped (num_envs, 6).
        First 3 elements are linear velocity.
        Last 3 elements are angular velocity.
    """
    # FrameTransformer doesn't provide velocity - return zeros as placeholder
    num_envs = env.num_envs
    return torch.zeros(num_envs, 3, device=env.device)


def ee_linear_velocity(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector linear velocity only.
    
    Args:
        env: The environment instance.
        ee_frame_cfg: Configuration for the end-effector frame.
    
    Returns:
        End-effector linear velocity tensor shaped (num_envs, 3).
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_vel_w[..., 0, :3]


def ee_to_object_relative_pose(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Complete relative pose: position + orientation difference.
    
    Returns 7D vector: [dx, dy, dz, qw, qx, qy, qz]
    where position is object relative to EE, and orientation is relative rotation.
    
    Args:
        env: The environment instance.
        object_cfg: Configuration for the object entity.
        ee_frame_cfg: Configuration for the end-effector frame.
    
    Returns:
        Relative pose tensor shaped (num_envs, 7).
    """
    from isaaclab.utils.math import quat_mul, quat_inv
    
    object_entity = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Relative position
    object_pos = object_entity.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    rel_pos = object_pos - ee_pos
    
    # Relative orientation
    object_quat = object_entity.data.root_quat_w
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    # Compute: object_quat * inv(ee_quat)
    rel_quat = quat_mul(object_quat, quat_inv(ee_quat))
    
    return torch.cat([rel_pos, rel_quat], dim=-1)
