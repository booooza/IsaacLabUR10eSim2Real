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

def relative_pose_from_scene_entity(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_index: int = 0,
) -> torch.Tensor:
    """Get relative pose [pos, quat] from a FrameTransformer.
    
    The FrameTransformer computes poses of target frames relative to its source frame.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the FrameTransformer sensor.
        target_index: Index of the target frame to retrieve. Defaults to 0.
    
    Returns:
        Relative pose tensor shaped (num_envs, 7): [x, y, z, qw, qx, qy, qz]
    """
    frame_transformer = env.scene[asset_cfg.name]
    
    # FrameTransformer provides relative transforms from source to targets
    rel_pos = frame_transformer.data.target_pos_source[..., target_index, :]  # (num_envs, 3)
    rel_quat = frame_transformer.data.target_quat_source[..., target_index, :]  # (num_envs, 4)
    
    return torch.cat([rel_pos, rel_quat], dim=-1)  # (num_envs, 7)

def relative_position_from_scene_entity(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_index: int = 0,
) -> torch.Tensor:
    """Get relative position from a FrameTransformer.
    
    The FrameTransformer computes poses of target frames relative to its source frame.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the FrameTransformer sensor.
        target_index: Index of the target frame to retrieve. Defaults to 0.
    
    Returns:
        Relative position tensor shaped (num_envs, 3): [x, y, z]
    """
    frame_transformer = env.scene[asset_cfg.name]
    
    return frame_transformer.data.target_pos_source[..., target_index, :]  # (num_envs, 3)

def relative_rotation_from_scene_entity(
            env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_index: int = 0,
) -> torch.Tensor:
    """Get relative rotation (quaternion) from a FrameTransformer.
    
    The FrameTransformer computes poses of target frames relative to its source frame.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the FrameTransformer sensor.
        target_index: Index of the target frame to retrieve. Defaults to 0.
    
    Returns:
        Relative pose tensor shaped (num_envs, 4): [qw, qx, qy, qz]
    """
    frame_transformer = env.scene[asset_cfg.name]
    
    return frame_transformer.data.target_quat_source[..., target_index, :]  # (num_envs, 4)
