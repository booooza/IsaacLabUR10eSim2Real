"""Observation functions for pick-and-place task."""

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import  quat_apply_inverse


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

def manipulability_index(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute Yoshikawa manipulability index.
    
    μ = sqrt(det(J @ J^T))
    
    Higher values = further from singularities
    Typical range: 0.001 (near singular) to 0.5 (good configuration)
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot.
    
    Returns:
        Manipulability index shaped (num_envs, 1).
    """
    robot = env.scene[asset_cfg.name]
    
    # Get end-effector Jacobian
    ee_link_name = "robotiq_hande_end"
    ee_jacobi_idx = robot.find_bodies(ee_link_name)[0][0]  # First body ID
    
    # Get Jacobian: shape (num_envs, 6, num_joints)
    jacobian = robot.root_physx_view.get_jacobians()[
        :, ee_jacobi_idx, :, asset_cfg.joint_ids
    ]
    
    # Compute manipulability: μ = sqrt(det(J @ J^T))
    # J @ J^T has shape (num_envs, 6, 6)
    JJT = torch.bmm(jacobian, jacobian.transpose(-2, -1))
    
    # Determinant of 6x6 matrices
    det = torch.det(JJT)
    
    # Clamp to avoid sqrt of negative (numerical errors)
    det = torch.clamp(det, min=1e-12)
    
    manipulability = torch.sqrt(det)
    
    return manipulability.unsqueeze(-1)

def contact_forces_b(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, 3 * num_sensors)`` with forces stacked horizontally as
        ``[fx, fy, fz]`` per sensor.
    """
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b
