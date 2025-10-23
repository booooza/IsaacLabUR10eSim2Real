"""Reward functions for pick-and-place task."""

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import quat_mul, quat_conjugate

POSITION_SUCCESS_THRESHOLD = 0.05 # 5 cm
ROTATION_SUCCESS_THRESHOLD = 0.2 # ~11.5 degrees

def distance_l2(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Euclidean distance between two frame transformers.
    
    Computes ||source_pos - target_pos||_2
    
    Args:
        env: The environment instance.
        source_frame_cfg: Configuration for the source frame (e.g., end-effector).
        target_frame_cfg: Configuration for the target frame (e.g., hover target).
    
    Returns:
        Distance tensor shaped (num_envs,).
    """
    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]
    
    source_pos = source_frame.data.target_pos_w[..., 0, :3]
    target_pos = target_frame.data.target_pos_w[..., 0, :3]
    
    distance = torch.norm(target_pos - source_pos, p=2, dim=-1)
    
    return distance

def distance_exponential(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Exponential distance reward: exp(-d²/2sigma²)"""
    distance = distance_l2(env, source_frame_cfg, target_frame_cfg)
    reward = torch.exp(-distance**2 / (2 * sigma**2))
    return reward

def orientation_alignment_l2(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    rot_eps: float = 0.1,
) -> torch.Tensor:
    """Orientation alignment reward between two frame transformers.
    
    Computes: r_rot = 1 / (|rot_dist| + rot_eps)
    where rot_dist = 2 * arcsin(min(||quat_diff[:,1:4]||_2, 1.0))
    
    This encourages the source frame to align with the target frame orientation.
    
    Args:
        env: The environment instance.
        source_frame_cfg: Configuration for the source frame (e.g., end-effector).
        target_frame_cfg: Configuration for the target frame (e.g., hover target).
        rot_eps: Small epsilon to avoid division by zero.
    
    Returns:
        Reward tensor shaped (num_envs,).
    """
    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]
    
    source_quat = source_frame.data.target_quat_w[..., 0, :]  # (N, 4) [w, x, y, z]
    target_quat = target_frame.data.target_quat_w[..., 0, :]
    
    # Compute quaternion difference: q_diff = q_target * q_source^-1
    quat_diff = quat_mul(target_quat, quat_conjugate(source_quat))
    
    # Rotation distance: rot_dist = 2 * arcsin(min(||quat_diff[:,1:4]||_2, 1.0))
    # quat_diff is [w, x, y, z], so [:, 1:4] gives [x, y, z]
    xyz_norm = torch.norm(quat_diff[:, 1:4], p=2, dim=-1)
    xyz_norm_clamped = torch.clamp(xyz_norm, max=1.0)
    rot_dist = 2.0 * torch.asin(xyz_norm_clamped)
    
    # Reward: r_rot = 1 / (|rot_dist| + rot_eps)
    reward = 1.0 / (torch.abs(rot_dist) + rot_eps)
    
    return reward

def reach_goal_bonus(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    position_threshold: float | None = POSITION_SUCCESS_THRESHOLD,
    rotation_threshold: float | None = ROTATION_SUCCESS_THRESHOLD,
) -> torch.Tensor:
    """Sparse bonus when source frame reaches target frame with correct pose.
    
    Returns 1.0 when both position and orientation are within thresholds.
    
    Args:
        env: The environment instance.
        source_frame_cfg: Configuration for the source frame (e.g., end-effector).
        target_frame_cfg: Configuration for the target frame (e.g., hover target).
        position_threshold: Distance threshold for success (meters). None to ignore position check.
        rotation_threshold: Angular threshold for success (radians). None to ignore rotation check.
    
    Returns:
        Binary success tensor shaped (num_envs,).
    """
    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]
    
    # Start with all True
    device = source_frame.data.target_pos_w.device
    num_envs = source_frame.data.target_pos_w.shape[0]
    success = torch.ones(num_envs, dtype=torch.bool, device=device)
    
    # Position check (if threshold provided)
    if position_threshold is not None:
        source_pos = source_frame.data.target_pos_w[..., 0, :3]
        target_pos = target_frame.data.target_pos_w[..., 0, :3]
        position_error = torch.norm(target_pos - source_pos, p=2, dim=-1)
        success = success & (position_error < position_threshold)
    
    # Orientation check (if threshold provided)
    if rotation_threshold is not None:
        source_quat = source_frame.data.target_quat_w[..., 0, :]
        target_quat = target_frame.data.target_quat_w[..., 0, :]
        
        quat_diff = quat_mul(target_quat, quat_conjugate(source_quat))
        xyz_norm = torch.norm(quat_diff[:, 1:4], p=2, dim=-1)
        xyz_norm_clamped = torch.clamp(xyz_norm, max=1.0)
        rot_dist = 2.0 * torch.asin(xyz_norm_clamped)
        
        success = success & (rot_dist < rotation_threshold)
    
    return success.float()

def manipulability_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.01,
) -> torch.Tensor:
    """Penalty for approaching manipulability limits.
    
    Penalizes when manipulability drops below threshold.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot.
        threshold: Manipulability threshold below which to penalize.
    
    Returns:
        Penalty tensor shaped (num_envs,).
    """
    from .observations import manipulability_index
    
    manip = manipulability_index(env, asset_cfg).squeeze(-1)
    
    # Penalty only when below threshold
    # Using quadratic: (threshold - μ)² when μ < threshold
    penalty = torch.clamp(threshold - manip, min=0.0)
    penalty = penalty ** 2  # Quadratic makes it smooth
    
    return penalty


## Additional reward functions for pick-and-place task
def distance_reward_ee_to_object(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reward based on distance from end-effector to object.
    
    Args:
        env: The environment instance.
        std: Standard deviation for exponential kernel.
        object_cfg: Configuration for the object entity.
        ee_frame_cfg: Configuration for the end-effector frame.
    
    Returns:
        Reward tensor shaped (num_envs,).
    """
    object = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get positions
    object_pos = object.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    
    # Compute distance
    distance = torch.norm(object_pos - ee_pos, p=2, dim=-1)
    
    # Exponential kernel for smooth reward
    reward = torch.exp(-distance / std)
    
    return reward


def distance_reward_object_to_target(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Dense reward based on distance from object to target.
    
    Args:
        env: The environment instance.
        std: Standard deviation for exponential kernel.
        object_cfg: Configuration for the object entity.
        target_cfg: Configuration for the target entity.
    
    Returns:
        Reward tensor shaped (num_envs,).
    """
    object = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    
    # Get positions
    object_pos = object.data.root_pos_w[:, :3]
    target_pos = target.data.root_pos_w[:, :3]
    
    # Compute distance
    distance = torch.norm(object_pos - target_pos, p=2, dim=-1)
    
    # Exponential kernel
    reward = torch.exp(-distance / std)
    
    return reward


def grasp_success_reward(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Sparse reward for successful grasp (proximity to object).
    
    Args:
        env: The environment instance.
        threshold: Distance threshold for successful grasp.
        object_cfg: Configuration for the object entity.
        ee_frame_cfg: Configuration for the end-effector frame.
    
    Returns:
        Reward tensor shaped (num_envs,).
    """
    object = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get positions
    object_pos = object.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    
    # Check if close enough for grasp
    distance = torch.norm(object_pos - ee_pos, p=2, dim=-1)
    success = (distance < threshold).float()
    
    return success


def pick_success_reward(
    env: "ManagerBasedRLEnv",
    height_threshold: float = 0.15,
    grasp_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Sparse reward for successful pick (object lifted while grasped).
    
    Args:
        env: The environment instance.
        height_threshold: Minimum height for successful pick.
        grasp_threshold: Distance threshold for maintaining grasp.
        object_cfg: Configuration for the object entity.
        ee_frame_cfg: Configuration for the end-effector frame.
    
    Returns:
        Reward tensor shaped (num_envs,).
    """
    object = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Check height
    object_height = object.data.root_pos_w[:, 2]
    lifted = object_height > height_threshold
    
    # Check if still grasped
    object_pos = object.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    distance = torch.norm(object_pos - ee_pos, p=2, dim=-1)
    grasped = distance < grasp_threshold
    
    # Both conditions must be met
    success = (lifted & grasped).float()
    
    return success


def place_success_reward(
    env: "ManagerBasedRLEnv",
    position_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Sparse reward for successful placement.
    
    Args:
        env: The environment instance.
        position_threshold: Distance threshold for successful placement.
        object_cfg: Configuration for the object entity.
        target_cfg: Configuration for the target entity.
    
    Returns:
        Reward tensor shaped (num_envs,).
    """
    object = env.scene[object_cfg.name]
    target = env.scene[target_cfg.name]
    
    # Get positions
    object_pos = object.data.root_pos_w[:, :3]
    target_pos = target.data.root_pos_w[:, :3]
    
    # Check placement
    distance = torch.norm(object_pos - target_pos, p=2, dim=-1)
    success = (distance < position_threshold).float()
    
    return success


def joint_limit_penalty(
    env: "ManagerBasedRLEnv",
    soft_ratio: float = 0.95,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for approaching joint limits.
    
    Args:
        env: The environment instance.
        soft_ratio: Ratio of joint range to start penalizing.
        robot_cfg: Configuration for the robot entity.
    
    Returns:
        Penalty tensor shaped (num_envs,).
    """
    robot = env.scene[robot_cfg.name]
    
    # Get joint positions and limits
    joint_pos = robot.data.joint_pos[:, :6]  # Only arm joints
    joint_limits = robot.data.soft_joint_pos_limits[:, :6, :]
    
    # Normalize to [-1, 1] based on limits
    joint_pos_norm = (joint_pos - joint_limits[:, :, 0]) / (
        joint_limits[:, :, 1] - joint_limits[:, :, 0]
    )
    joint_pos_norm = 2.0 * joint_pos_norm - 1.0
    
    # Penalty for exceeding soft ratio
    penalty = torch.where(
        torch.abs(joint_pos_norm) > soft_ratio,
        torch.abs(joint_pos_norm) - soft_ratio,
        torch.zeros_like(joint_pos_norm),
    )
    
    return -penalty.sum(dim=-1)


def collision_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty for collisions detected by contact sensor.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Force threshold for collision detection.
    
    Returns:
        Penalty tensor shaped (num_envs,).
    """
    # This is a placeholder - implement based on your contact sensor setup
    # For now, return zeros
    return torch.zeros(env.num_envs, device=env.device)


def object_dropped_penalty(
    env: "ManagerBasedRLEnv",
    height_threshold: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty if object falls below table height.
    
    Args:
        env: The environment instance.
        height_threshold: Minimum allowed object height.
        object_cfg: Configuration for the object entity.
    
    Returns:
        Penalty tensor shaped (num_envs,).
    """
    object = env.scene[object_cfg.name]
    
    # Check if object dropped
    object_height = object.data.root_pos_w[:, 2]
    dropped = (object_height < height_threshold).float()
    
    return -dropped

def object_collision_penalty(
    env: "ManagerBasedRLEnv",
    velocity_threshold: float = 0.3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty when object moves too fast (indicates robot hit it)."""
    object = env.scene[object_cfg.name]
    
    # Object velocity
    object_vel = object.data.root_lin_vel_w
    velocity_magnitude = torch.norm(object_vel, dim=1)
    
    # Penalize high velocities (robot crashed into it)
    penalty = torch.where(
        velocity_magnitude > velocity_threshold,
        -torch.ones_like(velocity_magnitude) * 2.0,  # -2.0 penalty
        torch.zeros_like(velocity_magnitude)
    )
    
    return penalty

def shaped_distance_reward(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    tolerance: float = 0.01,  # 1cm tolerance
    hover_height: float = 0.05, # 5cm above object
) -> torch.Tensor:
    """Multi-stage distance reward with better shaping.
    
    Provides dense reward throughout workspace, not just at close range.
    """
    object_entity = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    object_pos = object_entity.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    
    # Compute hover target (above object)
    hover_target = object_pos.clone()
    hover_target[:, 2] += hover_height

    distance = torch.norm(hover_target - ee_pos, p=2, dim=-1)
    
    # Multi-stage reward function
    # 1. Far away (>0.5m): Linear approach reward
    far_reward = torch.clamp(1.0 - distance / 0.5, 0.0, 1.0)
    
    # 2. Medium range (0.1-0.5m): Exponential
    med_reward = torch.exp(-distance / 0.1)
    
    # 3. Close range (<0.1m): Sharp exponential
    close_reward = torch.exp(-distance / 0.02)
    
    # Blend based on distance
    reward = torch.where(
        distance > 0.5,
        far_reward,
        torch.where(
            distance > 0.1,
            med_reward,
            close_reward
        )
    )
    
    # Bonus for being within tolerance
    success_bonus = (distance < tolerance).float() * 2.0
    
    return reward + success_bonus

def reaching_progress_reward(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for making progress (distance reduction).
    
    Encourages active approach behavior.
    """
    object_entity = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    object_pos = object_entity.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    
    current_distance = torch.norm(object_pos - ee_pos, p=2, dim=-1)
    
    # Store previous distance (initialize if needed)
    if not hasattr(env, '_prev_reach_distance'):
        env._prev_reach_distance = current_distance.clone()
    
    # Reward for distance reduction
    progress = env._prev_reach_distance - current_distance
    reward = torch.clamp(progress * 10.0, -1.0, 1.0)  # Scale and clip
    
    # Update stored distance
    env._prev_reach_distance = current_distance.clone()
    
    return reward

def ee_orientation_penalty(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty when gripper Z-axis (along fingers) doesn't point downward."""
    from isaaclab.utils.math import quat_apply
    
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Get Z-axis direction (along fingers) in world frame
    z_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    z_world = quat_apply(ee_quat, z_local)
    
    # Desired: Z pointing down (world -Z direction)
    down = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    
    # Alignment: 1.0 = perfect downward, -1.0 = pointing up
    alignment = torch.sum(z_world * down, dim=-1)
    
    # Convert to penalty: 0 at perfect, -1 at worst
    penalty = (alignment - 1.0) * 0.5  # Maps [1, -1] → [0, -1]
    
    return penalty

def ee_orientation_penalty_object_aligned(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty for EE not aligning with object orientation for top-down grasp.
    
    Computes desired EE orientation based on object's current orientation,
    then penalizes deviation.
    """
    from isaaclab.utils.math import quat_error_magnitude, quat_mul, quat_from_euler_xyz
    
    ee_frame = env.scene[ee_frame_cfg.name]
    object_entity = env.scene[object_cfg.name]
    
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    object_quat = object_entity.data.root_quat_w
    
    # Create a "top-down approach" offset (rotate 180° around X to point down)
    num_envs = env.num_envs
    approach_offset = quat_from_euler_xyz(
        torch.full((num_envs,), 3.14159, device=env.device),  # 180° roll
        torch.zeros(num_envs, device=env.device),
        torch.zeros(num_envs, device=env.device),
    )
    
    # Desired orientation: object orientation with top-down approach
    desired_quat = quat_mul(object_quat, approach_offset)
    
    # Compute angular error
    error = quat_error_magnitude(ee_quat, desired_quat)
    
    # Convert to penalty
    penalty = -error / 3.14159
    
    return penalty

def reach_success(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    threshold: float = 0.01,  # 1cm tolerance
    hover_height: float = 0.05
) -> torch.Tensor:
    """Binary success indicator for reaching task.
    
    Args:
        env: The environment instance.
        threshold: Distance threshold for success (meters).
        object_cfg: Configuration for the object entity.
        ee_frame_cfg: Configuration for the end-effector frame.
        hover_height: Height above object to reach.
    
    Returns:
        Success tensor shaped (num_envs,) with 1.0 for success, 0.0 for failure.
    """
    object_entity = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    object_pos = object_entity.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    
    # Compute hover target
    hover_target = object_pos.clone()
    hover_target[:, 2] += hover_height
    
    # Distance to hover target
    distance = torch.norm(hover_target - ee_pos, p=2, dim=-1)
    success = (distance < threshold).float()

    env.extras['reach_success'] = success.cpu().numpy()  # Store for logging
    
    return success