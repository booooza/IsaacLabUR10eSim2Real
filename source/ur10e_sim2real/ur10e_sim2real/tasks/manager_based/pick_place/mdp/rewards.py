"""Reward functions for pick-and-place task."""

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import quat_mul, quat_conjugate, quat_rotate, quat_apply, quat_error_magnitude
from isaaclab.assets import RigidObject, Articulation
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

REACH_POSITION_SUCCESS_THRESHOLD = 0.005 # 5 mm
REACH_ROTATION_SUCCESS_THRESHOLD = 0.0349066 # ~2 degrees
LIFT_HEIGHT_SUCCESS_THRESHOLD = 0.10 # 10 cm
PLACE_DISTANCE_THRESHOLD = 0.02  # 2cm from target

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

def distance_exp(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Exponential distance reward: exp(-d²/2sigma²)"""
    distance = distance_l2(env, source_frame_cfg, target_frame_cfg)
    return torch.exp(-distance**2 / (2 * sigma**2))

def distance_tanh(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    std: float = 0.1,
) -> torch.Tensor:
    """Tanh-shaped distance reward (Isaac Lab style).
    
    Returns reward in [0, 1] where:
    - 1.0 = at target
    - 0.5 = at distance 'std'
    - 0.0 = very far
    
    Args:
        env: The environment instance.
        source_frame_cfg: Source frame (e.g., end-effector).
        target_frame_cfg: Target frame (e.g., hover target).
        std: Standard deviation controlling reward steepness.
             Smaller = steeper (harder), Larger = gentler (easier)
    
    Returns:
        Reward tensor shaped (num_envs,).
    """
    distance = distance_l2(env, source_frame_cfg, target_frame_cfg)
    reward = 1.0 - torch.tanh(distance / std)
    # Tanh kernel: closer to target = higher reward
    return reward

def orientation_alignment_l2(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
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

def orientation_error(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
) -> torch.Tensor:
    """Orientation alignment reward between two frames as 
    the angular error between input quaternions in radians..
    
    Computes quaternion difference magnitude.
    """
    from isaaclab.utils.math import quat_error_magnitude

    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]

    source_quat = source_frame.data.target_quat_w[..., 0, :]
    target_quat = target_frame.data.target_quat_w[..., 0, :]

    # Full orientation matching
    return quat_error_magnitude(source_quat, target_quat)

def orientation_error_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    source_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
) -> torch.Tensor:
    """Orientation alignment reward between two frames using tanh kernel.
    
    Computes quaternion difference magnitude.
    """
    from isaaclab.utils.math import quat_error_magnitude

    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]

    source_quat = source_frame.data.target_quat_w[..., 0, :]
    target_quat = target_frame.data.target_quat_w[..., 0, :]

    # Full orientation matching
    quat_distance = quat_error_magnitude(source_quat, target_quat)
    return (1 - torch.tanh(quat_distance / std))

def orientation_error_exp(
    env: "ManagerBasedRLEnv",
    std: float = 0.3,
    source_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
) -> torch.Tensor:
    """Orientation alignment reward between two frames using exponential kernel.
    
    Computes bounded reward [0, 1] based on quaternion difference:
    - 1.0 = perfect orientation alignment
    - 0.0 = opposite orientations
    """
    angle_error = orientation_error(env, source_frame_cfg, target_frame_cfg)
    return torch.exp(-(angle_error**2) / (2 * std**2))

def orientation_alignment_exp_scaled(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    std: float = 0.3,
    distance_threshold: float = 0.15,
) -> torch.Tensor:
    """Orientation reward that scales with proximity to target.
    
    When far away: orientation reward is minimal (focus on getting close)
    When close: orientation reward is full (focus on alignment)
    
    Returns:
        Reward in range [0, 1] that scales with both orientation and proximity.
    """    
    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]
    
    source_quat = source_frame.data.target_quat_w[..., 0, :]
    target_quat = target_frame.data.target_quat_w[..., 0, :]
    
    # Calculate orientation reward
    angle_error = quat_error_magnitude(source_quat, target_quat)
    orientation_reward = torch.exp(-(angle_error**2) / (2 * std**2))
    
    # Calculate distance to target frame (consistent with orientation target)
    distance = distance_l2(env, source_frame_cfg, target_frame_cfg)
    
    # Scale orientation reward by proximity
    # At 0m: scale = 1.0 (full orientation reward)
    # At >distance_threshold: scale ≈ 0 (minimal orientation reward)
    proximity_scale = 1.0 - torch.tanh(distance / distance_threshold)
    
    scaled_reward = orientation_reward * proximity_scale
    
    return scaled_reward

def orientation_alignment_cos_scaled(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    distance_threshold: float = 0.15,
    alpha: float = 2.0,
    min_scale: float = 0.2,
) -> torch.Tensor:
    """Cosine-based orientation reward scaled by proximity.
    - 1.0 when perfectly aligned
    - 0.5 at 90° misalignment
    - 0.0 at 180° misalignment
    Smooth gradients, strong learning signal.
    """
    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]

    source_quat = source_frame.data.target_quat_w[..., 0, :]
    target_quat = target_frame.data.target_quat_w[..., 0, :]

    # quaternion angular error (radians)
    angle_error = quat_error_magnitude(source_quat, target_quat)
    angle_error = torch.clamp(angle_error, max=torch.pi)

    # cosine reward
    orientation_reward = ((1 + torch.cos(angle_error)) / 2) ** alpha

    distance = distance_l2(env, source_frame_cfg, target_frame_cfg)
    proximity_scale = min_scale + (1 - min_scale) * (1.0 - torch.tanh(distance / distance_threshold))

    return orientation_reward * proximity_scale

def orientation_angular_error(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Simple orientation angular error between source and target quaternions in radians."""
    from isaaclab.utils.math import quat_error_magnitude
    
    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]
    
    source_quat = source_frame.data.target_quat_w[..., 0, :]
    target_quat = target_frame.data.target_quat_w[..., 0, :]
    
    # Returns angular error in radians
    return quat_error_magnitude(source_quat, target_quat)

def action_rate_limit(
    env: "ManagerBasedRLEnv", 
    threshold_ratio: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize action rate changes exceeding a threshold ratio of joint velocity limits.
    
    Small oscillations within the threshold are acceptable and not penalized.
    
    Args:
        threshold_ratio: Ratio of joint velocity limits to use as threshold (default: 0.1 = 10%)
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get action difference
    action_diff = env.action_manager.action - env.action_manager.prev_action
    
    # Get joint velocity limits from the asset (for the controlled joints)
    joint_vel_limits = asset.data.joint_vel_limits[:, asset_cfg.joint_ids]
    
    # Calculate threshold (10% of velocity limits)
    threshold = joint_vel_limits * threshold_ratio
    
    # Calculate excess beyond threshold (zero if within threshold)
    excess = torch.abs(action_diff) - threshold
    excess = torch.clamp(excess, min=0.0)
    
    # Return squared penalty on excess
    return torch.sum(torch.square(excess), dim=1)

def reach_goal_bonus(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    position_threshold: float | None = REACH_POSITION_SUCCESS_THRESHOLD,
    rotation_threshold: float | None = REACH_ROTATION_SUCCESS_THRESHOLD,
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

    if (success.any() and position_threshold is not None and rotation_threshold is not None):
        print(f"Reached goal pose for {success.sum().item()} envs.")
    
    return success.float()


def success_bonus(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    position_threshold: float | None = REACH_POSITION_SUCCESS_THRESHOLD,
    rotation_threshold: float | None = REACH_ROTATION_SUCCESS_THRESHOLD,
) -> torch.Tensor:
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
        rotation_error = quat_error_magnitude(source_quat, target_quat)
        success = success & (rotation_error < rotation_threshold)

    if (success.any() and position_threshold is not None and rotation_threshold is not None):
        print(f"Reached goal pose for {success.sum().item()} envs.")
    
    return success.float()        
        
def precision_bonus(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    position_scale: float = 0.01,
    rotation_scale: float = 0.05,
) -> torch.Tensor:
    """Exponential bonus for high precision reaching.
    
    Rewards getting very close to the target with exponential scaling.
    This encourages the final millimeters of precision.
    
    Args:
        env: Environment instance
        source_frame_cfg: End-effector frame
        target_frame_cfg: Target frame
        position_scale: Distance scale for exponential (smaller = more precise)
        rotation_scale: Rotation scale for exponential
        
    Returns:
        Precision bonus [0, 1]
    """
    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]
    
    # Position error
    source_pos = source_frame.data.target_pos_w[..., 0, :3]
    target_pos = target_frame.data.target_pos_w[..., 0, :3]
    pos_error = torch.norm(target_pos - source_pos, p=2, dim=-1)
    
    # Rotation error (uses Isaac Lab's quaternion error magnitude)
    source_quat = source_frame.data.target_quat_w[..., 0, :]
    target_quat = target_frame.data.target_quat_w[..., 0, :]
    from isaaclab.utils.math import quat_error_magnitude
    rot_error = quat_error_magnitude(source_quat, target_quat)
    
    # Exponential bonus (peaks when error → 0)
    # exp(-error/scale): smaller scale = steeper falloff = harder to get reward
    pos_bonus = torch.exp(-pos_error / position_scale)
    rot_bonus = torch.exp(-rot_error / rotation_scale)
    
    # Multiply bonuses: both position AND orientation must be good
    return pos_bonus * rot_bonus

def smooth_precision_bonus(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    position_threshold: float = 0.02,
    rotation_threshold: float = 0.1,
    position_scale: float = 0.005,
    rotation_scale: float = 0.03,
    transition_width: float = 0.005,
) -> torch.Tensor:
    """Precision bonus with smooth activation around threshold.
    
    Uses sigmoid to smoothly transition from 0 reward (far from goal)
    to exponential precision bonus (near goal).
    """
    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]
    
    source_pos = source_frame.data.target_pos_w[..., 0, :3]
    target_pos = target_frame.data.target_pos_w[..., 0, :3]
    pos_error = torch.norm(target_pos - source_pos, p=2, dim=-1)
    
    source_quat = source_frame.data.target_quat_w[..., 0, :]
    target_quat = target_frame.data.target_quat_w[..., 0, :]
    from isaaclab.utils.math import quat_error_magnitude
    rot_error = quat_error_magnitude(source_quat, target_quat)
    
    # Exponential precision (as before)
    pos_precision = torch.exp(-pos_error / position_scale)
    rot_precision = torch.exp(-rot_error / rotation_scale)
    precision_score = pos_precision * rot_precision
    
    # Smooth activation gates (sigmoid centered at threshold)
    # When error > threshold: gate ≈ 0
    # When error < threshold: gate ≈ 1
    # Smooth transition over transition_width
    pos_gate = torch.sigmoid((position_threshold - pos_error) / transition_width)
    rot_gate = torch.sigmoid((rotation_threshold - rot_error) / transition_width)
    activation_gate = pos_gate * rot_gate
    
    return precision_score * activation_gate

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
    
    # Inverse penalty: 1/μ when below threshold
    penalty = torch.where(
        manip < threshold,
        1.0 / (manip + 1e-6),
        torch.zeros_like(manip)
    )
    
    return penalty

def manipulability_penalty_linear(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.02,
    soft_ratio: float = 0.5,
) -> torch.Tensor:
    """Penalize low manipulability similarly to joint limit penalties.
    
    Linearly penalizes when manipulability drops below threshold,
    with optional soft region (soft_ratio * threshold).
    """
    from .observations import manipulability_index
    
    manip = manipulability_index(env, asset_cfg).squeeze(-1)
    
    # Define a soft threshold below which penalty starts
    soft_threshold = threshold * soft_ratio
    
    # Compute out-of-bounds penalty
    out_of_limits = (soft_threshold - manip).clip(min=0.0)
    
    # Normalize so that at manip=0, penalty ≈ 1.0
    penalty = (out_of_limits / soft_threshold).clip(max=1.0)
    
    return penalty

def manipulability_penalty_quadratic(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.02,
    soft_ratio: float = 0.5,
) -> torch.Tensor:
    """Quadratic penalty for low manipulability.

    Penalizes configurations near kinematic singularities
    using a smooth, bounded, and differentiable quadratic kernel.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot.
        threshold: Manipulability value below which the penalty saturates.
        soft_ratio: Ratio of threshold at which penalty starts rising.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    from .observations import manipulability_index
    
    manip = manipulability_index(env, asset_cfg).squeeze(-1)
    
    # Define soft threshold: penalty starts below this
    soft_threshold = threshold * soft_ratio
    
    # Compute distance below soft threshold (negative = within good region)
    deficit = (soft_threshold - manip).clip(min=0.0)
    
    # Quadratic penalty normalized to [0, 1]
    # At manip=0, penalty ≈ 1.0
    penalty = torch.square(deficit / soft_threshold).clip(max=1.0)
    
    return penalty


def joint_velocity_l2_conditional(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    distance_threshold: float,
) -> torch.Tensor:
    """Penalize joint velocities only when close to target.
    """
    robot = env.scene[asset_cfg.name]
    source_frame = env.scene[source_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]
    
    # Distance check
    distance = torch.norm(
        source_frame.data.target_pos_w[..., 0, :3] - target_frame.data.target_pos_w[..., 0, :3],
        dim=-1
    )
    
    # Joint velocity magnitude
    joint_vel = robot.data.joint_vel[:, :6]  # Arm joints only
    vel_l2 = torch.sum(torch.square(joint_vel), dim=1)
    
    # CONDITIONAL: only penalize when close (Isaac Lab pattern)
    is_close = distance < distance_threshold
    return torch.where(is_close, vel_l2, torch.zeros_like(vel_l2))

def action_rate_l2_conditional(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    distance_threshold: float,
) -> torch.Tensor:
    """Penalize action changes only when close to target."""
    distance = distance_l2(env, source_frame_cfg, target_frame_cfg)
    
    # Action rate (change in action between steps)
    action_diff = env.action_manager.action - env.action_manager.prev_action
    action_rate_l2 = torch.sum(torch.square(action_diff), dim=1)
    
    # Exponentially increase penalty as we approach
    penalty_scale = torch.exp(-distance / distance_threshold)
    
    return action_rate_l2 * penalty_scale

def minimum_link_distance(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    min_distance: float = 0.05,
    link_pairs: list = None,
) -> torch.Tensor:
    """Penalize when robot links get too close to each other."""
    robot = env.scene[asset_cfg.name]
    
    # Get link positions
    link_pos_w = robot.data.body_pos_w
    
    # Calculate minimum distances between specified link pairs
    penalties = []
    for link1_name, link2_name in link_pairs:
        # Get link indices
        link1_idx = robot.find_bodies(link1_name)[0][0]
        link2_idx = robot.find_bodies(link2_name)[0][0]
        
        # Calculate distance
        distance = torch.norm(
            link_pos_w[:, link1_idx] - link_pos_w[:, link2_idx], 
            dim=-1
        )
        
        # Penalize when closer than min_distance
        penalty = torch.clamp(min_distance - distance, min=0.0)
        penalties.append(penalty)
    
    # Return maximum penalty across all link pairs
    return torch.stack(penalties, dim=-1).max(dim=-1)[0]

# ----
## Additional reward functions for pick-and-place task
def object_is_grasped(
    env: "ManagerBasedRLEnv",
    threshold: float,
    open_joint_pos: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["robotiq_hande_left_finger_joint"])
) -> torch.Tensor:
    """Simple grasp reward: proximity + closure.
    
    No velocity coupling to keep it simple and avoid false positives.
    Real grasp verification comes from lifting rewards.
    
    Returns: [0, 1] continuous value.
    """
    # Get positions
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids].squeeze(-1)
    
    # 1. Proximity: Exponential decay
    distance = torch.norm(object_pos - ee_pos, dim=-1)
    proximity = torch.exp(-distance / threshold)
    
    # 2. Closure: Linear scaling
    closed_ratio = (open_joint_pos - gripper_pos) / open_joint_pos
    closed_ratio = torch.clamp(closed_ratio, 0.0, 1.0)
    
    # Simple product (no velocity)
    return proximity * closed_ratio

def object_is_between_fingers(
    env: "ManagerBasedRLEnv",
    threshold: float,
    open_joint_pos: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_finger_body_name: str = "robotiq_hande_left_finger", 
    right_finger_body_name: str = "robotiq_hande_right_finger",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["robotiq_hande_left_finger_joint"])
) -> torch.Tensor:
    """Calculates a continuous reward based on the object being centered between the gripper links 
    while the gripper is closing.
    
    FIXED: Accesses link positions via the Articulation's body_link_pose_w data by looking up the 
    body index using the provided body name strings.
    
    Args:
        env: The environment instance.
        threshold: Proximity scale factor for the exponential decay (controls decay rate).
        open_joint_pos: The joint position when the gripper is fully open.
        object_cfg: Configuration for the object entity.
        left_finger_body_name: The string name of the left finger body.
        right_finger_body_name: The string name of the right finger body.
        robot_cfg: Configuration for the robot/gripper joint.
        
    Returns: 
        A continuous reward value in the range [0, 1].
    """
    
    # --- 1. Get Positions and Joint Data ---
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    
    # Get the robot Articulation
    robot: Articulation = env.scene[robot_cfg.name] 
    
    # Resolve the body indices using the string names from the Articulation's internal list.
    left_finger_id = robot.body_names.index(left_finger_body_name)
    right_finger_id = robot.body_names.index(right_finger_body_name)

    # Access the position (first 3 elements of the 7-D pose) from the Articulation's body data.
    # robot.data.body_link_pose_w has shape [num_envs, num_bodies, 7 (pos+quat)]
    left_finger_pos = robot.data.body_link_pose_w[:, left_finger_id, 0:3]
    right_finger_pos = robot.data.body_link_pose_w[:, right_finger_id, 0:3]
    
    # Get joint position for closure check
    # Assuming a single joint is used for closure calculation
    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids].squeeze(-1)

    # --- 2. Containment Check (Midpoint Proximity) ---
    # Calculate the midpoint between the two fingers
    midpoint_pos = (left_finger_pos + right_finger_pos) / 2
    
    # Calculate distance from the object center to the finger midpoint
    distance_to_midpoint = torch.norm(object_pos - midpoint_pos, dim=-1)
    
    # Reward is high only when the object is precisely aligned with the gripper center
    proximity_reward = torch.exp(-distance_to_midpoint / threshold)
    
    # --- 3. Closure Ratio ---
    # Measures how far the gripper has closed (0: open, 1: fully closed)
    closed_ratio = (open_joint_pos - gripper_pos) / open_joint_pos
    closed_ratio = torch.clamp(closed_ratio, 0.0, 1.0)
    
    # --- 4. Combined Reward ---
    # The reward is maximized only if the object is centered AND the gripper is closing.
    return proximity_reward * closed_ratio

def grasp_goal_bonus(
    env: "ManagerBasedRLEnv",
    threshold: float,
    open_joint_pos: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["robotiq_hande_left_finger_joint"])
) -> torch.Tensor:
    """Sparse bonus reward for successfully grasping the object.
    
    Checks two conditions:
    1. Object is within threshold distance of end-effector
    2. Gripper is sufficiently closed (below closed_threshold)
    
    Args:
        env: The environment instance.
        threshold: Maximum distance between EE and object for valid grasp (m).
        open_joint_pos: Joint position when gripper is fully open (m).
                       For Robotiq Hand-E: 0.025
        object_cfg: Configuration for the object entity.
        robot_cfg: Configuration for robot gripper joint.
    
    Returns:
        Bonus tensor (num_envs,): 1.0 if grasped, 0.0 otherwise.
        Use with high positive weight: RewardTermCfg(..., weight=500.0)
    """
    # Get end-effector and object positions
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    
    # Get gripper joint position from robot
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids].squeeze(-1)
    
    # Condition 1: Object close to gripper
    distance = torch.norm(object_pos - ee_tcp_pos, dim=-1)
    is_close = distance < threshold
    
    # Condition 2: Gripper is closed
    # Consider "closed" as less than 20% of full range (0.005 for 0.025 max)
    closed_threshold = open_joint_pos * 0.2
    is_closed = gripper_joint_pos < closed_threshold
    
    # Both conditions must be true
    is_grasped = (is_close & is_closed).float()
    
    return is_grasped

def object_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward for lifting object above its initial spawn height.
    
    Args:
        env: Environment instance.
        minimal_height: Minimum lift distance from initial position (m).
        object_cfg: Object entity configuration.
    
    Returns:
        1.0 if lifted by minimal_height above spawn, else 0.0.
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get current and spawn heights
    object_z = object.data.root_pos_w[:, 2]
    spawn_z = object.data.default_root_state[:, 2]
    spawn_z = object.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]

    
    # Lift distance
    lift_distance = object_z - spawn_z
    
    # Binary: 1.0 if lifted by minimal_height
    return torch.where(lift_distance > minimal_height, 1.0, 0.0)

def object_lift_height_reward(
    env: "ManagerBasedRLEnv",
    max_height: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Dense reward for lifting object higher (continuous, not binary).
    
    Uses LINEAR scaling (not tanh) because we want to encourage
    lifting as high as possible, not just "close enough".
    
    Returns values in [0, max_height] range (e.g., 0.0 to 0.2).
    Scale this with the weight parameter in RewardTermCfg.
    
    Args:
        spawn_offset: Object center height when on table (m).
        max_height: Maximum lift height to reward (m). Caps at this value.
    
    Returns:
        Reward proportional to lift height in meters, clipped to [0, max_height].
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get lift distance (same as object_is_lifted)
    object_z = object.data.root_pos_w[:, 2]
    spawn_z = object.data.default_root_state[:, 2]
    lift_distance = object_z - spawn_z
    
    # Linear reward: 0 when on table, increases linearly with height
    # Clipped to max_height to prevent unbounded rewards
    # Returns value in meters: e.g., 0.10m lift → reward = 0.10
    reward = torch.clamp(lift_distance, min=0.0, max=max_height)
    
    return reward

def object_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(object_pos_w - ee_w, dim=1)

    # Range: [0, 1], where 1 = at object, 0 = far away
    return 1 - torch.tanh(object_ee_distance / std)

def grasp_orientation_alignment(
    env: "ManagerBasedRLEnv",
    std: float = 0.3,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> torch.Tensor:
    """Reward alignment of end-effector orientation for top-down grasping.
    
    Uses exponential kernel for bounded reward in [0, 1], where:
    - 1.0 = perfectly aligned for grasping
    - 0.5 = ~30° error
    - 0.0 = perpendicular or opposite orientation
    
    Args:
        env: The environment instance.
        std: Standard deviation for exponential kernel (controls reward sharpness).
        ee_frame_cfg: Configuration for the end-effector frame.
        target_direction: Desired approach direction in world frame.
    
    Returns:
        Reward tensor shaped (num_envs,) in range [0, 1].
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get end-effector quaternion (num_envs, 4) [w, x, y, z]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Convert target direction to tensor
    target_dir = torch.tensor(target_direction, device=env.device, dtype=torch.float32)
    target_dir = target_dir / torch.norm(target_dir)
    target_dir = target_dir.unsqueeze(0).expand(env.num_envs, -1)
    
    # Get the z-axis of the end-effector in world frame
    local_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    ee_z_axis = quat_apply(ee_quat, local_z)
    
    # Compute alignment
    alignment = torch.sum(ee_z_axis * target_dir, dim=-1)
    alignment_clamped = torch.clamp(alignment, -1.0, 1.0)
    angle_error = torch.acos(alignment_clamped)
    
    # Exponential kernel: drops off much faster than tanh
    # At std=0.3: 30° error → ~0.76 reward, 45° error → ~0.53 reward
    reward = torch.exp(-(angle_error**2) / (2 * std**2))
    
    return reward

def grasp_orientation_alignment_adaptive(
    env: "ManagerBasedRLEnv",
    std: float = 0.3,
    distance_std: float = 0.1,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_frame_cfg: SceneEntityCfg = SceneEntityCfg("hover_target_frame"),
    target_direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> torch.Tensor:
    """Reward orientation alignment, scaled by proximity to target.
    
    Orientation matters more when close to target, less when far away.
    This prevents the agent from backing away to perfect orientation.
    
    Returns:
        Reward tensor shaped (num_envs,) in range [0, 1].
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    target_frame = env.scene[target_frame_cfg.name]
    
    # Get end-effector quaternion and position
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
    target_pos = target_frame.data.target_pos_w[..., 0, :3]
    
    # Calculate distance to target
    distance = torch.norm(target_pos - ee_pos, dim=-1)
    
    # Distance factor: orientation matters more when close
    # At 0m: factor = 1.0 (full orientation reward)
    # At >0.3m: factor ≈ 0 (orientation reward minimal)
    distance_factor = 1.0 - torch.tanh(distance / distance_std)
    
    # Convert target direction to tensor
    target_dir = torch.tensor(target_direction, device=env.device, dtype=torch.float32)
    target_dir = target_dir / torch.norm(target_dir)
    target_dir = target_dir.unsqueeze(0).expand(env.num_envs, -1)
    
    # Get the z-axis of the end-effector in world frame
    local_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    ee_z_axis = quat_apply(ee_quat, local_z)
    
    # Compute alignment
    alignment = torch.sum(ee_z_axis * target_dir, dim=-1)
    alignment_clamped = torch.clamp(alignment, -1.0, 1.0)
    angle_error = torch.acos(alignment_clamped)
    
    # Orientation reward (before scaling)
    orientation_reward = 1.0 - torch.tanh(angle_error / std)
    
    # Scale by distance: only care about orientation when close
    scaled_reward = orientation_reward * distance_factor
    
    return scaled_reward

def object_goal_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Reward for moving object toward target using tanh-kernel.
    
    Only active when object is lifted (to avoid rewarding dragging on table).
    
    Args:
        env: Environment instance.
        std: Standard deviation for tanh kernel (larger = smoother/wider reward).
        minimal_height: Minimum lift distance from spawn to activate this reward (m).
        object_cfg: Object entity configuration.
        target_cfg: Target entity configuration.
    
    Returns:
        Reward in [0, 1] when object is lifted and moving toward target.
        0.0 when object is on table (not lifted).
    """
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    
    # Get positions in world frame
    object_pos = object.data.root_pos_w  # (N, 3)
    target_pos = target.data.root_pos_w  # (N, 3)
    
    # Compute distance to target (3D Euclidean distance)
    distance = torch.norm(object_pos - target_pos, dim=-1)  # (N,)
    
    # Check if lifted
    object_z = object_pos[:, 2]
    spawn_z = object.data.default_root_state[:, 2]
    lift_distance = object_z - spawn_z
    is_lifted = (lift_distance > minimal_height).float()
    
    # Tanh-kernel reward: smoothly approaches 1 as distance -> 0
    # Only applied when object is lifted
    reward = (1.0 - torch.tanh(distance / std)) * is_lifted
    
    return reward

def gripper_close_when_near_object(
    env: "ManagerBasedRLEnv",
    distance_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing gripper when near object, opening when far.
    
    Helps learn: approach with open gripper, close when at object.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Distance to object
    object_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object_pos - ee_pos, dim=-1)
    
    # Gripper opening (sum of finger joints)
    # Assuming last 2 joints are gripper fingers
    gripper_joints = robot.data.joint_pos[:, -2:]
    gripper_opening = gripper_joints.sum(dim=-1)
    
    # When near: want closed (small opening)
    # When far: want open (large opening)
    near = (distance < distance_threshold).float()
    far = 1.0 - near
    
    # Target: closed (0.0) when near, open (0.05) when far
    target_opening = 0.05 * far
    
    # Reward proximity to target opening
    reward = -torch.abs(gripper_opening - target_opening)
    
    return reward

def object_height_above_ground(
    env: "ManagerBasedRLEnv",
    target_height: float,
    spawn_offset: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward for lifting object above spawn.
    
    Args:
        target_height: Target LIFT DISTANCE from spawn (m).
        spawn_offset: Object center height when on table (m).
    
    Returns:
        Reward [0, 1]: 0.0 at spawn, 1.0 when lifted by target_height.
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    object_z = object.data.root_pos_w[:, 2]
    robot_z = robot.data.root_pos_w[:, 2]
    relative_height = object_z - robot_z
    
    # Measure lift from spawn
    lift_distance = relative_height - spawn_offset
    
    return torch.clamp(lift_distance / target_height, 0.0, 1.0)

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
    """Penalty for object dropping below table height.
    
    Returns positive penalty value (to be used with negative weight).
    
    Args:
        env: The environment instance.
        height_threshold: Minimum allowed object height above table (m). 
                         Since table is at z=0, this is absolute height.
        object_cfg: Configuration for the object entity.
    
    Returns:
        Penalty tensor (num_envs,): 1.0 if dropped, 0.0 otherwise.
        Use with negative weight: RewardTermCfg(..., weight=-100.0)
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_height = object.data.root_pos_w[:, 2]
    
    # Return 1.0 when dropped (penalty), 0.0 when safe
    is_dropped = (object_height < height_threshold).float()
    return is_dropped

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