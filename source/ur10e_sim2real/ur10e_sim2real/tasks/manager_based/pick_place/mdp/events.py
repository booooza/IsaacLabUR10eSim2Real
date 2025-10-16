from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import reach_success
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz

def reset_object_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    position_range_x: tuple[float, float] = (0.4, 0.6),
    position_range_y: tuple[float, float] = (-0.2, 0.2),
    height: float = 0.00,
):
    """Reset object to random pose on table.

    Args:
        env: Environment instance.
        env_ids: Environment IDs to reset.
        object_cfg: Object entity configuration.
        position_range_x: X-position range (min, max).
        position_range_y: Y-position range (min, max).
        height: Fixed Z height above table.
    """
    object_entity = env.scene[object_cfg.name]
    num_resets = len(env_ids)

    # Sample local positions relative to table
    pos_x = sample_uniform(position_range_x[0], position_range_x[1], (num_resets, 1), device=env.device)
    pos_y = sample_uniform(position_range_y[0], position_range_y[1], (num_resets, 1), device=env.device)
    pos_z = torch.full((num_resets, 1), height, device=env.device)
    local_positions = torch.cat([pos_x, pos_y, pos_z], dim=-1)

    # Add env origins to place object correctly in each env
    env_origins = env.scene.env_origins[env_ids]   # (num_resets, 3)
    positions = local_positions + env_origins

    # Orientation: -90° on X + random yaw
    roll = torch.full((num_resets,), -3.14 / 2, device=env.device)
    pitch = torch.zeros(num_resets, device=env.device)
    yaw = sample_uniform(-3.14, 3.14, (num_resets,), device=env.device)
    orientations = quat_from_euler_xyz(roll, pitch, yaw)

    poses = torch.cat([positions, orientations], dim=-1)
    object_entity.write_root_pose_to_sim(poses, env_ids=env_ids)

    velocities = torch.zeros((num_resets, 6), device=env.device)
    object_entity.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_target_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    position_range_x: tuple[float, float] = (0.5, 0.7),
    position_range_y: tuple[float, float] = (-0.2, 0.2),
    height: float = 0.00,
    min_distance_from_object: float = 0.15,
):
    """Reset target to random pose on table.
    
    Args:
        env: Environment instance.
        env_ids: Environment IDs to reset.
        target_cfg: Target entity configuration.
        position_range_x: X-position range (min, max).
        position_range_y: Y-position range (min, max).
        height: Fixed Z height above table.
        min_distance_from_object: Minimum distance from object (currently unused).
    """
    target_entity = env.scene[target_cfg.name]
    
    # Sample random positions
    num_resets = len(env_ids)
    pos_x = sample_uniform(position_range_x[0], position_range_x[1], (num_resets, 1), device=env.device)
    pos_y = sample_uniform(position_range_y[0], position_range_y[1], (num_resets, 1), device=env.device)
    pos_z = torch.full((num_resets, 1), height, device=env.device)
    
    env_origins = env.scene.env_origins[env_ids]
    positions = torch.cat([pos_x, pos_y, pos_z], dim=-1)
    positions = positions + env_origins
    
    # Identity orientation
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(num_resets, 1)
    
    # Combine and write
    poses = torch.cat([positions, orientations], dim=-1)
    target_entity.write_root_pose_to_sim(poses, env_ids=env_ids)
    
    velocities = torch.zeros((num_resets, 6), device=env.device)
    target_entity.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def reset_articulation_to_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
        articulation_asset.set_joint_velocity_target(default_joint_vel, env_ids=env_ids)
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

def log_custom_metrics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    # Compute reach success metric for the specified env ids
    success_tensor = reach_success(env)  # returns tensor (num_envs,)
    
    # Convert to numpy and select only the environments relevant for this event call
    success_vals = success_tensor[env_ids].cpu().numpy()
    
    # Return dictionary with logging keys expected by IsaacLab
    # Follow naming convention used in others like 'Episode_Reward/...'
    return {"Success/reach": success_vals}
