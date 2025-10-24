from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import REACH_POSITION_SUCCESS_THRESHOLD, REACH_ROTATION_SUCCESS_THRESHOLD, reach_success
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import reach_goal_bonus
from isaaclab.envs.mdp import reset_root_state_uniform

def reset_object_on_success(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    pose_range: dict,
    velocity_range: dict,
    position_threshold: float | None = REACH_POSITION_SUCCESS_THRESHOLD,
    rotation_threshold: float | None = REACH_ROTATION_SUCCESS_THRESHOLD,
):
    """Reset object when reach goal is achieved.
    """    
    # Check which envs succeeded
    success = reach_goal_bonus(env, source_frame_cfg, target_frame_cfg, position_threshold, rotation_threshold)
    success_env_ids = (success > 0.5).nonzero(as_tuple=False).squeeze(-1)
    
    if len(success_env_ids) > 0:        
        reset_root_state_uniform(
            env,
            success_env_ids,
            pose_range=pose_range,
            velocity_range=velocity_range,
            asset_cfg=object_cfg,
        )

def update_success_metrics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    position_threshold: float = REACH_POSITION_SUCCESS_THRESHOLD,
    rotation_threshold: float = REACH_ROTATION_SUCCESS_THRESHOLD,
):
    """Track metrics for non-terminating multi-success reaching task."""
    combined_success = reach_goal_bonus(
        env, source_frame_cfg, target_frame_cfg, 
        position_threshold, rotation_threshold
    )
    success_mask = combined_success > 0.5
    
    # Initialize trackers once
    if not hasattr(env, "_metrics"):
        env._metrics = {
            "episode_reward": torch.zeros(env.num_envs, device=env.device),
            "consecutive_successes": torch.zeros(env.num_envs, dtype=torch.int, device=env.device),
            "episode_successes": torch.zeros(env.num_envs, dtype=torch.int, device=env.device),
            "total_successes": 0,
            "last_episode_reward": 0.0,
            "last_successes_per_episode": 0.0,
        }
    
    metrics = env._metrics
    
    # Update metrics
    metrics["episode_reward"] += env.reward_buf
    metrics["consecutive_successes"] = torch.where(
        success_mask,
        metrics["consecutive_successes"] + 1,
        torch.zeros_like(metrics["consecutive_successes"])
    )
    metrics["episode_successes"][success_mask] += 1
    metrics["total_successes"] += success_mask.sum().item()
    
    # Log ALL metrics every step
    if "log" not in env.extras:
        env.extras["log"] = {}
    
    env.extras["log"]["episode_reward"] = metrics["last_episode_reward"]
    env.extras["log"]["consecutive_successes"] = metrics["consecutive_successes"].float().mean().item()
    env.extras["log"]["success_rate"] = success_mask.float().mean().item()
    env.extras["log"]["total_successes"] = metrics["total_successes"]
    env.extras["log"]["successes_per_episode"] = metrics["last_successes_per_episode"]


def reset_success_counters(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Reset episode metrics on termination."""
    if not hasattr(env, "_metrics"):
        return
    
    if len(env_ids) == 0:
        return
    
    metrics = env._metrics
    
    # Log completed episode stats
    metrics["last_episode_reward"] = metrics["episode_reward"][env_ids].mean().item()
    metrics["last_successes_per_episode"] = metrics["episode_successes"][env_ids].float().mean().item()
    
    # Reset counters for terminated envs
    metrics["episode_reward"][env_ids] = 0.0
    metrics["consecutive_successes"][env_ids] = 0
    metrics["episode_successes"][env_ids] = 0

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
