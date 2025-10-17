from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import reach_success
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
    position_threshold: float | None = 0.1,
    rotation_threshold: float | None = 0.1,
):
    """Reset object when reach goal is achieved.
    """    
    # Check which envs succeeded
    success = reach_goal_bonus(env, source_frame_cfg, target_frame_cfg, position_threshold, rotation_threshold)
    success_env_ids = (success > 0.5).nonzero(as_tuple=False).squeeze(-1)
    
    env.extras['success'] = success.cpu().numpy()  # Store for logging

    if len(success_env_ids) > 0:        
        reset_root_state_uniform(
            env,
            success_env_ids,
            pose_range=pose_range,
            velocity_range=velocity_range,
            asset_cfg=object_cfg,
        )

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
