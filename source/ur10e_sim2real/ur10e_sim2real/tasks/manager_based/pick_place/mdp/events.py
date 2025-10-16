from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import reach_success
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz

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
