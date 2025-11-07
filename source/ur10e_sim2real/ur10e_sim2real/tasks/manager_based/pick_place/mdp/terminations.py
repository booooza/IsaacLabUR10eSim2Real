import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import REACH_POSITION_SUCCESS_THRESHOLD, REACH_ROTATION_SUCCESS_THRESHOLD, object_is_lifted, reach_goal_bonus, success_bonus

def success_termination(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    position_threshold: float | None = REACH_POSITION_SUCCESS_THRESHOLD,
    rotation_threshold: float | None = REACH_ROTATION_SUCCESS_THRESHOLD,
) -> torch.Tensor:
    """Terminate episode when EE reaches hover target position.
    
    Matches the success_bonus success condition.
    
    Args:
        env: The environment instance.
        source_frame_cfg: Configuration for the source frame (e.g., end-effector).
        target_frame_cfg: Configuration for the target frame (e.g., hover target).
        position_threshold: Distance threshold for success (meters). None to ignore position check.
        rotation_threshold: Angular threshold for success (radians). None to ignore rotation check.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """

    success = success_bonus(env, source_frame_cfg, target_frame_cfg, position_threshold, rotation_threshold)
    
    return success > 0.5 # float (0.0 or 1.0) to boolean

def reach_termination(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    position_threshold: float | None = REACH_POSITION_SUCCESS_THRESHOLD,
    rotation_threshold: float | None = REACH_ROTATION_SUCCESS_THRESHOLD,
) -> torch.Tensor:
    """Terminate episode when EE reaches hover target position.
    
    Matches the reach_goal_bonus success condition.
    
    Args:
        env: The environment instance.
        source_frame_cfg: Configuration for the source frame (e.g., end-effector).
        target_frame_cfg: Configuration for the target frame (e.g., hover target).
        position_threshold: Distance threshold for success (meters). None to ignore position check.
        rotation_threshold: Angular threshold for success (radians). None to ignore rotation check.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """

    success = reach_goal_bonus(env, source_frame_cfg, target_frame_cfg, position_threshold, rotation_threshold)
    
    return success > 0.5 # float (0.0 or 1.0) to boolean

def lift_termination(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    spawn_offset: float = 0.01,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary reward for lifting object above robot base + threshold.
    
    Args:
        minimal_height: Minimum height above robot base to consider "lifted" (m).
    
    Returns:
        1.0 if object lifted above (robot_base_z + minimal_height), else 0.0.
    """
    lifted = object_is_lifted(env, minimal_height=minimal_height, object_cfg=object_cfg)
    return lifted > 0.5 # float (0.0 or 1.0) to boolean
    
def out_of_bound(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        in_bound_range: The range in x, y, z such that the object is considered in range
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = object.data.root_pos_w - env.scene.env_origins
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def abnormal_robot_state(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)
