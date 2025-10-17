import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import POSITION_SUCCESS_THRESHOLD, ROTATION_SUCCESS_THRESHOLD, reach_goal_bonus

def reach_termination(
    env: "ManagerBasedRLEnv",
    source_frame_cfg: SceneEntityCfg,
    target_frame_cfg: SceneEntityCfg,
    position_threshold: float | None = POSITION_SUCCESS_THRESHOLD,
    rotation_threshold: float | None = ROTATION_SUCCESS_THRESHOLD,
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
