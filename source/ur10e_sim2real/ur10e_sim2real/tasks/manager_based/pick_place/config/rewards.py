from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp

from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg

from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import *

@configclass
class ReachStageRewardsCfg:
    """Rewards optimized for reach task."""
    
    # Main reaching reward
    reaching_reward = RewardTermCfg(
        func=shaped_distance_reward,
        weight=3.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "tolerance": 0.02,
        },
    )
    
    # Progress incentive
    progress_reward = RewardTermCfg(
        func=reaching_progress_reward,
        weight=1.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )
    
    # Regularization
    action_rate_penalty = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    
    joint_vel_penalty = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
@configclass
class PickPlaceRewardsCfg:
    """Reward terms configuration."""
    
    # Stage 1: Reaching rewards
    distance_ee_to_object = RewardTermCfg(
        func=distance_reward_ee_to_object,
        weight=2.0,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )
    
    # Sparse success reward
    grasp_success = RewardTermCfg(
        func=grasp_success_reward,
        weight=1.0,
        params={
            "threshold": 0.05, # 5 cm threshold for grasp success
            "object_cfg": SceneEntityCfg("object"), 
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )
    
    # Penalties
    action_rate_penalty = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.001,
    )
    
    joint_vel_penalty = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    joint_limit_penalty = RewardTermCfg(
        func=joint_limit_penalty,
        weight=-0.5,
        params={
            "soft_ratio": 0.95,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    