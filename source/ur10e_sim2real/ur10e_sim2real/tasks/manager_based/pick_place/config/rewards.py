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
            "tolerance": 0.01,
            "hover_height": 0.05,
        },
    )
    
    reach_success = RewardTermCfg(
        func=reach_success,
        weight=5.0,
        params={
            "threshold": 0.10,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "hover_height": 0.05,
        },
    )
    
    # Regularization
    collision_penalty = RewardTermCfg(
        func=object_collision_penalty,
        weight=1.0,
        params={
            "velocity_threshold": 0.3,
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    
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
        func=shaped_distance_reward,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "tolerance": 0.01,
            "hover_height": 0.05,
        },
    )
    
    reach_success = RewardTermCfg(
        func=mdp.reach_success,
        weight=1.0,  # Zero weight - only for tracking
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "threshold": 0.01,  # 1cm
            "hover_height": 0.05,
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

    pick_success = RewardTermCfg(
        func=mdp.pick_success_reward,
        weight=1.0,  # Zero weight - only for tracking
        params={
            "height_threshold": 0.15,  # 15 cm
            "grasp_threshold": 0.05,  # 5 cm
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )
    
    # Penalties
    orientation_penalty = RewardTermCfg(
        func=mdp.ee_orientation_penalty,
        weight=1.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )
    
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
    