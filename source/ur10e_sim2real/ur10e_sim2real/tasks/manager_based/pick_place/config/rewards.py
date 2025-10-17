from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp

from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg

from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import *

@configclass
class ReachStageRewardsCfg:
    """Rewards optimized for reach task."""
    
    # Distance Reward
    # Max reward: 0.0 (when distance = 0, right at target)
    # The distance d_goal measures how close the end effector is to the target. A smaller
    # distance means that the end effector is closer to the target, which is desirable to fulfill the task.
    # d_goal = ||obj_pos - target_pos||_2
    distance_to_target = RewardTermCfg(
        func=mdp.distance_l2,
        weight=-2.0,  # dist_reward_scale = -2.0
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )

    # Rotation Distance
    # Max reward: ~10.0 (when perfectly aligned)
    # The rotation is represented as the difference between quaternions (quat_diff). The smaller
    # this rotation distance is, the better the end effectors’ orientation.
    # rot_dist = 2 * arcsin(min(||quat_diff[:,1:4]||_2, 1.0))

    # Rotation Reward
    # r_rot = \frac{1}{|rot_dist| + rot_eps} * rot_reward_scale
    # rot_reward_scale = 1.0, rot_eps = 0.1
    orientation_alignment = RewardTermCfg(
        func=mdp.orientation_alignment_l2,
        weight=1.0,  # rot_reward_scale = 1.0
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "rot_eps": 0.1,
        },
    )

    # Action Penalty
    # r_act = -( \sum_{t=1}^{6} a_t^2 ) * action_penalty_scale
    # action_penalty_scale = -0.0002
    action_penalty = RewardTermCfg(
        func=mdp.action_l2,
        weight=-0.0002,
    )

    # Reach Bonus: Sparse reward when reaching target
    # successTolerance = 0.1
    # reach_position_bonus = RewardTermCfg(
    #     func=mdp.reach_goal_bonus,
    #     weight=100.0,
    #     params={
    #         "source_frame_cfg": SceneEntityCfg("ee_frame"),
    #         "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
    #         "position_threshold": 0.1,  # 10cm
    #         "rotation_threshold": None,  # Ignore rotation
    #     },
    # )
    reach_pose_bonus = RewardTermCfg(
        func=mdp.reach_goal_bonus,
        weight=250.0,  # reach_goal_bonus = 250
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": 0.1,  # 10cm
            "rotation_threshold": 0.1,  # ~5.7 degrees
        },
    )

    # Total Reward:
    # r_t = - (d_goal * dist_reward_scale) + r_t - r_act + reach_goal_bonus
    # reach_goal_bonus = 250
    # dist_reward_scale = -2.0
    # velObsScale = 0.2

    # Optional: Joint velocity penalty (using velObsScale = 0.2)
    # joint_vel_penalty = RewardTermCfg(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.0001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

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
    
    action = RewardTermCfg(func=mdp.action_l2, weight=-0.005)

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
    