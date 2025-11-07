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
    # Coarse guidance when far from target
    distance_to_target = RewardTermCfg(
        func=mdp.distance_l2,
        weight=-2.0,  # dist_reward_scale = -2.0
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )

    # Rotation Reward
    # r_rot = \frac{1}{|rot_dist| + rot_eps} * rot_reward_scale
    # rot_reward_scale = 1.0, rot_eps = 0.1
    orientation_alignment = RewardTermCfg(
        func=mdp.orientation_alignment_l2,
        weight=0.5,  # rot_reward_scale = 1.0
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "rot_eps": 0.1,
        },
    )

    # Manipulability Penalty: Penalize low manipulability
    manipulability_penalty = RewardTermCfg(
        func=mdp.manipulability_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # Keep robot in safer, more dexterous configurations
            "threshold": 0.05,  # Penalize when manipulability < 0.05
        }
    )

    # Action Penalties
    # r_act = -( \sum_{t=1}^{6} a_t^2 ) * action_penalty_scale
    # action_penalty_scale = -0.0002
    # Penalize the actions using L2 squared kernel.
    action_penalty = RewardTermCfg(
        func=mdp.action_l2,
        weight=-0.0002,
    )

    # Penalize the rate of change of the actions using L2 squared kernel.
    action_rate_penalty = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-1e-5,  # will be modified by curriculum up to -0.001
    )

    # Joint Position Limits Penalty
    joint_pos_limits_penalty = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Joint Velocity Penalty when near target
    # Encourage slowing down near the target but move fast when far away.
    joint_vel_penalty_near_target = RewardTermCfg(
        func=mdp.joint_velocity_l2_conditional,
        weight=-1e-5,  # will be modified by curriculum up to -0.5
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "distance_threshold": 0.05,
        },
    )

    # Reach Bonus: Sparse reward when reaching target
    # successTolerance = 0.1
    reach_pose_bonus = RewardTermCfg(
        func=mdp.reach_goal_bonus,
        weight=250.0,  # reach_goal_bonus = 250
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": REACH_POSITION_SUCCESS_THRESHOLD,
            "rotation_threshold": REACH_ROTATION_SUCCESS_THRESHOLD,
        },
    )
    precision_bonus = RewardTermCfg(
        func=mdp.precision_bonus,
        weight=50.0,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_scale": 0.002,  # 2mm scale
            "rotation_scale": 0.05,  # ~2° scale
        },
    )

    # Total Reward:
    # r_t = - (d_goal * dist_reward_scale) + r_t - r_act + reach_goal_bonus
    # reach_goal_bonus = 250
    # dist_reward_scale = -2.0
    # velObsScale = 0.2

@configclass
class ReachStageRewardsCfgV2:
    """Rewards optimized for reach task."""
    distance_l2 = RewardTermCfg(
        func=distance_l2,
        weight=-0.2,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )
    distance_tanh = RewardTermCfg(
        func=distance_tanh,
        weight=0.1,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "std": 2 * REACH_POSITION_SUCCESS_THRESHOLD,
        },
    )
    orientation_error = RewardTermCfg(
        func=orientation_error,
        weight=-0.1,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )
    success_bonus = RewardTermCfg(
        func=mdp.success_bonus,
        weight=10.0,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": REACH_POSITION_SUCCESS_THRESHOLD,
            "rotation_threshold": REACH_ROTATION_SUCCESS_THRESHOLD,
        },
    )

    # Penalty for large action commands - if actions seem too large or erratic, increase the penalty
    # action_l2 = RewardTermCfg(func=mdp.action_l2, weight=-0.001)
    # Penalty for rapid action changes - if the robot is jittery/oscillating, increase the penalty
    # action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.001) # TODO: -0.001 prevents learning.. try start from -0.0001 and set to -0.005 with curriculum
    
    # Penalty for high joint velocities - if the robot is moving too fast, increase the penalty
    # TODO: -0.001 prevents learning.. try start from -0.0001 and set to -0.001 with curriculum after 300*625=187500
    # joint_vel_l2 = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001, params={"asset_cfg": SceneEntityCfg("robot")})

    # Penalties for joint limits violation - should be rarely hit
    # joint_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")})
    #joint_vel_limits = RewardTermCfg(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot")})

    # Manipulability Penalty: Penalize low manipulability
    # manipulability_penalty = RewardTermCfg(
    #     func=mdp.manipulability_penalty,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         # Keep robot in safer, more dexterous configurations
    #         "threshold": 0.05,  # Penalize when manipulability < 0.05
    #     }
    # )

@configclass
class ReachStageRewardsCfgV3:
    distance_l2 = RewardTermCfg(
        func=distance_l2,
        weight=-0.2,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )
    distance_tanh = RewardTermCfg(
        func=distance_tanh,
        weight=0.1,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "std": 0.3,
        },
    )
    orientation_error = RewardTermCfg(
        func=orientation_error,
        weight=-0.1,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )
    success_bonus = RewardTermCfg(
        func=mdp.success_bonus,
        weight=10.0,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": REACH_POSITION_SUCCESS_THRESHOLD,
            "rotation_threshold": REACH_ROTATION_SUCCESS_THRESHOLD,
        },
    )

    action_l2 = RewardTermCfg(func=mdp.action_l2, weight=-0.001)
    action_rate_limit = RewardTermCfg(
        func=action_rate_limit, 
        weight=-0.02, # Gentle discouragement
        # weight=-0.03   # Balanced discouragement
        # weight=-0.05 # Strong smoothness enforcement
        params={
            "threshold_ratio": 0.1,  # 10% of joint vel limits
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"])
        },
    )
    
    # Penalties for joint limits violation - should be rarely hit
    joint_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel_limits = RewardTermCfg(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot")})   
    manipulability = RewardTermCfg(
        func=mdp.manipulability_penalty_linear,
        weight=-1.00,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # Keep robot in safer, more dexterous configurations
            "threshold": 0.02,  # Penalize when manipulability < 0.02
            "soft_ratio": 0.5
        }
    )

    # abnormal_robot = RewardTermCfg(func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"})

@configclass
class IsaacReachStageDenseRewardsCfg:
    """Rewards optimized for reach task."""
    distance_l2 = RewardTermCfg(
        func=distance_l2,
        weight=-0.2,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )
    distance_tanh = RewardTermCfg(
        func=distance_tanh,
        weight=0.1,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "std": 0.1,
        },
    )
    orientation_error = RewardTermCfg(
        func=orientation_error,
        weight=-0.1,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )
    
@configclass
class IsaacReachStageSparseRewardsCfg:
    """Rewards optimized for reach task."""
    distance_l2 = RewardTermCfg(
        func=distance_l2,
        weight=-0.2,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )
    distance_tanh = RewardTermCfg(
        func=distance_tanh,
        weight=0.1,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "std": 0.1,
        },
    )
    orientation_error = RewardTermCfg(
        func=orientation_error,
        weight=-0.1,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )
    success_bonus = RewardTermCfg(
        func=mdp.success_bonus,
        weight=1.0,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": REACH_POSITION_SUCCESS_THRESHOLD,
            "rotation_threshold": REACH_ROTATION_SUCCESS_THRESHOLD,
        },
    )
 
@configclass
class IsaacReachStageHighSparseRewardsCfg:
    """Rewards optimized for reach task."""
    distance_l2 = RewardTermCfg(
        func=distance_l2,
        weight=-0.2,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )
    distance_tanh = RewardTermCfg(
        func=distance_tanh,
        weight=0.1,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "std": 2 * REACH_POSITION_SUCCESS_THRESHOLD,
        },
    )
    orientation_error = RewardTermCfg(
        func=orientation_error,
        weight=-0.1,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )
    success_bonus = RewardTermCfg(
        func=mdp.success_bonus,
        weight=10.0,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": REACH_POSITION_SUCCESS_THRESHOLD,
            "rotation_threshold": REACH_ROTATION_SUCCESS_THRESHOLD,
        },
    )
    
@configclass
class GraspStageRewardsCfg(ReachStageRewardsCfg):
    """Rewards optimized for grasp task."""
    # Inherits all rewards from ReachStageRewardsCfg

    # Modify reach pose bonus weight for grasp stage
    reach_pose_bonus = None 
    distance_to_target = RewardTermCfg(
        func=mdp.distance_l2,
        weight=-2.0,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("object_frame"),
        },
    )

    # Stage 2: Grasping rewards
    grasping_object = RewardTermCfg(
        func=mdp.object_is_between_fingers, # Dense reward
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.01,  # 1cm threshold to encourage grasping
            "open_joint_pos": 0.025,
            "left_finger_body_name": "robotiq_hande_left_finger",
            "right_finger_body_name": "robotiq_hande_right_finger",
            "robot_cfg": SceneEntityCfg("robot", joint_names=["robotiq_hande_left_finger_joint"])
        }  
    )

    # grasp_success_bonus = RewardTermCfg(
    #     func=mdp.grasp_goal_bonus, # Sparse bonus
    #     weight=100.0,
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "threshold": 0.01,  # 1cm threshold for grasp success
    #         "open_joint_pos": 0.025,
    #         "robot_cfg": SceneEntityCfg("robot", joint_names=["robotiq_hande_left_finger_joint"])
    #     }
    # )

@configclass
class LiftStageRewardsCfg:
    """Rewards optimized for lift task."""
    
    reaching_object = RewardTermCfg(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewardTermCfg(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=150.0)
    
    lifting_object_tracking = RewardTermCfg(
        func=mdp.object_lift_height_reward,
        params={"max_height": 0.2, "object_cfg": SceneEntityCfg("object")},
        weight=100.0,  # 10cm lift = 0.10 * 100 = 10 reward points, from 2 cm to 20 cm = 2 to 20 reward points
    )

    object_goal_tracking = RewardTermCfg(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "target_cfg": SceneEntityCfg("target")},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewardTermCfg(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04,  "target_cfg": SceneEntityCfg("target")},
        weight=5.0,
    )

    # action penalty
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class PickPlaceRewardsCfg:
    """Reward terms configuration."""
    
    # Distance Reward
    # Max reward: 0.0 (when distance = 0, right at target)
    # The distance d_goal measures how close the end effector is to the target. A smaller
    # distance means that the end effector is closer to the target, which is desirable to fulfill the task.
    # d_goal = ||obj_pos - target_pos||_2
    # Coarse guidance when far from target
    distance_to_target = RewardTermCfg(
        func=mdp.distance_l2,
        weight=-2.0,  # dist_reward_scale = -2.0
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
        },
    )

    # Rotation Reward
    # r_rot = \frac{1}{|rot_dist| + rot_eps} * rot_reward_scale
    # rot_reward_scale = 1.0, rot_eps = 0.1
    orientation_alignment = RewardTermCfg(
        func=mdp.orientation_alignment_l2,
        weight=0.5,  # rot_reward_scale = 1.0
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "rot_eps": 0.1,
        },
    )

    # Manipulability Penalty: Penalize low manipulability
    manipulability_penalty = RewardTermCfg(
        func=mdp.manipulability_penalty,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # Keep robot in safer, more dexterous configurations
            "threshold": 0.05,  # Penalize when manipulability < 0.05
        }
    )

    # Action Penalties
    # r_act = -( \sum_{t=1}^{6} a_t^2 ) * action_penalty_scale
    # action_penalty_scale = -0.0002
    # Penalize the actions using L2 squared kernel.
    action_penalty = RewardTermCfg(
        func=mdp.action_l2,
        weight=-0.0002,
    )

    # Penalize the rate of change of the actions using L2 squared kernel.
    action_rate_penalty = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.0001, # modified by curriculum
    )

    # Joint Position Limits Penalty
    joint_pos_limits_penalty = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.0001, # modified by curriculum
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Joint Velocity Penalty when near target
    # Encourage slowing down near the target but move fast when far away.
    joint_vel_penalty_near_target = RewardTermCfg(
        func=mdp.joint_velocity_l2_conditional,
        weight=-0.0001,  # modified by curriculum
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "distance_threshold": 0.05,
        },
    )

    # Reach Bonus: Sparse reward when reaching target
    # successTolerance = 0.1
    reach_pose_bonus = RewardTermCfg(
        func=mdp.reach_goal_bonus,
        weight=250.0,  # reach_goal_bonus = 250
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": REACH_POSITION_SUCCESS_THRESHOLD,
            "rotation_threshold": REACH_ROTATION_SUCCESS_THRESHOLD,
        },
    )