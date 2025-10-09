from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.managers import ObservationTermCfg
from isaaclab.managers import ObservationGroupCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

import torch

# Import MDP functions from isaaclab
from isaaclab.envs.mdp import (
    joint_pos,
    joint_pos_rel,
    joint_pos_limit_normalized,
    joint_vel_rel,
    root_pos_w,
    root_lin_vel_w,
    last_action,
)

# Import custom observation functions 
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.observations import (
    distance_to_object,
    ee_pose,
    ee_velocity,
    ee_to_object_vector,
    object_to_target_vector,
)


@configclass
class PickPlaceObservationsCfg:
    """Observation specifications with asymmetric actor-critic."""
    
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Actor observations (with noise)."""
        
        # Proprioception
        joint_pos_norm = ObservationTermCfg(
            func=joint_pos_limit_normalized,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
                )
            },
            # noise=GaussianNoise(mean=0.0, std=0.001),  # Encoder noise
        )
        
        gripper_state = ObservationTermCfg(
            func=joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint"]) 
            },
            # noise=GaussianNoise(mean=0.0, std=0.01),  # Gripper noise
        )
        
        # Task-specific (with noise)
        object_pose = ObservationTermCfg(
            func=root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object")},
            # noise=GaussianNoise(mean=0.0, std=0.01),  # Position noise
        )
        
        target_pose = ObservationTermCfg(
            func=root_pos_w, 
            params={"asset_cfg": SceneEntityCfg("target")},
            # noise=GaussianNoise(mean=0.0, std=0.01),
        )
        
        # Previous actions
        prev_actions = ObservationTermCfg(func=last_action)
        
        # Distance to target
        distance_to_object = ObservationTermCfg(
            func=distance_to_object,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(PolicyCfg):
        """Critic observations (privileged, no noise)."""
        
        # Velocities (privileged)
        joint_vel = ObservationTermCfg(
            func=joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"])},
        )
        
        # End-effector state
        ee_pose = ObservationTermCfg(
            func=ee_pose,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        
        ee_vel = ObservationTermCfg(
            func=ee_velocity,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        
        object_vel = ObservationTermCfg(
            func=root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        
        # Relative transforms
        ee_to_object = ObservationTermCfg(
            func=ee_to_object_vector,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            }
        )
        
        object_to_target = ObservationTermCfg(
            func=object_to_target_vector,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "target_cfg": SceneEntityCfg("target"),
            }
        )        
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    