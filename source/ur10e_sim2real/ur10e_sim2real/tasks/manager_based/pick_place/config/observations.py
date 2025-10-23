from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.managers import ObservationTermCfg
from isaaclab.managers import ObservationGroupCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

import torch

# Import MDP functions from isaaclab
from isaaclab.envs.mdp import (
    joint_pos_rel,
    joint_pos_limit_normalized,
    joint_vel_rel,
    last_action,
)

# Import custom observation functions 
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.observations import (
    distance_to_object,
    relative_position_from_scene_entity,
    relative_rotation_from_scene_entity,
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
                "asset_cfg": SceneEntityCfg("robot", joint_names=["robotiq_hande_left_finger_joint"]) 
            },
            # noise=GaussianNoise(mean=0.0, std=0.01),  # Gripper noise
        )
        
        # End-effector state
        tcp_position_base  = ObservationTermCfg(
            func=relative_position_from_scene_entity,
            params={"asset_cfg": SceneEntityCfg("ee_frame")},
        )

        tcp_rotation_base  = ObservationTermCfg(
            func=relative_rotation_from_scene_entity,
            params={"asset_cfg": SceneEntityCfg("ee_frame")},
        )

        # Target object state (assumption: perfectly known)
        target_position_base  = ObservationTermCfg(
            func=relative_position_from_scene_entity,
            params={"asset_cfg": SceneEntityCfg("target_frame")},
        )

        target_rotation_base  = ObservationTermCfg(
            func=relative_rotation_from_scene_entity,
            params={"asset_cfg": SceneEntityCfg("target_frame")},
        )

        tcp_to_target_rotation = ObservationTermCfg(
            func=mdp.relative_rotation_from_scene_entity,
            params={"asset_cfg": SceneEntityCfg("ee_target_frame")},
        )

        # Task-specific object (ground-truth simulation state)
        object_position_base  = ObservationTermCfg(
            func=relative_position_from_scene_entity,
            params={"asset_cfg": SceneEntityCfg("object_frame")},
            # noise=GaussianNoise(mean=0.0, std=0.01),  # Vision noise
        )

        object_rotation_base  = ObservationTermCfg(
            func=relative_rotation_from_scene_entity,
            params={"asset_cfg": SceneEntityCfg("object_frame")},
            # noise=GaussianNoise(mean=0.0, std=0.01),  # Vision noise
        )

        # Object Awareness: TCP rotation relative to object and target
        tcp_to_object_rotation = ObservationTermCfg(
            func=mdp.relative_rotation_from_scene_entity,
            params={"asset_cfg": SceneEntityCfg("ee_object_frame")},
            # noise=GaussianNoise(mean=0.0, std=0.01),  # Vision noise
        )
        
        # Distance to target
        distance_to_object = ObservationTermCfg(
            func=distance_to_object,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            }
        )

        # Previous actions
        prev_actions = ObservationTermCfg(func=last_action)
        
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
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    