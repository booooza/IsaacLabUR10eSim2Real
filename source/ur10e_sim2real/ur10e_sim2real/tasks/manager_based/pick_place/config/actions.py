from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.envs.mdp.actions import JointEffortActionCfg, JointVelocityActionCfg, JointPositionActionCfg, BinaryJointPositionActionCfg, BinaryJointVelocityActionCfg, RelativeJointPositionActionCfg, JointPositionToLimitsActionCfg, AbsBinaryJointPositionActionCfg, EMAJointPositionToLimitsActionCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.actions_cfg import DeltaJointVelocityActionCfg

##
# MDP Configurations
##

@configclass
class ReachStageActionsCfg:
    """Actions for reach stage - gripper frozen open."""
    
    # Arm control only
    # Velocity Control - Velocity commands
    joint_velocities = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        scale={
            "shoulder_.*": 2.0, # Maps [-1, 1] to [-2.0, 2.0] rad/s
            "elbow_joint": 3.0, # Maps [-1, 1] to [-3.0, 3.0] rad/s
            "wrist_.*": 3.0, # Maps [-1, 1] to [-3.0, 3.0] rad/s
        },  
        use_default_offset=True,  # Defaults to zero velocity (stationary)
    )
        
    # NO gripper action - will be frozen via events
    
@configclass
class ReachStageDeltaJointVelocityActionsCfg:
    """Actions for reach stage with OI∆ joint velocity."""
    joint_velocity_deltas = DeltaJointVelocityActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        c_scale=20.0,
        clip={
            "shoulder_.*": (-2.0944, 2.0944),
            "elbow_joint": (-3.1416, 3.1416),
            "wrist_.*": (-3.1416, 3.1416),
        }
    )

@configclass
class GraspStageActionsCfg(ReachStageActionsCfg):
    """Actions for grasp stage - gripper control"""
    # Gripper with centered action space
    # Action: [-1, 1] → Gripper: [0, 0.025]
    # -1.0 = closed (0.0), 0.0 = half open (0.0125), 1.0 = fully open (0.025)
    gripper = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["robotiq_hande_left_finger_joint"],
        scale=0.0125,         # Half the range (0.025 / 2)
        offset=0.0125,        # Center at half-open position
        use_default_offset=False,  # Don't use articulation default
    )
    
    # gripper = BinaryJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "robotiq_hande_left_finger_joint",
    #         "robotiq_hande_right_finger_joint"
    #     ],
    #     open_command_expr={
    #         "robotiq_hande_left_finger_joint": 0.025,
    #         "robotiq_hande_right_finger_joint": 0.025,
    #     },
    #     close_command_expr={
    #         "robotiq_hande_left_finger_joint": 0.0,
    #         "robotiq_hande_right_finger_joint": 0.0,
    #     },
    # )
    
    # gripper = AbsBinaryJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "robotiq_hande_left_finger_joint",
    #         "robotiq_hande_right_finger_joint"
    #     ],
    #     open_command_expr={
    #         "robotiq_hande_left_finger_joint": 0.025,
    #         "robotiq_hande_right_finger_joint": 0.025,
    #     },
    #     close_command_expr={
    #         "robotiq_hande_left_finger_joint": 0.0,
    #         "robotiq_hande_right_finger_joint": 0.0,
    #     },
    #     threshold = 0.5,
    #     # The threshold for the binary action.
    #     positive_threshold = True
    #     # Whether to use positive (Open actions > Close actions) threshold. 
    # )

@configclass  
class PickPlaceActionsCfg:
    """Actions for full pick-place - includes gripper."""
    
    # Torque/Effort Control - Direct torque commands
    # joint_efforts = JointEffortActionCfg(
    #     asset_name="robot",
    #     joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
    #                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
    #     scale=1.0,  # Actions are in N·m
    # )

    # Velocity Control - Velocity commands
    # joint_velocities = JointVelocityActionCfg(
    #     asset_name="robot",
    #     joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
    #                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
    #     scale=1.0,  # Actions are in rad/s
    # )

    # Position Control - Position commands
    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        scale=0.5,  # Actions are  absolute positions
        use_default_offset=True,
    )
    
    # Gripper with centered action space
    # Action: [-1, 1] → Gripper: [0, 0.025]
    # -1.0 = closed (0.0), 0.0 = half open (0.0125), 1.0 = fully open (0.025)
    gripper = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["robotiq_hande_left_finger_joint"],
        scale=0.0125,         # Half the range (0.025 / 2)
        offset=0.0125,        # Center at half-open position
        use_default_offset=False,  # Don't use articulation default
    )
