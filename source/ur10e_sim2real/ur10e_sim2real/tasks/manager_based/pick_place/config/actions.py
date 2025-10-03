from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.envs.mdp.actions import JointEffortActionCfg, JointVelocityActionCfg, JointPositionActionCfg, BinaryJointVelocityActionCfg

##
# MDP Configurations
##

@configclass
class ReachStageActionsCfg:
    """Actions for reach stage - gripper frozen open."""
    
    # Arm control only
    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        scale=0.5,
        use_default_offset=True,
    )
    
    # NO gripper action - will be frozen via events

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

    # Position Control - Position commands (or deltas)
    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        scale=0.5,  # Actions are position deltas or absolute positions
        use_default_offset=True,  # Relative to current position
    )
    
    # gripper_action = BinaryJointVelocityActionCfg(
    #     asset_name="robot",
    #     joint_names=["finger_joint"],
    #     open_command_expr={"finger_joint": -0.04},
    #     close_command_expr={"finger_joint": 0.04},
    # )
