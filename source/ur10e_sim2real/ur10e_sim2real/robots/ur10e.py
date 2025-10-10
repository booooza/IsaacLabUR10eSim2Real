"""
UR10e robot with HandE gripper configuration based on custom URDF file.

This is similar to the UR10e with Robotiq gripper configuration from Isaac Lab:
from isaaclab_assets import UR10e_ROBOTIQ_GRIPPER_CFG
"""

import os
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

URDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "ur10e/ur10e_with_hande.urdf"))
print(f"Using URDF path: {URDF_PATH}")

UR10e_HANDE_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        # URDF converter settings are direct parameters of UrdfFileCfg
        fix_base=True,
        merge_fixed_joints=False,
        link_density=1000.0, # Default density in kg/m^3 for links whose "inertial" properties are missing in the URDF. Defaults to 0.0.
        convert_mimic_joints_to_normal_joints=True,
        # Joint drive configuration
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                # If None, the stiffness is set to the value parsed from the URDF file.
                stiffness=None,
                # If None, the damping is set to the value parsed from the URDF file or 0.0 if no value is found in the URDF. 
                damping=None,
            ),
        ),
        # Rigid body and articulation properties
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # UR10e arm joints
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -(np.pi / 2),
            "elbow_joint": -(np.pi / 2),
            "wrist_1_joint": -(np.pi / 2),
            "wrist_2_joint": (np.pi / 2),
            "wrist_3_joint": 0.0,
            # HandE gripper joints (open position)
            "robotiq_hande_left_finger_joint": 0.025,
            "robotiq_hande_right_finger_joint": 0.025,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # UR10e arm actuators
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            effort_limit_sim=330.0,
            velocity_limit_sim=2.0944,  # 120 deg/s
            stiffness=1320.0,
            damping=72.6636085,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=3.1416,  # 180 deg/s
            stiffness=600.0,
            damping=34.64101615,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            effort_limit_sim=54.0,
            velocity_limit_sim=3.1416,  # 180 deg/s
            stiffness=216.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),
        # HandE gripper actuators
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_hande_.*_finger_joint"],
            effort_limit_sim=130.0,
            velocity_limit_sim=0.15,
            stiffness=100.0, # TODO: verify values for HandE gripper
            damping=10.0, # TODO: verify values for HandE gripper
            friction=0.1,
            armature=0.0,
        ),
    },
)
