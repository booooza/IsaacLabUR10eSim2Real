import isaaclab.sim as sim_utils
import os
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.robots.ur10e import UR10e_CFG, UR10e_HANDE_GRIPPER_CFG, URDF_UR10e_BASE_PATH, URDF_UR10e_HANDE_PATH
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.reach.scene import ReachSceneCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class GraspSceneCfg(ReachSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=URDF_UR10e_HANDE_PATH,
            # URDF converter settings are direct parameters of UrdfFileCfg
            fix_base=True,
            merge_fixed_joints=False,
            # mimic joint bug!
            convert_mimic_joints_to_normal_joints=True,
            link_density=1000.0, # Default density in kg/m^3 for links whose "inertial" properties are missing in the URDF. Defaults to 0.0.
            collider_type='mesh',
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,   # Smaller = tighter closure
                rest_offset=0.0005,     # Half of contact_offset
            ),
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
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                # https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup_tutorials/tutorial_configure_manipulator.html
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.00005,
                stabilization_threshold=0.00001
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # UR10e arm joints
                    "shoulder_pan_joint": 0.0,
                    "shoulder_lift_joint": -1.10,
                    "elbow_joint": 2.0,
                    "wrist_1_joint": -2.5,
                    "wrist_2_joint": -1.57,
                    "wrist_3_joint": 0.0,
                    # HandE gripper joints (open position)
                    "robotiq_hande_left_finger_joint": 0.025,
                    "robotiq_hande_right_finger_joint": 0.025,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            # UR10e arm actuators
            # Sources:
            # [1] https://www.universal-robots.com/articles/ur/robot-care-maintenance/max-joint-torques-cb3-and-e-series/
            # [2] https://www.universal-robots.com/media/1807466/ur10e_e-series_datasheets_web.pdf
            # [3] https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/universal_robots.py
            # [4] https://robotiq.com/products/adaptive-grippers#Hand-E
            "shoulder_pan": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint"],
                effort_limit_sim=330.0, # Size 4: 330 Nm [1]
                velocity_limit_sim=2.0944,  # 120 deg/s [2]
                stiffness=50.0,
                damping=5.0,
                friction=0.0,
                armature=0.0,
            ),
            "shoulder_lift": ImplicitActuatorCfg(
                    joint_names_expr=["shoulder_lift_joint"],
                    effort_limit_sim=330.0, # Size 4: 330 Nm [1]
                    velocity_limit_sim=2.0944,  # 120 deg/s [2]
                    stiffness=40.0,
                    damping=4.0,
                    friction=0.0,
                    armature=0.0,
                ),
            "elbow": ImplicitActuatorCfg(
                    joint_names_expr=["elbow_joint"],
                    effort_limit_sim=150.0, # Size 3: 150 Nm [1]
                    velocity_limit_sim=3.1416,  # 180 deg/s [2]
                    stiffness=50.0,
                    damping=5.0,
                    friction=0.0,
                    armature=0.0,
                ),
            "wrist_1": ImplicitActuatorCfg(
                    joint_names_expr=["wrist_1_joint"],
                    effort_limit_sim=54.0, # Size 2: 54 Nm [1]
                    velocity_limit_sim=3.1416,  # 180 deg/s [2]
                    stiffness=60.0,
                    damping=6.0,
                    friction=0.0,
                    armature=0.0,
                ),
            "wrist_2": ImplicitActuatorCfg(
                    joint_names_expr=["wrist_2_joint"],
                    effort_limit_sim=54.0, # Size 2: 54 Nm [1]
                    velocity_limit_sim=3.1416,  # 180 deg/s [2]
                    stiffness=50.0,
                    damping=5.0,
                    friction=0.0,
                    armature=0.0,
                ),
            "wrist_3": ImplicitActuatorCfg(
                    joint_names_expr=["wrist_3_joint"],
                    effort_limit_sim=54.0, # Size 2: 54 Nm [1]
                    velocity_limit_sim=3.1416,  # 180 deg/s [2]
                    stiffness=50.0,
                    damping=5.0,
                    friction=0.0,
                    armature=0.0,
                ),
            "gripper_left": ImplicitActuatorCfg(
                    joint_names_expr=["robotiq_hande_left_finger_joint"],
                    effort_limit_sim=185.0, # 20–185 N per finger ​[4]
                    velocity_limit_sim=0.15, # 20-150 mm/s ​[4]
                    # if this is too low, the gripper can't hold an object
                    stiffness=2000.0, # p-gain
                    damping=100.0, # d-gain
                    # friction=0.1,
                    # armature=0.001,
                ),
            "gripper_right": ImplicitActuatorCfg(
                    joint_names_expr=["robotiq_hande_right_finger_joint"],
                    effort_limit_sim=185.0, # 20–185 N per finger ​[4]
                    velocity_limit_sim=0.15, # 20-150 mm/s ​[4]
                    # if this is too low, the gripper can't hold an object
                    stiffness=2000.0, # p-gain
                    damping=100.0, # d-gain
                    # friction=0.1,
                    # armature=0.001,
                ),
        },
    )
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_hande_.*_finger",  # Regex for finger links
        update_period=0.0,  # Update every physics step
        debug_vis=True,      # Visualize contacts in simulator
        history_length=1     # Store latest contact data
    )
    # Object to manipulate - randomized cubes
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg( 
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                    kinematic_enabled=False, # disable kinematics for reaching
                ),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.74687, 0.17399, 0.015), # Center of the workspace (60cm in front of robot base)
            rot=(1, 0, 0, 0)
        ),
    )
    # End-effector frame tracker - tracking UR10e TCP 
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/FrameTransformer",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.05, 0.05, 0.05),
                ),
            }
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/robotiq_hande_end",
                name="end_effector",
            ),
        ],
    )
    # Object frame tracker - tracking the object position wrt robot base
    object_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                name="object",
            ),
        ],
        debug_vis=False,  # Set to True to visualize
        visualizer_cfg=FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/FrameTransformer",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.05, 0.05, 0.05),
                ),
            }
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=2500.0
        ),
    )
