import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.robots.ur10e import UR10e_CFG, UR10e_CFG_POSITION_CONTROL
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuator
from source.ur10e_sim2real.ur10e_sim2real.robots.ur10e import UR10e_CFG, UR10e_HANDE_GRIPPER_CFG, URDF_UR10e_BASE_PATH, URDF_UR10e_HANDE_PATH

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=URDF_UR10e_BASE_PATH,
            # URDF converter settings are direct parameters of UrdfFileCfg
            fix_base=True,
            merge_fixed_joints=False,
            link_density=1000.0, # Default density in kg/m^3 for links whose "inertial" properties are missing in the URDF. Defaults to 0.0.
            # Joint drive configuration
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="velocity",
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
                enabled_self_collisions=True,
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
                "shoulder_lift_joint": -1.57, # -90 degrees
                "elbow_joint": 1.57, # 90 degrees
                "wrist_1_joint": -1.57, # -90 degrees
                "wrist_2_joint": -1.57, # -90 degrees
                "wrist_3_joint": 0.0,
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
                stiffness=0.0,
                damping=625.0,
                friction=0.0,
                armature=0.0,
            ),
            "shoulder_lift": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_lift_joint"],
                effort_limit_sim=330.0, # Size 4: 330 Nm [1]
                velocity_limit_sim=2.0944,  # 120 deg/s [2]
                stiffness=0.0,
                damping=625.0,
                friction=0.0,
                armature=0.0,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                effort_limit_sim=150.0, # Size 3: 150 Nm [1]
                velocity_limit_sim=3.1416,  # 180 deg/s [2]
                stiffness=0.0,
                damping=625.0,
                friction=0.0,
                armature=0.0,
            ),
            "wrist_1": ImplicitActuatorCfg(
                joint_names_expr=["wrist_1_joint"],
                effort_limit_sim=54.0, # Size 2: 54 Nm [1]
                velocity_limit_sim=3.1416,  # 180 deg/s [2]
                stiffness=0.0,
                damping=625.0,
                friction=0.0,
                armature=0.0,
            ),
            "wrist_2": ImplicitActuatorCfg(
                joint_names_expr=["wrist_2_joint"],
                effort_limit_sim=54.0, # Size 2: 54 Nm [1]
                velocity_limit_sim=3.1416,  # 180 deg/s [2]
                stiffness=0.0,
                damping=625.0,
                friction=0.0,
                armature=0.0,
            ),
            "wrist_3": ImplicitActuatorCfg(
                joint_names_expr=["wrist_3_joint"],
                effort_limit_sim=54.0, # Size 2: 54 Nm [1]
                velocity_limit_sim=3.1416,  # 180 deg/s [2]
                stiffness=0.0,
                damping=625.0,
                friction=0.0,
                armature=0.0,
            ),
        },
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
                prim_path="{ENV_REGEX_NS}/Robot/tool0",
                name="end_effector",
            ),
        ],
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=2500.0
        ),
    )
