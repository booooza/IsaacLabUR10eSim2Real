import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.robots.ur10e import UR10e_HANDE_GRIPPER_CFG
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import UR10e_ROBOTIQ_GRIPPER_CFG

##
# Scene Configuration
##

@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene configuration for pick-and-place task."""
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.793)),  # Lowered to align with table surface
    )
    
    # Table - Thorlabs table around 0.79m high, 0.9m long and 0.6m wide
    # Assuming the robot base is at (0, 0, 0), the table is centered at (0.05, 0, 0)
    # for a workspace of 50 x 50 cm in front of the robot with an offset of 20 cm 
    # from the center of the robot base
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -0.05, 0.0), 
            rot=(0.7071068, 0.0, 0.0, -0.7071068),  # -90° around Z-axis
        ),
    )

    # Robot - UR10e with Hand-E gripper
    robot: ArticulationCfg = UR10e_HANDE_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UR10e_HANDE_GRIPPER_CFG.init_state.replace(
            joint_pos={
                # UR10e arm joints
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.5707963267948966, # -90 degrees
                "elbow_joint": 1.5707963267948966, # 90 degrees
                "wrist_1_joint": -1.5707963267948966, # -90 degrees
                "wrist_2_joint": -1.5707963267948966, # -90 degrees
                "wrist_3_joint": 0.0,
                # HandE gripper joints (open position)
                "robotiq_hande_left_finger_joint": 0.025,
                "robotiq_hande_right_finger_joint": 0.025,
            },
            rot=(0.7071068, 0.0, 0.0, -0.7071068),  # -90° around Z-axis
        )
    )

    # Object to manipulate
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.25, -0.45, 0.00), 
            rot=(1, 0, 0, 0)
        ),
    )

    # Target location marker
    # Consider adding a physical target object like a bin (e.g. Isaac/Props/KLT_Bin/small_KLT.usd)
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,  # Target shouldn't collide
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), 
                opacity=1.0,
                metallic=0.0,
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, -0.45, 0),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    
    # End-effector frame tracker - tracking actual TCP of Robotiq gripper
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",  # base_link is a rigid body
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/FrameTransformer",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
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
    
    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=2500.0
        ),
    )
    