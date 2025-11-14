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

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    robot: ArticulationCfg = UR10e_CFG_POSITION_CONTROL.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UR10e_CFG_POSITION_CONTROL.init_state.replace(
            joint_pos={
                # UR10e arm joints
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.5707963267948966, # -90 degrees
                "elbow_joint": 1.5707963267948966, # 90 degrees
                "wrist_1_joint": -1.5707963267948966, # -90 degrees
                "wrist_2_joint": -1.5707963267948966, # -90 degrees
                "wrist_3_joint": 0.0,
            },
        )
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
