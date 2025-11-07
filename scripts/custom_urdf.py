import argparse
import torch
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Simple custom UR10e robot")
parser.add_argument("--num_envs", type=int, default=1, 
                    help="Number of parallel environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

from source.ur10e_sim2real.ur10e_sim2real.robots.ur10e import UR10e_HANDE_GRIPPER_CFG

@configclass
class ValidationSceneCfg(InteractiveSceneCfg):
    """Simple scene with UR10e robot."""
    
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    robot: Articulation = UR10e_HANDE_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UR10e_HANDE_GRIPPER_CFG.init_state.replace(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.5707963267948966,
                "elbow_joint": -1.5707963267948966,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 1.5707963267948966,
                "wrist_3_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    robot.spawn.rigid_props.disable_gravity = True


def main():
    """Main function."""
    
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])
    
    # Setup scene
    scene_cfg = ValidationSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play simulator
    sim.reset()

    # Reset to known home position
    robot = scene["robot"]
    home_joints = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(home_joints, robot.data.default_joint_vel)
    robot.reset()

    print("[INFO]: Setup complete...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
