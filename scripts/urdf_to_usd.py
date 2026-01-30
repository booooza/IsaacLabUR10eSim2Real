import os
import argparse

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Convert URDF to USD for UR10e with HandE gripper.")
parser.add_argument("--urdf_path", type=str, required=True, help="Path to the URDF file.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the converted USD file.")
parser.add_argument("--usd_file_name", type=str, default="ur10e_with_hande.usd", help="Name of the output USD file.")
#parser.add_argument("--headless", action="store_true", help="Run in headless mode.")

args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli, headless=True)

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg


def convert_urdf_to_usd():
    """Convert URDF to USD file."""

    urdf_path = args_cli.urdf_path
    output_dir = args_cli.output_dir
    usd_file_name = args_cli.usd_file_name

    # Create converter config
    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=output_dir,
        usd_file_name=usd_file_name,
        fix_base=True,
        merge_fixed_joints=False,
        link_density=1000.0,
        self_collision=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="velocity",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                # If None, the stiffness is set to the value parsed from the URDF file.
                stiffness=None,
                # If None, the damping is set to the value parsed from the URDF file or 0.0 if no value is found in the URDF. 
                damping=None,
            ),
        ),
    )
    
    # Convert
    converter = UrdfConverter(cfg)
    print(f"Converted USD saved to: {converter.usd_path}")
    return converter.usd_path

if __name__ == "__main__":
    usd_path = convert_urdf_to_usd()
    print(f"Use this path in your config: {usd_path}")


