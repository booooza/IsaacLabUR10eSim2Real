"""
Simple kinematics validation script for UR10e.

Tests: Send joint positions → Wait → Measure actual end-effector position vs expected

Usage:
    # Single test
    ./isaaclab.sh -p scripts/validate_kinematics.py --joints 0.0 -1.57 1.57 -1.57 -1.57 0.0

    # Multiple environments (parallel tests with same config)
    ./isaaclab.sh -p scripts/validate_kinematics.py --joints 0.0 -1.57 1.57 -1.57 -1.57 0.0 --num_envs 4
"""

import argparse
import torch
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Simple UR10e Kinematics Validation")
parser.add_argument("--joints", type=float, nargs=6, 
                    default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    help="Target joint positions in radians (6 values)")
parser.add_argument("--num_envs", type=int, default=1, 
                    help="Number of parallel environments")
parser.add_argument("--settle_steps", type=int, default=200,
                    help="Steps to wait for robot to settle")
parser.add_argument("--output_csv", type=str, default="kinematics_validation.csv",
                    help="Output CSV filename")
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
from isaaclab_assets import UR10e_CFG

# Import the analytical FK class
from scripts.kinematics.ur10e_kinematics import UR10eKinematics

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
    
    robot: Articulation = UR10e_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=UR10e_CFG.init_state.replace(
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


def run_validation(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run simple kinematics validation test."""
    
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    analytical_fk = UR10eKinematics()
    
    # Convert commanded joints to tensor
    commanded_joints = torch.tensor([args_cli.joints], device=sim.device)
    commanded_joints = commanded_joints.repeat(scene.num_envs, 1)

    # Compute expected end-effector pose using analytical FK using commanded joints
    joints_np = commanded_joints[0].cpu().numpy()
    expected_ee_pos, expected_ee_rot = analytical_fk.get_ee_pose(joints_np)
    expected_ee_euler = analytical_fk.rotation_matrix_to_euler_xyz(expected_ee_rot)
    
    # Convert to tensors for comparison
    expected_ee_pos_torch = torch.tensor(expected_ee_pos, device=sim.device).unsqueeze(0)
    expected_ee_euler_torch = torch.tensor(expected_ee_euler, device=sim.device).unsqueeze(0)
    
    
    print(f"\n{'='*60}")
    print(f"UR10e Kinematics Validation")
    print(f"{'='*60}")
    print(f"Commanded joints (rad): {args_cli.joints}")
    print(f"Commanded joints (deg): {[f'{j*57.2958:.2f}' for j in args_cli.joints]}")
    print(f"Number of environments: {scene.num_envs}")
    print(f"Settle steps: {args_cli.settle_steps}")
    print(f"{'='*60}\n")
    
    # Print initial joint positions
    initial_joints = robot.data.joint_pos[:, :6].clone()
    print(f"Initial joint positions (deg): {torch.rad2deg(initial_joints[0]).cpu().numpy()}\n")
    
    # Step 1: Send joint position commands
    print("Sending joint position commands...")
    robot.set_joint_position_target(commanded_joints)
    robot.write_data_to_sim()
    
    # Step 2: Wait until settled (with progress monitoring)
    print(f"Waiting {args_cli.settle_steps} steps for robot to settle...")
    
    for step in range(args_cli.settle_steps):
        sim.step()
        scene.update(sim_dt)
        
        # Print progress every 50 steps
        if step % 50 == 0 and step > 0:
            current_joints = robot.data.joint_pos[:, :6]
            current_error = torch.rad2deg(torch.abs(current_joints[0] - commanded_joints[0]))
            max_error = torch.max(current_error).item()
            print(f"  Step {step}/{args_cli.settle_steps} - Max error: {max_error:.2f}°")
    
    # Step 3: Measure actual joint positions
    actual_joints = robot.data.joint_pos[:, :6].clone()
    
    # Step 4: Get actual end-effector position
    ee_body_idx = robot.find_bodies("wrist_3_link")[0][0]
    ee_pose_w = robot.data.body_pose_w[:, ee_body_idx, :]
    actual_ee_position = ee_pose_w[:, 0:3]  # xyz position
    actual_ee_quat_sim = ee_pose_w[:, 3:7]  # quaternion
    actual_ee_euler_sim = euler_xyz_from_quat(actual_ee_quat_sim)
    actual_ee_euler_sim = torch.stack(actual_ee_euler_sim, dim=1)  # stacks as columns [num_envs, 3]
    

    # Step 5: Calculate errors
    joint_errors = actual_joints - commanded_joints
    joint_errors_deg = torch.rad2deg(joint_errors)
    
    # Position error: Isaac Sim vs Analytical FK
    position_error = actual_ee_position - expected_ee_pos_torch.repeat(scene.num_envs, 1)
    position_error_norm = torch.norm(position_error, dim=1)
    
    # Orientation error: Isaac Sim vs Analytical FK
    orientation_error = actual_ee_euler_sim - expected_ee_euler_torch.repeat(scene.num_envs, 1)
    # Handle angle wrapping
    orientation_error = torch.atan2(torch.sin(orientation_error), torch.cos(orientation_error))
    orientation_error_norm = torch.norm(orientation_error, dim=1)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    
    converged_count = 0
    fk_match_count = 0
    
    for env_idx in range(scene.num_envs):
        max_joint_err = torch.max(torch.abs(joint_errors_deg[env_idx])).item()
        pos_err_mm = position_error_norm[env_idx].item() * 1000
        orient_err_deg = torch.rad2deg(orientation_error_norm[env_idx]).item()
        
        print(f"\nEnvironment {env_idx}:")
        print(f"  Commanded joints (deg): {torch.rad2deg(commanded_joints[env_idx]).cpu().numpy()}")
        print(f"  Actual joints (deg):    {torch.rad2deg(actual_joints[env_idx]).cpu().numpy()}")
        print(f"  Joint errors (deg):     {joint_errors_deg[env_idx].cpu().numpy()}")
        print(f"  Max joint error: {max_joint_err:.4f}°")
        
        # Check joint convergence
        if max_joint_err < 1.0:
            print(f"  Joint Status: ✓ CONVERGED (error < 1°)")
            converged_count += 1
        elif max_joint_err < 5.0:
            print(f"  Joint Status: ⚠ CLOSE (error < 5°)")
        else:
            print(f"  Joint Status: ✗ NOT CONVERGED (error > 5°)")
        
        print(f"\n  === Forward Kinematics Comparison ===")
        print(f"  (Commanded joints used for analytical FK)")
        print(f"  Isaac Sim EE pos (mm):  {actual_ee_position[env_idx].cpu().numpy() * 1000}")
        print(f"  Analytical EE pos (mm): {expected_ee_pos * 1000}")
        print(f"  Position error (mm):    {pos_err_mm:.3f}")
        
        print(f"\n  Isaac Sim EE euler (deg):  {torch.rad2deg(actual_ee_euler_sim[env_idx]).cpu().numpy()}")
        print(f"  Analytical EE euler (deg): {np.rad2deg(expected_ee_euler)}")
        print(f"  Orientation error (deg):   {orient_err_deg:.3f}")
        
        # Check FK match
        if pos_err_mm < 1.0 and orient_err_deg < 1.0:
            print(f"  FK Status: ✓ MATCH (Isaac Sim ≈ Analytical FK)")
            fk_match_count += 1
        else:
            print(f"  FK Status: ✗ MISMATCH")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Joint convergence: {converged_count}/{scene.num_envs} environments")
    print(f"  FK validation:     {fk_match_count}/{scene.num_envs} environments match")
    print(f"{'='*60}")
    
    # Step 6: Save to CSV
    save_to_csv(commanded_joints, actual_joints, joint_errors, 
                actual_ee_position, expected_ee_pos_torch, position_error)
    
    print(f"\nResults appended to {args_cli.output_csv}")
    print(f"{'='*60}\n")


def save_to_csv(commanded, actual, joint_errors, ee_pos_sim, ee_pos_analytical, ee_error):
    """Append results to CSV file."""
    
    csv_file = Path(args_cli.output_csv)
    file_exists = csv_file.exists()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            writer.writerow([
                'timestamp',
                'cmd_j1', 'cmd_j2', 'cmd_j3', 'cmd_j4', 'cmd_j5', 'cmd_j6',
                'actual_j1', 'actual_j2', 'actual_j3', 'actual_j4', 'actual_j5', 'actual_j6',
                'err_j1_deg', 'err_j2_deg', 'err_j3_deg', 'err_j4_deg', 'err_j5_deg', 'err_j6_deg',
                'max_joint_err_deg',
                'ee_sim_x', 'ee_sim_y', 'ee_sim_z',
                'ee_analytical_x', 'ee_analytical_y', 'ee_analytical_z',
                'ee_err_x', 'ee_err_y', 'ee_err_z',
                'ee_err_norm_mm',
                'converged',
                'fk_match'
            ])
        
        # Write data for each environment
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        num_envs = commanded.shape[0]
        
        for env_idx in range(num_envs):
            cmd = commanded[env_idx].cpu().numpy()
            act = actual[env_idx].cpu().numpy()
            err = torch.rad2deg(joint_errors[env_idx]).cpu().numpy()
            max_err = torch.max(torch.abs(torch.rad2deg(joint_errors[env_idx]))).item()
            
            ee_sim = ee_pos_sim[env_idx].cpu().numpy()
            ee_ana = ee_pos_analytical[env_idx].cpu().numpy()
            ee_e = ee_error[env_idx].cpu().numpy()
            ee_norm_mm = torch.norm(ee_error[env_idx]).item() * 1000
            
            converged = 1 if max_err < 1.0 else 0
            fk_match = 1 if ee_norm_mm < 1.0 else 0
            
            row = [timestamp]
            row.extend(cmd)
            row.extend(act)
            row.extend(err)
            row.append(max_err)
            row.extend(ee_sim)
            row.extend(ee_ana)
            row.extend(ee_e)
            row.append(ee_norm_mm)
            row.append(converged)
            row.append(fk_match)
            
            writer.writerow(row)


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
    
    # Run validation
    run_validation(sim, scene)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
