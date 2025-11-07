"""
Simple UR10e Joint Movement Test

Tests each joint individually to verify all 6 joints can move.

Usage:
    ./isaaclab.sh -p scripts/test_joints.py
"""

import torch
from isaaclab.app import AppLauncher
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Test UR10e Joint Movements")
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
from source.ur10e_sim2real.ur10e_sim2real.robots.ur10e import UR10e_HANDE_GRIPPER_CFG


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Simple scene with UR10e robot."""
    
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    robot: Articulation = UR10e_HANDE_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


def test_joint_movements(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Test each joint individually."""
    
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    
    joint_names = [
        "shoulder_pan_joint",    # Joint 0
        "shoulder_lift_joint",   # Joint 1
        "elbow_joint",           # Joint 2
        "wrist_1_joint",         # Joint 3
        "wrist_2_joint",         # Joint 4
        "wrist_3_joint",         # Joint 5
    ]
    
    print(f"\n{'='*60}")
    print(f"UR10e JOINT MOVEMENT TEST")
    print(f"{'='*60}\n")
    
    # Starting position (all zeros)
    home_position = torch.zeros(robot.num_joints, device=sim.device).unsqueeze(0)
    
    # Test each joint
    for joint_idx in range(6):
        joint_name = joint_names[joint_idx]
        
        print(f"\n--- Testing {joint_name} (Joint {joint_idx}) ---")
        
        # Reset to home
        print("  Returning to home position...")
        robot.set_joint_position_target(home_position)
        robot.write_data_to_sim()
        for _ in range(100):  # Wait to settle
            sim.step()
            scene.update(sim_dt)
        
        # Move joint to +45 degrees
        print(f"  Moving to +45° (+0.785 rad)...")
        target_position = home_position.clone()
        target_position[0, joint_idx] = 0.785  # 45 degrees in radians
        
        robot.set_joint_position_target(target_position)
        robot.write_data_to_sim()
        
        for step in range(150):
            sim.step()
            scene.update(sim_dt)
            
            if step % 50 == 0:
                current_pos = robot.data.joint_pos[0, joint_idx].item()
                print(f"    Step {step}: Position = {current_pos:.3f} rad ({current_pos*57.3:.1f}°)")
        
        # Check final position
        final_pos = robot.data.joint_pos[0, joint_idx].item()
        target_pos = 0.785
        error = abs(final_pos - target_pos)
        
        if error < 0.01:  # 0.01 rad ≈ 0.57 degrees
            print(f"  ✓ SUCCESS: Reached {final_pos:.3f} rad (error: {error*57.3:.2f}°)")
        else:
            print(f"  ✗ FAILED: Only reached {final_pos:.3f} rad (error: {error*57.3:.2f}°)")
        
        # Move joint to -45 degrees
        print(f"  Moving to -45° (-0.785 rad)...")
        target_position[0, joint_idx] = -0.785
        
        robot.set_joint_position_target(target_position)
        robot.write_data_to_sim()
        
        for step in range(150):
            sim.step()
            scene.update(sim_dt)
            
            if step % 50 == 0:
                current_pos = robot.data.joint_pos[0, joint_idx].item()
                print(f"    Step {step}: Position = {current_pos:.3f} rad ({current_pos*57.3:.1f}°)")
        
        # Check final position
        final_pos = robot.data.joint_pos[0, joint_idx].item()
        target_pos = -0.785
        error = abs(final_pos - target_pos)
        
        if error < 0.01:
            print(f"  ✓ SUCCESS: Reached {final_pos:.3f} rad (error: {error*57.3:.2f}°)")
        else:
            print(f"  ✗ FAILED: Only reached {final_pos:.3f} rad (error: {error*57.3:.2f}°)")
    
    # Final test: Move all joints together
    print(f"\n--- Testing All Joints Together ---")
    print("  Moving all joints to 30° (0.524 rad)...")
    
    target_position = torch.ones(robot.num_joints, device=sim.device).unsqueeze(0) * 0.524
    robot.set_joint_position_target(target_position)
    robot.write_data_to_sim()
    
    for step in range(200):
        sim.step()
        scene.update(sim_dt)
        
        if step % 50 == 0:
            current_pos = robot.data.joint_pos[0, :6]
            print(f"  Step {step}: Joints = {[f'{p*57.3:.1f}°' for p in current_pos.cpu().numpy()]}")
    
    final_positions = robot.data.joint_pos[0, :6]
    errors = torch.abs(final_positions - 0.524)
    max_error = torch.max(errors).item()
    
    print(f"\n  Final positions: {[f'{p*57.3:.1f}°' for p in final_positions.cpu().numpy()]}")
    print(f"  Max error: {max_error*57.3:.2f}°")
    
    if max_error < 0.01:
        print(f"  ✓ ALL JOINTS WORKING!")
    else:
        print(f"  ✗ Some joints did not reach target")
    
    print(f"\n{'='*60}")
    print(f"TEST COMPLETE")
    print(f"{'='*60}\n")


def main():
    """Main function."""
    
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.5])
    
    # Setup scene
    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play simulator
    sim.reset()
    
    print("[INFO]: Setup complete...")
    
    # Run test
    test_joint_movements(sim, scene)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
