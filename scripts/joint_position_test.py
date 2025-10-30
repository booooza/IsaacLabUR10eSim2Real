# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to test 360-degree rotation for each joint individually using relative position control."""
"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="360-degree joint rotation test with relative position control.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--rotation_degrees", type=float, default=360.0, help="Degrees to rotate each joint.")
parser.add_argument("--steps_per_rotation", type=int, default=200, help="Number of steps for full rotation.")
parser.add_argument("--action_scale", type=float, default=0.1, help="Action scale from environment config.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import ur10e_sim2real.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import torch
import math
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """360-degree rotation test for each joint with relative position control."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # get number of joints from action space
    num_joints = env.action_space.shape[-1]
    print(f"[INFO]: Number of joints: {num_joints}")
    
    # Get simulation parameters from config
    decimation = env_cfg.decimation
    physics_dt = env_cfg.sim.dt
    control_dt = decimation * physics_dt
    control_frequency = 1.0 / control_dt
    
    print(f"\n[INFO]: Simulation Configuration:")
    print(f"  Physics dt: {physics_dt:.4f}s ({1/physics_dt:.0f} Hz)")
    print(f"  Decimation: {decimation}")
    print(f"  Control dt: {control_dt:.4f}s ({control_frequency:.1f} Hz)")
    
    # Calculate position delta per step
    rotation_radians = math.radians(args_cli.rotation_degrees)
    steps_per_rotation = args_cli.steps_per_rotation
    
    # IMPORTANT: Account for the action scale
    # The environment will apply: actual_delta = action * scale
    # So we need: action = desired_delta / scale
    position_delta_radians = rotation_radians / steps_per_rotation
    action_value = position_delta_radians / args_cli.action_scale
    
    print(f"\n[INFO]: Rotation Configuration:")
    print(f"  Target rotation: {args_cli.rotation_degrees}° ({rotation_radians:.4f} rad)")
    print(f"  Steps per rotation: {steps_per_rotation}")
    print(f"  Action scale (from env): {args_cli.action_scale}")
    print(f"  Desired position delta per step: {position_delta_radians:.6f} rad ({math.degrees(position_delta_radians):.4f}°)")
    print(f"  Action value to send: {action_value:.6f}")
    print(f"  Total time per rotation: {steps_per_rotation * control_dt:.2f}s")
    
    # Sanity check
    if abs(action_value) > 1.0:
        print(f"\n[WARNING]: Action value {action_value:.3f} exceeds [-1, 1] range!")
        print(f"           Consider increasing --steps_per_rotation or --action_scale")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            env.close()
            simulation_app.close()
            return
    
    # reset environment
    obs, _ = env.reset()
    
    # Get robot reference
    robot = env.unwrapped.scene.articulations['robot']
    
    # Get initial joint positions for tracking
    initial_joint_pos = robot.data.joint_pos.clone()
    
    print(f"\n[INFO]: Initial joint positions (rad):")
    for i in range(num_joints):
        pos_rad = initial_joint_pos[0, i].item()
        pos_deg = math.degrees(pos_rad)
        print(f"  Joint {i}: {pos_rad:7.4f} rad ({pos_deg:8.2f}°)")
    
    print(f"\n[INFO]: Using RELATIVE position deltas")
    print(f"         Actions are incremental changes from current position")
    print(f"         Actions are scaled by {args_cli.action_scale}")
    
    # simulate environment - test each joint individually from last to first
    total_steps = 0
    
    for joint_idx in range(num_joints - 1, -1, -1):  # iterate from last to first joint
        print(f"\n{'='*80}")
        print(f"[INFO]: Testing Joint {joint_idx}")
        print(f"{'='*80}")
        
        # Get current joint position before movement
        start_pos = robot.data.joint_pos[:, joint_idx][0].item()
        
        # Move in positive direction
        print(f"\n[INFO]: Phase 1 - Moving in POSITIVE direction")
        print(f"  Starting position: {start_pos:.4f} rad ({math.degrees(start_pos):.2f}°)")
        
        for step in range(steps_per_rotation):
            if not simulation_app.is_running():
                break
                
            with torch.inference_mode():
                # Create action with scaled action value for this joint only
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                actions[:, joint_idx] = action_value  # positive action value (will be scaled in env)
                
                # apply actions
                obs, reward, terminated, truncated, info = env.step(actions)
                total_steps += 1
                
                # Progress reporting
                if step % 50 == 0 or step == steps_per_rotation - 1:
                    current_pos = robot.data.joint_pos[:, joint_idx][0].item()
                    delta = current_pos - start_pos
                    expected_delta = (step + 1) * position_delta_radians
                    error = delta - expected_delta
                    print(f"  Step {step:4d}/{steps_per_rotation}: "
                          f"pos={current_pos:7.4f} rad ({math.degrees(current_pos):8.2f}°) | "
                          f"Δ={delta:7.4f} rad ({math.degrees(delta):8.2f}°) | "
                          f"error={error:7.4f} rad ({math.degrees(error):7.2f}°)")
        
        if not simulation_app.is_running():
            break
        
        # Get position after positive movement
        end_pos_positive = robot.data.joint_pos[:, joint_idx][0].item()
        delta_positive = end_pos_positive - start_pos
        expected_delta = rotation_radians
        error_positive = delta_positive - expected_delta
        
        print(f"\n  Phase 1 Complete:")
        print(f"    Total movement: {delta_positive:.4f} rad ({math.degrees(delta_positive):.2f}°)")
        print(f"    Expected: {expected_delta:.4f} rad ({math.degrees(expected_delta):.2f}°)")
        print(f"    Error: {error_positive:.4f} rad ({math.degrees(error_positive):.2f}°)")
        if abs(expected_delta) > 0.001:
            print(f"    Error %: {100*abs(error_positive)/expected_delta:.2f}%")
        
        # Move in negative direction
        print(f"\n[INFO]: Phase 2 - Moving in NEGATIVE direction")
        start_pos_negative = end_pos_positive
        print(f"  Starting position: {start_pos_negative:.4f} rad ({math.degrees(start_pos_negative):.2f}°)")
        
        for step in range(steps_per_rotation):
            if not simulation_app.is_running():
                break
                
            with torch.inference_mode():
                # Create action with negative scaled action value for this joint only
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                actions[:, joint_idx] = -action_value  # negative action value
                
                # apply actions
                obs, reward, terminated, truncated, info = env.step(actions)
                total_steps += 1
                
                # Progress reporting
                if step % 50 == 0 or step == steps_per_rotation - 1:
                    current_pos = robot.data.joint_pos[:, joint_idx][0].item()
                    delta = current_pos - start_pos_negative
                    expected_delta = -(step + 1) * position_delta_radians
                    error = delta - expected_delta
                    print(f"  Step {step:4d}/{steps_per_rotation}: "
                          f"pos={current_pos:7.4f} rad ({math.degrees(current_pos):8.2f}°) | "
                          f"Δ={delta:7.4f} rad ({math.degrees(delta):8.2f}°) | "
                          f"error={error:7.4f} rad ({math.degrees(error):7.2f}°)")
        
        if not simulation_app.is_running():
            break
        
        # Get final position after negative movement
        end_pos_negative = robot.data.joint_pos[:, joint_idx][0].item()
        delta_negative = end_pos_negative - start_pos_negative
        total_delta = end_pos_negative - start_pos
        expected_delta_neg = -rotation_radians
        error_negative = delta_negative - expected_delta_neg
        
        print(f"\n  Phase 2 Complete:")
        print(f"    Total movement: {delta_negative:.4f} rad ({math.degrees(delta_negative):.2f}°)")
        print(f"    Expected: {expected_delta_neg:.4f} rad ({math.degrees(expected_delta_neg):.2f}°)")
        print(f"    Error: {error_negative:.4f} rad ({math.degrees(error_negative):.2f}°)")
        if abs(expected_delta_neg) > 0.001:
            print(f"    Error %: {100*abs(error_negative)/abs(expected_delta_neg):.2f}%")
        
        print(f"\n  Full Cycle Summary:")
        print(f"    Net movement (should be ~0): {total_delta:.4f} rad ({math.degrees(total_delta):.2f}°)")
        if abs(rotation_radians) > 0.001:
            print(f"    Return accuracy: {100*(1 - abs(total_delta)/rotation_radians):.2f}%")
    
    print(f"\n{'='*80}")
    print(f"[INFO]: Test completed successfully!")
    print(f"  Total control steps executed: {total_steps}")
    print(f"  Total simulation time: {total_steps * control_dt:.2f}s")
    print(f"{'='*80}")
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()