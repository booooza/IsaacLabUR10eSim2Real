
"""Script to test 360-degree rotation for each joint individually."""
"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="joint rotation test for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="UR10e-Sim2Real-PickPlace-v0", help="Name of the task.")
parser.add_argument("--rotation_degrees", type=float, default=360.0, help="Degrees to rotate each joint.")
parser.add_argument("--velocity_scale", type=float, default=1.0, help="Velocity command scale (fraction of max).")
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
    """360-degree rotation test for each joint with Isaac Lab environment."""
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
    
    # Calculate steps needed for desired rotation (This calculation is crucial for 360-degree accuracy)
    action_scale = 2.0  # From your ReachStageActionsCfg
    velocity_command = args_cli.velocity_scale  # User-specified velocity (fraction of max)
    actual_velocity = velocity_command * action_scale  # rad/s
    
    # Use a full 360 degrees for a complete rotation cycle demonstration
    rotation_degrees = 360.0 # Use 360 for a full revolution
    rotation_radians = math.radians(rotation_degrees)
    
    # Since the user might pass a smaller degree value, we'll respect 'args_cli.rotation_degrees'
    # but use 360 degrees for the *full cycle* demonstration, or just use the passed value.
    # We will stick to the logic of the original script and use the passed value for consistency.
    rotation_radians_test = math.radians(args_cli.rotation_degrees)
    time_needed = rotation_radians_test / abs(actual_velocity)  # seconds
    steps_per_direction = int(time_needed * control_frequency)
    
    if steps_per_direction == 0:
        print("[ERROR]: Calculated steps per direction is 0. Increase --rotation_degrees or decrease --velocity_scale.")
        env.close()
        simulation_app.close()
        return

    print(f"\n[INFO]: Infinite Rotation Configuration:")
    print(f"  Target rotation: {args_cli.rotation_degrees}° ({rotation_radians_test:.4f} rad)")
    print(f"  Actual velocity: {actual_velocity:.2f} rad/s")
    print(f"  Steps per direction: {steps_per_direction}")
    
    # reset environment
    obs, _ = env.reset()
    robot = env.unwrapped.scene.articulations['robot']
    
    # --- Main Infinite Loop ---
    cycle_count = 0
    
    while simulation_app.is_running():
        
        cycle_count += 1
        print(f"\n\n{'#'*80}")
        print(f"  CYCLE {cycle_count} STARTING: Looping through all {num_joints} joints.")
        print(f"{'#'*80}")

        # Iterate through each joint
        # We use a simple forward iteration this time, as the primary goal is infinite motion.
        for joint_idx in range(num_joints):
        
            
            # Get current joint position before movement
            current_joint_pos = robot.data.joint_pos[:, joint_idx].clone()
            start_pos = current_joint_pos[0].item()
            
            # --- Phase 1: Move in Positive Direction ---
            print(f"\n{'='*60}")
            print(f"[INFO]: Joint {joint_idx} - Moving POSITIVE: {args_cli.rotation_degrees}°")
            print(f"{'='*60}")

            for step in range(steps_per_direction):
                if not simulation_app.is_running():
                    break
                    
                with torch.inference_mode():
                    # Create action with only this joint moving in positive direction
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    actions[:, joint_idx] = args_cli.velocity_scale  # positive velocity delta
                    
                    # Apply actions
                    env.step(actions)
            
            if not simulation_app.is_running():
                break

            end_pos_positive = robot.data.joint_pos[:, joint_idx][0].item()
            delta_positive = end_pos_positive - start_pos
            print(f"  Phase 1 Complete. Delta: {delta_positive:.4f} rad ({math.degrees(delta_positive):.2f}°)")

            # --- Phase 2: Move in Negative Direction (Reverse) ---
            print(f"\n{'='*60}")
            print(f"[INFO]: Joint {joint_idx} - Moving NEGATIVE: {args_cli.rotation_degrees}° (Return to start)")
            print(f"{'='*60}")

            for step in range(steps_per_direction):
                if not simulation_app.is_running():
                    break
                    
                with torch.inference_mode():
                    # Create action with only this joint moving in negative direction
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    actions[:, joint_idx] = -args_cli.velocity_scale  # negative velocity delta
                    
                    # Apply actions
                    env.step(actions)

            if not simulation_app.is_running():
                break
            
            end_pos_negative = robot.data.joint_pos[:, joint_idx][0].item()
            total_delta = end_pos_negative - start_pos
            print(f"  Phase 2 Complete. Net Delta: {total_delta:.4f} rad ({math.degrees(total_delta):.2f}°)")


        if not simulation_app.is_running():
            break

    print(f"\n{'='*80}")
    print(f"[INFO]: Simulation stopped by user after {cycle_count} full cycles. Closing environment.")
    print(f"{'='*80}")
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()