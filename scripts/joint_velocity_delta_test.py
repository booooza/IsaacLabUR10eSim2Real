
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
    """360-degree rotation test for each joint with Delta Action Space support."""
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
    
    # Delta action space parameters (adjust these to match your config)
    c_scale = 20.0  # Your DeltaJointVelocityActionCfg c_scale
    action_scale = 1.0  # Since delta uses [-1,1] directly without additional scaling
    
    # Calculate target velocity and ramp-up characteristics
    velocity_command = args_cli.velocity_scale  # User-specified velocity (fraction of max action)
    
    # For delta action spaces, sustained action leads to velocity ramp-up
    # With OI∆: v_target = c_scale * action_command (approximately, after ramp-up)
    # The velocity increases by (c_scale * action * dt) each step
    target_velocity = c_scale * velocity_command  # rad/s (approximate steady-state)
    
    # Calculate ramp-up time to reach ~90% of target velocity
    # For OI∆, velocity ramps up roughly linearly initially
    ramp_up_time = target_velocity / c_scale  # Approximate time to reach target
    ramp_up_steps = int(ramp_up_time * control_frequency * 2)  # Add buffer for settling
    
    # Calculate rotation time and total steps
    rotation_radians = math.radians(args_cli.rotation_degrees)
    time_at_target_velocity = abs(rotation_radians / target_velocity)  # seconds
    steady_state_steps = int(time_at_target_velocity * control_frequency)
    
    # Total steps includes ramp-up and steady-state motion
    steps_per_direction = ramp_up_steps + steady_state_steps
    
    # Add deceleration buffer (to come to stop smoothly)
    deceleration_steps = ramp_up_steps  # Same as ramp-up
    
    if steps_per_direction == 0:
        print("[ERROR]: Calculated steps per direction is 0. Increase --rotation_degrees or --velocity_scale.")
        env.close()
        simulation_app.close()
        return
    
    print("\n" + "="*80)
    print("DETAILED DEBUG INFO")
    print("="*80)

    # 1. Check action manager
    print("\n[1] Action Manager Terms:")
    for term_name, term in env.unwrapped.action_manager._terms.items():
        print(f"\n  Term: '{term_name}'")
        print(f"    Class: {type(term).__name__}")
        print(f"    Action dim: {term.action_dim}")
        print(f"    Joint IDs: {term._joint_ids}")
        print(f"    Joint names: {term._joint_names}")

    # 2. Check total action space
    print(f"\n[2] Total Action Space: {env.action_space.shape}")
    print(f"    Expected: {num_joints} (if only one action term)")

    # 3. Check robot info
    robot = env.unwrapped.scene.articulations['robot']
    print(f"\n[3] Robot Info:")
    print(f"    Total joints: {robot.num_joints}")
    print(f"    Joint names: {robot.joint_names}")

    # 4. Check actuators
    print(f"\n[4] Actuator Groups:")
    for actuator_name, actuator in robot.actuators.items():
        print(f"\n  Actuator: '{actuator_name}'")
        print(f"    Joint IDs: {actuator.joint_indices}")
        print(f"    Joint names: {[robot.joint_names[i] for i in actuator.joint_indices]}")
        print(f"    Stiffness: {actuator.stiffness[0]}")
        print(f"    Damping: {actuator.damping[0]}")

    print("="*80 + "\n")

    print(f"\n[INFO]: Delta Action Space Configuration:")
    print(f"  c_scale: {c_scale}")
    print(f"  Action command: {velocity_command:.2f} (fraction of [-1,1])")
    print(f"  Target velocity: {target_velocity:.2f} rad/s")
    print(f"  Ramp-up steps: {ramp_up_steps}")
    print(f"  Steady-state steps: {steady_state_steps}")
    print(f"  Deceleration steps: {deceleration_steps}")
    print(f"  Total steps per direction: {steps_per_direction}")
    print(f"\n[INFO]: Target Rotation:")
    print(f"  Rotation: {args_cli.rotation_degrees}° ({rotation_radians:.4f} rad)")
    print(f"  Expected time: {(steps_per_direction / control_frequency):.2f}s per direction")
    
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
        for joint_idx in range(num_joints):
            print(f"\n{'='*60}")
            print(f"Testing Joint {joint_idx}: {robot.joint_names[joint_idx]}")
            print(f"{'='*60}")
            
            # Get starting state
            start_pos = robot.data.joint_pos[0].clone()
            start_vel = robot.data.joint_vel[0].clone()
            
            print(f"Initial state (all joints):")
            print(f"  Positions: {start_pos.cpu().numpy()}")
            print(f"  Velocities: {start_vel.cpu().numpy()}")
            
            # Move positive
            for step in range(50):  # Just 50 steps for testing
                with torch.inference_mode():
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    actions[:, joint_idx] = 0.5
                    
                    # CRITICAL DEBUG: Check what's actually being commanded
                    if step == 0:
                        print(f"\nStep 0 - Before env.step():")
                        print(f"  Action vector: {actions[0].cpu().numpy()}")
                        
                    env.step(actions)
                    
                    if step == 0:
                        print(f"Step 0 - After env.step():")
                        # Check what velocity targets were set
                        term = env.unwrapped.action_manager._terms['joint_velocity_deltas']
                        print(f"  Processed actions (velocity targets): {term.processed_actions[0].cpu().numpy()}")
                        # Check actual velocities
                        print(f"  Actual velocities: {robot.data.joint_vel[0].cpu().numpy()}")
                        # Check position changes
                        pos_change = robot.data.joint_pos[0] - start_pos
                        print(f"  Position changes: {pos_change.cpu().numpy()}")
            
            # Check final state
            end_pos = robot.data.joint_pos[0].clone()
            end_vel = robot.data.joint_vel[0].clone()
            pos_delta = end_pos - start_pos
            
            print(f"\nAfter 50 steps:")
            print(f"  Position deltas: {pos_delta.cpu().numpy()}")
            print(f"  Final velocities: {end_vel.cpu().numpy()}")
            print(f"\n  Joint {joint_idx} moved: {pos_delta[joint_idx]:.4f} rad")
            print(f"  Other joints moved: {[f'{i}:{pos_delta[i]:.4f}' for i in range(num_joints) if i != joint_idx]}")
            
            # Stop after first joint for debugging
            break            
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
                    # For delta action spaces, sustained action maintains velocity
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    actions[:, joint_idx] = velocity_command  # Sustained positive command
                    
                    # Apply actions
                    env.step(actions)
                    print(actions)
                    
                    # Print velocity periodically to show ramp-up
                    if step % 25 == 0 or step < 10:
                        current_vel = robot.data.joint_vel[:, joint_idx][0].item()
                        current_pos = robot.data.joint_pos[:, joint_idx][0].item()
                        delta_so_far = current_pos - start_pos
                        print(f"  Step {step:4d}: vel={current_vel:+.3f} rad/s, "
                              f"pos_delta={delta_so_far:+.4f} rad ({math.degrees(delta_so_far):+.2f}°)")
            
            if not simulation_app.is_running():
                break

            # Decelerate to stop smoothly
            print(f"  Decelerating...")
            for step in range(deceleration_steps):
                if not simulation_app.is_running():
                    break
                    
                with torch.inference_mode():
                    # Apply zero action to let velocity decay to zero
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    env.step(actions)
                    print(actions)

            end_pos_positive = robot.data.joint_pos[:, joint_idx][0].item()
            final_vel = robot.data.joint_vel[:, joint_idx][0].item()
            delta_positive = end_pos_positive - start_pos
            print(f"  Phase 1 Complete.")
            print(f"    Delta: {delta_positive:.4f} rad ({math.degrees(delta_positive):.2f}°)")
            print(f"    Final velocity: {final_vel:.3f} rad/s")

            # --- Phase 2: Move in Negative Direction (Return to start) ---
            print(f"\n{'='*60}")
            print(f"[INFO]: Joint {joint_idx} - Moving NEGATIVE: {args_cli.rotation_degrees}° (Return to start)")
            print(f"{'='*60}")

            for step in range(steps_per_direction):
                if not simulation_app.is_running():
                    break
                    
                with torch.inference_mode():
                    # Create action with only this joint moving in negative direction
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    actions[:, joint_idx] = -velocity_command  # Sustained negative command
                    
                    # Apply actions
                    env.step(actions)
                    print(actions)
                    
                    # Print velocity periodically
                    if step % 25 == 0 or step < 10:
                        current_vel = robot.data.joint_vel[:, joint_idx][0].item()
                        current_pos = robot.data.joint_pos[:, joint_idx][0].item()
                        delta_so_far = current_pos - end_pos_positive
                        print(f"  Step {step:4d}: vel={current_vel:+.3f} rad/s, "
                              f"pos_delta={delta_so_far:+.4f} rad ({math.degrees(delta_so_far):+.2f}°)")

            if not simulation_app.is_running():
                break
            
            # Decelerate to stop
            print(f"  Decelerating...")
            for step in range(deceleration_steps):
                if not simulation_app.is_running():
                    break
                    
                with torch.inference_mode():
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    env.step(actions)
                    print(actions)
            
            end_pos_negative = robot.data.joint_pos[:, joint_idx][0].item()
            final_vel = robot.data.joint_vel[:, joint_idx][0].item()
            total_delta = end_pos_negative - start_pos
            print(f"  Phase 2 Complete.")
            print(f"    Net Delta: {total_delta:.4f} rad ({math.degrees(total_delta):.2f}°)")
            print(f"    Final velocity: {final_vel:.3f} rad/s")
            print(f"    Return accuracy: {abs(total_delta):.4f} rad ({math.degrees(abs(total_delta)):.2f}°) error")

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