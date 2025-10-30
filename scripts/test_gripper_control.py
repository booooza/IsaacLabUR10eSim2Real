"""Script to test gripper control (last 2 actions) from -1 to +1."""
"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Gripper control test for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="UR10e-Sim2Real-Lift-v0", help="Name of the task.")
parser.add_argument("--steps", type=int, default=100, help="Number of steps to go from -1 to +1.")
parser.add_argument("--cycles", type=int, default=5, help="Number of cycles to repeat (0 for infinite).")
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
import numpy as np
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Gripper control test sweeping from -1 to +1."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    
    # get number of actions from action space
    num_actions = env.action_space.shape[-1]
    print(f"[INFO]: Number of actions: {num_actions}")
    print(f"[INFO]: Gripper actions (last 2): indices {num_actions-2}, {num_actions-1}")
    
    # Get simulation parameters from config
    decimation = env_cfg.decimation
    physics_dt = env_cfg.sim.dt
    control_dt = decimation * physics_dt
    control_frequency = 1.0 / control_dt
    
    print(f"\n[INFO]: Simulation Configuration:")
    print(f"  Physics dt: {physics_dt:.4f}s ({1/physics_dt:.0f} Hz)")
    print(f"  Decimation: {decimation}")
    print(f"  Control dt: {control_dt:.4f}s ({control_frequency:.1f} Hz)")
    print(f"  Steps for -1 to +1: {args_cli.steps}")
    print(f"  Total time per sweep: {args_cli.steps * control_dt:.2f}s")
    
    # reset environment
    obs, _ = env.reset()
    
    # Create smooth transition values from -1 to +1
    gripper_values = np.linspace(-1.0, 1.0, args_cli.steps)
    
    # --- Main Loop ---
    cycle_count = 0
    infinite_mode = (args_cli.cycles == 0)
    
    print(f"\n[INFO]: Starting gripper control test")
    print(f"[INFO]: Mode: {'Infinite' if infinite_mode else f'{args_cli.cycles} cycles'}")
    
    while simulation_app.is_running():
        if not infinite_mode and cycle_count >= args_cli.cycles:
            break
            
        cycle_count += 1
        print(f"\n{'#'*80}")
        print(f"  CYCLE {cycle_count}: Sweeping gripper from -1 to +1")
        print(f"{'#'*80}")
        
        # --- Phase 1: Sweep from -1 to +1 ---
        print(f"\n[Phase 1]: Moving gripper from -1.0 to +1.0 ({args_cli.steps} steps)")
        
        for step_idx, gripper_value in enumerate(gripper_values):
            if not simulation_app.is_running():
                break
                
            with torch.inference_mode():
                # Create action tensor with zeros for all joints
                actions = torch.zeros((args_cli.num_envs, num_actions), device=env.unwrapped.device)
                
                # Set last 2 actions (gripper) to the current value
                actions[:, -2] = gripper_value
                actions[:, -1] = gripper_value
                
                # Apply actions
                env.step(actions)
                
                # Print progress periodically
                if (step_idx + 1) % (args_cli.steps // 10) == 0 or step_idx == 0 or step_idx == args_cli.steps - 1:
                    progress = (step_idx + 1) / args_cli.steps * 100
                    print(f"  Step {step_idx+1}/{args_cli.steps} ({progress:.0f}%): Gripper value = {gripper_value:+.3f}")
        
        if not simulation_app.is_running():
            break
        
        # --- Phase 2: Sweep from +1 to -1 (reverse) ---
        print(f"\n[Phase 2]: Moving gripper from +1.0 to -1.0 ({args_cli.steps} steps)")
        
        for step_idx, gripper_value in enumerate(reversed(gripper_values)):
            if not simulation_app.is_running():
                break
                
            with torch.inference_mode():
                # Create action tensor with zeros for all joints
                actions = torch.zeros((args_cli.num_envs, num_actions), device=env.unwrapped.device)
                
                # Set last 2 actions (gripper) to the current value
                actions[:, -2] = gripper_value
                actions[:, -1] = gripper_value
                
                # Apply actions
                env.step(actions)
                
                # Print progress periodically
                if (step_idx + 1) % (args_cli.steps // 10) == 0 or step_idx == 0 or step_idx == args_cli.steps - 1:
                    progress = (step_idx + 1) / args_cli.steps * 100
                    print(f"  Step {step_idx+1}/{args_cli.steps} ({progress:.0f}%): Gripper value = {gripper_value:+.3f}")
        
        if not simulation_app.is_running():
            break
        
        print(f"\n[INFO]: Cycle {cycle_count} complete!")
    
    print(f"\n{'='*80}")
    print(f"[INFO]: Test completed after {cycle_count} cycles. Closing environment.")
    print(f"{'='*80}")
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    