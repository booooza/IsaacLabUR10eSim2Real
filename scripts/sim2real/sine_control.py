# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--joint-id", type=int, default=0, help="ID of the joint to control.")
parser.add_argument(
    "--max-steps",
    type=int,
    default=1250,
    help="Maximum number of control steps (default: 1250)"
)
parser.add_argument(
    "--amplitude",
    type=float,
    default=0.05,
    help="Amplitude of the joint sine wave in radians (default: π/8=45°)"
)
parser.add_argument(
    "--freq",
    type=float,
    default=0.5,
    help="Frequency of the joint sine wave in Hz (default: 0.5)"
)
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
from isaaclab_tasks.utils import parse_env_cfg

def getJointSineTarget(q_actual, timestep, amplitude=0.05, freq=0.5, joint_id=0, device='cpu'):
    """
    Generate a periodic sinusoid in joint space for testing servoJ (vectorized).
    
    Args:
        q_actual: Current joint positions (num_envs, num_joints) or (num_joints,)
        timestep: Current time in seconds (float)
        amplitude: Amplitude in radians (default: 0.05 rad ≈ 3°)
        freq: Frequency in Hz (default: 0.5)
        joint_id: Which joint to move (int, default: 0)
        device: Torch device (default: 'cpu')
    
    Returns:
        q_target: Target joint positions (same shape as q_actual)
    """
    # Ensure tensor
    if not isinstance(q_actual, torch.Tensor):
        q_actual = torch.tensor(q_actual, dtype=torch.float32, device=device)
    
    # Clone to avoid modifying original
    q_target = q_actual.clone()
    
    # Compute sine wave offset
    offset = amplitude * math.sin(2 * math.pi * freq * timestep)
    
    # Apply to specific joint (handles both 1D and 2D cases)
    if q_target.dim() == 1:
        # Shape: (num_joints,)
        q_target[joint_id] = q_actual[joint_id] + offset
    else:
        # Shape: (num_envs, num_joints)
        q_target[:, joint_id] = q_actual[:, joint_id] + offset
    
    return q_target

def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, _ = env.reset()
    robot = env.unwrapped.scene.articulations['robot']
    q_home = robot.data.joint_pos.mean(dim=0).cpu().numpy()
    logs = []
    step = 0
    time_counter = 0.0
    
    print(f"[INFO]: Initial observation: {obs}")
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            t_loop_start = time.time()
            q_target = getJointSineTarget(q_home, time_counter, amplitude=args_cli.amplitude, freq=args_cli.freq, joint_id=args_cli.joint_id, device=env.unwrapped.device)


            # apply actions
            # Expand to (num_envs, num_joints) shape
            q_target_batched = q_target.unsqueeze(0).repeat(args_cli.num_envs, 1)

            obs, reward, terminated, truncated, info = env.step(q_target_batched)
            
            robot.data.joint_pos # shape (num_envs, num_joints)
            actual_q = robot.data.joint_pos.mean(dim=0).cpu().numpy()
            actual_qd = robot.data.joint_vel.mean(dim=0).cpu().numpy()
            ee_frame = env.unwrapped.scene['ee_frame']
            actual_pose = ee_frame._data.target_pos_source.squeeze(1).mean(dim=0).cpu().numpy()
            
            q_target = q_target.cpu().numpy()
            logs.append({
                "t": time_counter,
                "q_target": q_target.tolist(),
                "q_target_0": float(q_target[args_cli.joint_id]),
                "q_target_1": float(q_target[1]),
                "q_target_2": float(q_target[2]),
                "q_target_3": float(q_target[3]),
                "q_target_4": float(q_target[4]),
                "q_target_5": float(q_target[5]),
                "q_actual": actual_q.tolist(),
                "q_actual_0": float(actual_q[0]),
                "q_actual_1": float(actual_q[1]),
                "q_actual_2": float(actual_q[2]),
                "q_actual_3": float(actual_q[3]),
                "q_actual_4": float(actual_q[4]),
                "q_actual_5": float(actual_q[5]),
                "qd_actual": actual_qd.tolist(),
                "qd_actual_0": float(actual_qd[0]),
                "qd_actual_1": float(actual_qd[1]),
                "qd_actual_2": float(actual_qd[2]),
                "qd_actual_3": float(actual_qd[3]),
                "qd_actual_4": float(actual_qd[4]),
                "qd_actual_5": float(actual_qd[5]),
                "tcp_actual": actual_pose.tolist(),
                "tcp_actual_x": float(actual_pose[0]),
                "tcp_actual_y": float(actual_pose[1]),
                "tcp_actual_z": float(actual_pose[2]),
                # "tcp_actual_rx": actual_pose[3],
                # "tcp_actual_ry": actual_pose[4],
                # "tcp_actual_rz": actual_pose[5],
                "loop_dt": time.time() - t_loop_start,
            })
            time_counter += env_cfg.sim.dt
            step += 1
            if step >= args_cli.max_steps:
                break

    # close the simulator
    env.close()
    with open(f"logs/servo_log_{env_cfg.sim.dt}hz_{args_cli.joint_id}.json", "w") as f:
        json.dump(logs, f, indent=2)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
