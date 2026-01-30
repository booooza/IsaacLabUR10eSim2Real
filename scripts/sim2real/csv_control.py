"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import time
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--max-steps",
    type=int,
    default=1861,
    help="Maximum number of control steps (default: 1861)"
)
parser.add_argument(
    "--csv",
    type=str,
    default="data.csv",
    help="Output CSV filename (default: targets.csv)"
)
parser.add_argument(
    "--control",
    type=str,
    default="position",
    help="Control mode: position or velocity (default: position)"
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
import pandas as pd
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
import numpy as np
from scipy.spatial.transform import Rotation

def urdf_to_dh(urdf_pose):
    """
    Convert URDF convention to real robot TCP (DH convention).
    """
    pos = np.array(urdf_pose[:3])
    rot_vec = np.array(urdf_pose[3:])
    
    # Position: flip X and Y signs
    real_pos = np.array([-pos[0], -pos[1], pos[2]])
    
    # Orientation: apply Rz(-180°) transformation
    R_urdf = Rotation.from_rotvec(rot_vec)
    R_transform = Rotation.from_euler('z', -180, degrees=True)
    R_real = R_transform * R_urdf
    
    real_rot_vec = R_real.as_rotvec()
    
    return np.concatenate([real_pos, real_rot_vec])

def main():
    """Zero actions agent with Isaac Lab environment."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    df_targets = pd.read_csv(args_cli.csv)
    steps = df_targets.shape[0]
    duration_s = steps * env_cfg.sim.dt
    env_cfg.episode_length_s = duration_s
    env_cfg.randomize_joints = False
    env_cfg.scene.robot.init_state.joint_pos = {					
                # UR10e arm joints				
                "shoulder_pan_joint": -0.034638,
                "shoulder_lift_joint": -1.914078,
                "elbow_joint": -1.920668,
                "wrist_1_joint": -0.860232,
                "wrist_2_joint": 1.572753,
                "wrist_3_joint": -0.820956,
                # "robotiq_hande_left_finger_joint": 0.025,
                # "robotiq_hande_right_finger_joint": 0.025,
            }

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
            # Check if we've reached the end of the CSV data
            if step >= len(df_targets) or step >= args_cli.max_steps:
                print(f"[INFO]: Reached end of targets at step {step}")
                break
            
            # Read target positions and velocities from CSV for current step
            row = df_targets.iloc[step]
            pose_targets = torch.tensor([
                row['target_tcp_pos_x'],
                row['target_tcp_pos_y'],
                row['target_tcp_pos_z'],
                row['target_tcp_rot_rx'],
                row['target_tcp_rot_ry'],
                row['target_tcp_rot_rz']
            ], device=env.unwrapped.device, dtype=torch.float32)
            # Extract target joint positions
            q_target = torch.tensor([
                row['target_joint_0_pos'],
                row['target_joint_1_pos'],
                row['target_joint_2_pos'],
                row['target_joint_3_pos'],
                row['target_joint_4_pos'],
                row['target_joint_5_pos']
            ], device=env.unwrapped.device, dtype=torch.float32)
            # Extract target joint velocities
            qd_target = torch.tensor([
                row['target_joint_0_vel'],
                row['target_joint_1_vel'],
                row['target_joint_2_vel'],
                row['target_joint_3_vel'],
                row['target_joint_4_vel'],
                row['target_joint_5_vel']
            ], device=env.unwrapped.device, dtype=torch.float32)
            
            if args_cli.control == "position":
                actions = q_target.repeat(env.unwrapped.num_envs, 1)
            elif args_cli.control == "velocity":
                actions = qd_target.repeat(env.unwrapped.num_envs, 1)
            else:
                raise ValueError(f"Unknown control mode: {args_cli.control}")

            # apply actions
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # logging
            robot.data.joint_pos # shape (num_envs, num_joints)
            actual_q = robot.data.joint_pos.mean(dim=0).cpu().numpy()
            actual_qd = robot.data.joint_vel.mean(dim=0).cpu().numpy()
            ee_frame = env.unwrapped.scene['ee_frame']
            ee_pos = ee_frame._data.target_pos_source.squeeze(1).mean(dim=0).cpu().numpy()
            ee_quat = ee_frame._data.target_quat_source.squeeze(1).mean(dim=0).cpu().numpy()
            ee_pose = np.concatenate([ee_pos, ee_quat])
            # Position: flip X and Y signs (URDF to DH)
            actual_pose = np.array([-ee_pos[0], -ee_pos[1], ee_pos[2]])
            
            pose_targets = pose_targets.cpu().numpy()
            q_target = q_target.cpu().numpy()
            qd_target = qd_target.cpu().numpy()
            logs.append({
                "timestamp": time_counter,
                "target_joint_pos": q_target.tolist(),
                "target_joint_0_pos": float(q_target[0]),
                "target_joint_1_pos": float(q_target[1]),
                "target_joint_2_pos": float(q_target[2]),
                "target_joint_3_pos": float(q_target[3]),
                "target_joint_4_pos": float(q_target[4]),
                "target_joint_5_pos": float(q_target[5]),
                "joint_pos": actual_q.tolist(),
                "joint_0_pos": float(actual_q[0]),
                "joint_1_pos": float(actual_q[1]),
                "joint_2_pos": float(actual_q[2]),
                "joint_3_pos": float(actual_q[3]),
                "joint_4_pos": float(actual_q[4]),
                "joint_5_pos": float(actual_q[5]),
                "target_joint_vel": qd_target.tolist(),
                "target_joint_0_vel": float(qd_target[0]),
                "target_joint_1_vel": float(qd_target[1]),
                "target_joint_2_vel": float(qd_target[2]),
                "target_joint_3_vel": float(qd_target[3]),
                "target_joint_4_vel": float(qd_target[4]),
                "target_joint_5_vel": float(qd_target[5]),
                "joint_vel": actual_qd.tolist(),
                "joint_0_vel": float(actual_qd[0]),
                "joint_1_vel": float(actual_qd[1]),
                "joint_2_vel": float(actual_qd[2]),
                "joint_3_vel": float(actual_qd[3]),
                "joint_4_vel": float(actual_qd[4]),
                "joint_5_vel": float(actual_qd[5]),
                "target_tcp_pos_x": float(pose_targets[0]),
                "target_tcp_pos_y": float(pose_targets[1]),
                "target_tcp_pos_z": float(pose_targets[2]),
                "target_tcp_rot_rx": float(pose_targets[3]),
                "target_tcp_rot_ry": float(pose_targets[4]),
                "target_tcp_rot_rz": float(pose_targets[5]),
                "tcp_pos": actual_pose.tolist(),
                "tcp_pos_x": float(actual_pose[0]),
                "tcp_pos_y": float(actual_pose[1]),
                "tcp_pos_z": float(actual_pose[2]),
                # "tcp_rot_rx": actual_pose[3],
                # "tcp_rot_ry": actual_pose[4],
                # "tcp_rot_rz": actual_pose[5],
                "system_time": time.time() - t_loop_start,
            })
            time_counter += env_cfg.sim.dt
            step += 1
            if step >= args_cli.max_steps:
                break

    
    with open(f"logs/isaac_log.json", "w") as f:
        json.dump(logs, f, indent=2)
        
    df = pd.DataFrame(logs)

    
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs = axs.flatten()
    for joint_id in range(6):
        axs[joint_id].plot(df["timestamp"], df[f"joint_{joint_id}_vel"], label="actual")
        axs[joint_id].plot(df["timestamp"], df[f"target_joint_{joint_id}_vel"], label="target")
        axs[joint_id].set_xlabel("Time (s)")
        axs[joint_id].set_ylabel("Joint velocity (rad/s)")
        axs[joint_id].set_title(f"Joint {joint_id}: Target vs Actual")
        axs[joint_id].legend()
        axs[joint_id].grid(True)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/joints_vel_plot.png")
    plt.show(block=False)

    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs = axs.flatten()

    for joint_id in range(6):
        axs[joint_id].plot(df["timestamp"], df[f"joint_{joint_id}_pos"], label="actual")
        axs[joint_id].plot(df["timestamp"], df[f"target_joint_{joint_id}_pos"], label="target")
        axs[joint_id].set_xlabel("Time (s)")
        axs[joint_id].set_ylabel("Joint position (rad)")
        axs[joint_id].set_title(f"Joint {joint_id}: Target vs Actual")
        axs[joint_id].legend()
        axs[joint_id].grid(True)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/joints_pos_plot.png")
    plt.show(block=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Extract XYZ columns
    x_t, y_t, z_t = df["target_tcp_pos_x"], df["target_tcp_pos_y"], df["target_tcp_pos_z"]
    x_a, y_a, z_a = df["tcp_pos_x"], df["tcp_pos_y"], df["tcp_pos_z"]
    
    # Plot target trajectory
    ax.plot(x_t, y_t, z_t, label='TCP Target', linewidth=2)
    # Plot actual trajectory
    ax.plot(x_a, y_a, z_a, label='TCP Actual', linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("TCP Target vs TCP Actual (3D Trajectory)")

    plt.show(block=False)
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs = axs.flatten()
    for i, pos in enumerate(['x', 'y', 'z']):
        axs[2*i].plot(df["timestamp"], df[f"target_tcp_pos_{pos}"], label=f"Target {pos.upper()}")
        axs[2*i].plot(df["timestamp"], df[f"tcp_pos_{pos}"], label=f"Actual {pos.upper()}")
        axs[2*i].set_xlabel("Time (s)")
        axs[2*i].set_ylabel(f"TCP Position {pos.upper()} (m)")
        axs[2*i].set_title(f"TCP Position {pos.upper()}: Target vs Actual")
        axs[2*i].legend()
        axs[2*i].grid(True) 

        axs[2*i+1].plot(df["timestamp"], np.sqrt((df[f"tcp_pos_{pos}"] - df[f"target_tcp_pos_{pos}"])**2), label="RMS Error")
        axs[2*i+1].set_xlabel("Time (s)")
        axs[2*i+1].set_ylabel(f"TCP Position {pos.upper()} Error (m)")
        axs[2*i+1].set_title(f"TCP Position {pos.upper()}: RMS Error")
        axs[2*i+1].legend()
        axs[2*i+1].grid(True)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    plt.show(block=True)

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
