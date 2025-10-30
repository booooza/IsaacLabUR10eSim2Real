# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import torch
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logs.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import math
import os
import random
import time

import gymnasium as gym
import ur10e_sim2real.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, rew, dones, extras = env.step(actions)
            # logging
            if args_cli.debug:
                log(log_dir, env.unwrapped.scene, env.unwrapped.common_step_counter, dt, env.unwrapped.episode_length_buf)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

def log(log_dir: str, scene: InteractiveScene, step_count: int, dt: float, episode_length_buf: torch.Tensor):
    """Log environment information during play."""
    csv_file = Path(log_dir) / "play_log.csv"
    file_exists = csv_file.exists()
    
    # Get robot data
    robot = scene.articulations['robot']
    joint_names = robot._data.joint_names
    num_envs = robot._data.joint_pos.shape[0]
    joint_pos = robot._data.joint_pos  # (num_envs, 8)
    joint_pos_target = robot._data.joint_pos_target  # (num_envs, 8)
    joint_pos_limits = robot._data.joint_pos_limits  # (num_envs, 8, 2)
    soft_joint_pos_limits = robot._data.soft_joint_pos_limits  # (num_envs, 8, 2)
    
    joint_vel = robot._data.joint_vel  # (num_envs, 8)
    joint_vel_target = robot._data.joint_vel_target  # (num_envs, 8)
    joint_vel_limits = robot._data.joint_vel_limits  # (num_envs, 8)
    soft_joint_vel_limits = robot._data.soft_joint_vel_limits  # torch.Size([1, 8])
    
    # Calculate simulation time from step counter
    sim_time = step_count * dt
    
    # Get end effector data
    ee_frame_transform = scene['ee_frame']
    ee_frame_pos = ee_frame_transform.data.target_pos_source  # torch.Size([num_envs, 1, 3])
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            header = ['step', 'sim_time', 'env_id', 'episode_step']
            # Add joint position columns
            for name in joint_names:
                header.append(f'{name}_pos')
            # Add joint position target columns
            for name in joint_names:
                header.append(f'{name}_pos_target')
            # Add joint velocity columns
            for name in joint_names:
                header.append(f'{name}_vel')
            # Add joint velocity target columns
            for name in joint_names:
                header.append(f'{name}_vel_target')
            # Add joint position limits (lower and upper)
            for name in joint_names:
                header.append(f'{name}_pos_limit_lower')
                header.append(f'{name}_pos_limit_upper')
            # Add soft joint position limits (lower and upper)
            for name in joint_names:
                header.append(f'{name}_soft_pos_limit_lower')
                header.append(f'{name}_soft_pos_limit_upper')
            # Add joint velocity limits
            for name in joint_names:
                header.append(f'{name}_vel_limit')
            # Add soft joint velocity limits (lower and upper)
            for name in joint_names:
                header.append(f'{name}_soft_vel_limit')
            # Add end effector position columns
            header.extend(['ee_frame_pos_x', 'ee_frame_pos_y', 'ee_frame_pos_z'])
            writer.writerow(header)
        
        # Write data for each environment
        for env_id in range(num_envs):
            row = [step_count, sim_time, env_id, episode_length_buf[env_id].item()]
            
            # Add joint positions
            row.extend(joint_pos[env_id].cpu().numpy().tolist())
            # Add joint position targets
            row.extend(joint_pos_target[env_id].cpu().numpy().tolist())
            # Add joint velocities
            row.extend(joint_vel[env_id].cpu().numpy().tolist())
            # Add joint velocity targets
            row.extend(joint_vel_target[env_id].cpu().numpy().tolist())
            # Add joint position limits (lower, upper for each joint)
            for joint_idx in range(len(joint_names)):
                row.append(joint_pos_limits[env_id, joint_idx, 0].item())  # lower limit
                row.append(joint_pos_limits[env_id, joint_idx, 1].item())  # upper limit
            # Add soft joint position limits (lower, upper for each joint)
            for joint_idx in range(len(joint_names)):
                row.append(soft_joint_pos_limits[env_id, joint_idx, 0].item())  # soft lower limit
                row.append(soft_joint_pos_limits[env_id, joint_idx, 1].item())  # soft upper limit
            # Add joint velocity limits
            row.extend(joint_vel_limits[env_id].cpu().numpy().tolist())
            # Add soft joint velocity limits (symmetric, so ±soft_vel_limit)
            row.extend(soft_joint_vel_limits[env_id].cpu().numpy().tolist())
            # Add end effector frame positions
            row.extend([
                ee_frame_pos[env_id, 0, 0].item(),
                ee_frame_pos[env_id, 0, 1].item(),
                ee_frame_pos[env_id, 0, 2].item()
            ])
            
            writer.writerow(row)
          
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
