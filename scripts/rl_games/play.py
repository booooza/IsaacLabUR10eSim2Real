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
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run per environment.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export model to ONNX format.")
parser.add_argument("--onnx_path", type=str, default=None, help="Path to save ONNX model.")

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
import onnx

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
from scripts.rl_games.logger import PlayLogger
from rl_games.algos_torch import flatten

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
    
    # initialize the custom logger
    logger = PlayLogger(log_dir) if args_cli.debug else None

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
    
    # Export to ONNX if requested
    if args_cli.export_onnx:
        # Determine save path
        if args_cli.onnx_path is not None:
            onnx_save_path = args_cli.onnx_path
        else:
            onnx_save_path = os.path.join(log_dir, "exported_model.onnx")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)
        
        # Get a sample observation
        sample_obs = env.reset()
        if isinstance(sample_obs, dict):
            sample_obs = sample_obs["obs"]
        
        # Convert to agent format
        sample_obs = agent.obs_to_torch(sample_obs)
        
        # Export the model
        try:
            export_to_onnx(agent, sample_obs, onnx_save_path)
            export_actor_to_onnx(agent, sample_obs, onnx_save_path.replace(".onnx", "_actor.onnx"))
            print(f"[INFO] Exported model to: {onnx_save_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to export or test ONNX model: {e}")
            import traceback
            traceback.print_exc()

    dt = env.unwrapped.step_dt
    max_episode_length = env.unwrapped.max_episode_length

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    episodes_completed = torch.zeros(env.unwrapped.num_envs, dtype=torch.int32, device=env.unwrapped.device)
    max_episodes = args_cli.num_episodes
    timestep = 0
    
    print(f"[INFO] Running {max_episodes} episode(s) per environment")
    print(f"[INFO] Episode length: {max_episode_length} steps ({max_episode_length * dt:.1f} seconds)")

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
        # Check if all environments have completed the required number of episodes
        if (episodes_completed >= max_episodes).all():
            print(f"[INFO] All environments completed {max_episodes} episode(s). Exiting.")
            break
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, rew, dones, extras = env.step(actions)
            # logging
            #if args_cli.debug:
                #log(log_dir, env.unwrapped.scene, env.unwrapped.common_step_counter, dt, env.unwrapped.episode_length_buf)

            if args_cli.debug and logger is not None:
                logger.log_step(
                    env.unwrapped,
                    dt,
                    episodes_completed
                )

            # perform operations for terminated episodes
            if len(dones) > 0:
                # Count completed episodes
                episodes_completed[dones] += 1
                
                # Print progress
                for env_id in dones.nonzero(as_tuple=True)[0]:
                    if episodes_completed[env_id] <= max_episodes:
                        print(f"[INFO] Env {env_id.item()}: Episode {episodes_completed[env_id].item()}/{max_episodes} completed")
                    
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        timestep += 1
        if args_cli.video:
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.debug and logger is not None:
        logger.save(overwrite=True)
    
    # close the simulator
    env.close()

def export_to_onnx(agent, obs, save_path, opset_version=11):
    """Export the agent's model to ONNX format.
    
    Args:
        agent: The RL-Games agent/player
        obs: Sample observation to trace the model
        save_path: Path where to save the ONNX model
        opset_version: ONNX opset version to use
    """
    print(f"[INFO] Exporting model to ONNX format...")
    
    # Get the model from the agent
    model = agent.model
    model.eval()
    
    # Prepare the observation
    if isinstance(obs, dict):
        obs_tensor = obs["obs"]
    else:
        obs_tensor = obs
    
    # Convert to torch tensor if needed
    if not isinstance(obs_tensor, torch.Tensor):
        obs_tensor = torch.from_numpy(obs_tensor)
    
    # Take a single observation (batch size 1) for export
    inputs = {
        'obs' : torch.zeros((1,) + agent.obs_shape).to(agent.device),
        'rnn_states' : agent.states,
    }
    
    class ModelWrapper(torch.nn.Module):
        '''
        Main idea is to ignore outputs which we don't need from model
        '''
        def __init__(self, model):
            torch.nn.Module.__init__(self)
            self._model = model
            
            
        def forward(self,input_dict):
            input_dict['obs'] = self._model.norm_obs(input_dict['obs'])
            '''
            just model export doesn't work. Looks like onnx issue with torch distributions
            thats why we are exporting only neural network
            '''
            #print(input_dict)
            #output_dict = self._model.a2c_network(input_dict)
            #input_dict['is_train'] = False
            #return output_dict['logits'], output_dict['values']
            return self._model.a2c_network(input_dict)
    
    with torch.no_grad():
        adapter = flatten.TracingAdapter(ModelWrapper(agent.model), inputs, allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
        flattened_outputs = traced(*adapter.flattened_inputs)
        print(flattened_outputs)
    
    print("[INFO] Exporting feedforward model...")
    
    try:
        # Export with the wrapped model
        torch.onnx.export(
            traced,
            *adapter.flattened_inputs,
            save_path,
            input_names=['obs'],
            output_names=['mu','log_std', 'value'],
            opset_version=opset_version,
            export_params=True
        )
        
        print(f"[INFO] Model exported to: {save_path}")
        
        # Verify the exported model
        try:
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print("[INFO] ONNX model verification passed!")
            
            # Print model info
            print(f"[INFO] Model inputs: {[input.name for input in onnx_model.graph.input]}")
            print(f"[INFO] Model outputs: {[output.name for output in onnx_model.graph.output]}")
            
            
        except Exception as e:
            print(f"[WARNING] ONNX model verification failed: {e}")
        
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return save_path

def export_actor_to_onnx(agent, obs, save_path, opset_version=11):
    """Export only the actor/policy to ONNX format.
    
    Args:
        agent: The RL-Games agent/player
        obs: Sample observation to trace the model
        save_path: Path where to save the ONNX model
        opset_version: ONNX opset version to use
    """
    print(f"[INFO] Exporting actor/policy to ONNX format...")
    
    # Get the model from the agent
    model = agent.model
    model.eval()
    
    # Prepare the observation
    if isinstance(obs, dict):
        obs_tensor = obs["obs"]
    else:
        obs_tensor = obs
    
    # Convert to torch tensor if needed
    if not isinstance(obs_tensor, torch.Tensor):
        obs_tensor = torch.from_numpy(obs_tensor)
    
    inputs = {
        'obs' : torch.zeros((1,) + agent.obs_shape).to(agent.device),
        'rnn_states' : agent.states,
    }

    # Wrapper that includes normalization
    class ActorWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, obs):
            # Convert tensor input to the dict format RL-Games expects
            input_dict = {
                'obs': self.model.norm_obs(obs),
                'is_train': False
            }
            output = self.model(input_dict)
            
            # Return deterministic actions (mus)
            if isinstance(output, dict):
                return output.get('mus', output.get('actions'))
            else:
                return output[0] if isinstance(output, (tuple, list)) else output
            
    with torch.no_grad():
        adapter = flatten.TracingAdapter(ActorWrapper(agent.model), inputs, allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
        flattened_outputs = traced(*adapter.flattened_inputs)
        
    print("[INFO] Exporting actor network with normalization...")
    
    try:
        torch.onnx.export(
            traced,
            *adapter.flattened_inputs,
            save_path,
            input_names=['obs'],
            output_names=['actions'],
            dynamic_axes={
                'obs': {0: 'batch'},
                'actions': {0: 'batch'}
            },
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True
        )
        
        print(f"[INFO] Actor exported to: {save_path}")
        
        # Verify the exported model
        try:
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print("[INFO] ONNX model verification passed!")
            
            # Print model info
            print(f"[INFO] Model inputs: {[input.name for input in onnx_model.graph.input]}")
            print(f"[INFO] Model outputs: {[output.name for output in onnx_model.graph.output]}")
            
            # Get input/output shapes
            for input_tensor in onnx_model.graph.input:
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_tensor.type.tensor_type.shape.dim]
                print(f"[INFO] Input '{input_tensor.name}' shape: {shape}")
            
            for output_tensor in onnx_model.graph.output:
                shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_tensor.type.tensor_type.shape.dim]
                print(f"[INFO] Output '{output_tensor.name}' shape: {shape}")
            
        except Exception as e:
            print(f"[WARNING] ONNX model verification failed: {e}")
        
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return save_path
       
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
