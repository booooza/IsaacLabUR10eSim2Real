# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from source.ur10e_sim2real.ur10e_sim2real.robots.ur10e import UR10e_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

from isaaclab.envs.mdp.actions import JointAction, JointEffortAction, JointVelocityAction, JointPositionAction

import gymnasium as gym
import numpy as np
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.grasp.scene import GraspSceneCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.reach.reach_env_cfg import ReachEnvCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



@configclass
class GraspEnvCfg(ReachEnvCfg):
    # - spaces definition
    action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
    observation_space = gym.spaces.Box(
        low=np.array([
            -1, -1, -1, -1, -1, -1, # 6 joint positions default pos normalized
            -1, # 1 gripper positions default normalized
            -1, -1, -1, -1, -1, -1, # 6 joint velocities limit normalized
            -1, # 1 gripper velocity limit normalized
            -2000, -2000, -2000, # 3 ee position xyz
            -1, -1, -1, -1, # 4 ee orientation quat
            -2000, -2000, -2000, # 3 target position xyz
            -1, -1, -1, -1, # 4 target orientation quat
            -1, # 1 target width gripper max width normalized
            -1, -1, -1, -1, -1, -1 # 6 previous actions
        ]),
        high=np.array([
            1, 1, 1, 1, 1, 1, # 6 joint positions default pos normalized
            1, # 1 gripper positions default normalized
            1, 1, 1, 1, 1, 1, # 6 joint velocities limit normalized
            1, # 1 gripper velocity limit normalized
            2000, 2000, 2000, # 3 ee position xyz
            1, 1, 1, 1, # 4 ee orientation quat
            2000, 2000, 2000, # 3 target position xyz
            1, 1, 1, 1, # 4 target orientation quat
            1, # 1 target width gripper max width normalized
            1, 1, 1, 1, 1, 1 # 6 previous actions
        ]), 
        shape=(35,), dtype=np.float64
    )
    state_space = 0

    # scene
    scene: InteractiveSceneCfg = GraspSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=True)
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    gripper_joint_names = ["robotiq_hande_left_finger_joint", "robotiq_hande_right_finger_joint"]

    # config
    # UR10e arm joints
    action_scale = [0.5, 0.5, 0.5, 0.8, 0.8, 0.8]  # Shoulder/elbow: ±0.25 rad range per step, Wrist joints: ±0.4 rad range per step
    use_default_offset = True
    action_type = "position"
    reach_pos_threshold = 0.02 # 2 cm
    reach_rot_threshold = 0.2 # 11.4592 deg
    reach_pos_w = 1.0
    reach_rot_w = 1.0
    reach_success_w = 10.0
    success_bonus_stable_steps = 10 # 10*(1/125) = (~80 ms)

    # reward weights
    distance_tanh_w = 1.0
    
    distance_l2_w = -0.2
    grip_width_tanh_w = 0.1
    grip_width_l2_w = -0.2
    lift_reward_w = 2.0

    # penalties
    action_l2_w = -0.001
    action_rate_l2_w = -0.005
    joint_pos_limit_w = -1.0
    joint_vel_limit_w = -1.0
    success_bonus_w = 10.0

    # target 
    target_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
            ),
        }
    )
    
    viewer = ViewerCfg(eye = (2.5, 2.5, 2.5), lookat = (0.5, 0.0, 0.5))

@configclass
class GraspEnvPlayCfg(GraspEnvCfg):
    """Configuration for grasp environment during play/testing."""
    episode_length_s = 3.0 # ~375 steps @ 125 Hz
    num_envs = 1
