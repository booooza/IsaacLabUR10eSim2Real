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
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.grasp.grasp_scene import GraspSceneCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.reach.reach_env_cfg import ReachEnvCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



@configclass
class GraspEnvCfg(ReachEnvCfg):
    # - spaces definition
    action_space = gym.spaces.Box(
        low=np.array([-2.0944, -2.0944, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159]), 
        high=np.array([2.0944, 2.0944, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159]), 
        shape=(7,), dtype=np.float32
    )
    observation_space = gym.spaces.Box(
        low=np.array([
            -6.28319, -6.28319, -6.28319, -6.28319, -6.28319, -6.28319, # 6 joint positions
            -2.0944, -2.0944, -3.14159, -3.14159, -3.14159, -3.14159, # 6 joint velocities
            -1.3, -1.3, -1.3, # 3 ee position xyz
            -1, -1, -1, -1, # 4 ee orientation quat
            -1.3, -1.3, -1.3, # 3 object position xyz
            -1, -1, -1, -1, # 4 object orientation quat
            -1, # 1 gripper position (-1 fully closed)
            -1, # 1 target width gripper max width normalized
            -2.0944, -2.0944, -3.14159, -3.14159, -3.14159, -3.14159, -3.14159 # 7 previous actions
        ]),
        high=np.array([
            6.28319, 6.28319, 6.28319, 6.28319, 6.28319, 6.28319, # 6 joint positions
            2.0944, 2.0944, 3.14159, 3.14159, 3.14159, 3.14159, # 6 joint velocities
            1300, 1300, 1300, # 3 object position xyz
            1, 1, 1, 1, # 4 object orientation quat
            1300, 1300, 1300, # 3 target position xyz
            1, 1, 1, 1, # 4 target orientation quat
            1, # 1 gripper positions (+1 fully open)
            1, # 1 target width gripper max width normalized
            2.0944, 2.0944, 3.14159, 3.14159, 3.14159, 3.14159, 3.14159 # 7 previous actions
        ]), 
        shape=(35,), dtype=np.float64
    )
    state_space = 0

    # scene
    scene: InteractiveSceneCfg = GraspSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=True)
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    gripper_joint_names = ["robotiq_hande_left_finger_joint", "robotiq_hande_right_finger_joint"]

    # config

    use_default_offset = True
    action_type = "velocity"
    reach_pos_threshold = 0.02 # 2 cm
    reach_rot_threshold = 0.1 # 5.73 deg
    reach_pos_w = 1.0
    reach_rot_w = 1.0
    reach_success_w = 10.0
    success_bonus_stable_steps = 5 # 5*(1/125) = (~40 ms)

    # reward weights
    distance_tanh_w = 0.1
    distance_l2_w = -0.2
    orientation_error_w = -0.1
    reach_bonus_w = 1.0
    grasp_bonus_w = 1.0
    lift_bonus_w = 1.0
    
    grip_width_tanh_w = 0.1
    grip_width_l2_w = -0.2

    grasp_force_threshold = 10.0 # 10 nm
    grasp_width_threshold = 0.005 # 5 mm
    minimal_lift_height = 0.04 # 4 cm

    # penalties
    action_l2_w = 0.00001
    action_rate_l2_w = 0.00001
    joint_pos_limit_w = 0.1
    joint_vel_limit_w = 0.1
    min_link_distance_w = 0.1
    success_bonus_w = 100.0

    randomize_joints = True
    object_pose_range = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)}
    joint_pos_range = (-1.0, 1.0)
    
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
    
    viewer = ViewerCfg(eye = (5.0, 0.0, 1.5), lookat = (0.0, 0.0, 0.0))

@configclass
class GraspEnvPlayCfg(GraspEnvCfg):
    """Configuration for grasp environment during play/testing."""
    episode_length_s = 3.0 # ~375 steps @ 125 Hz
    num_envs = 1
