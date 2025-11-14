# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from source.ur10e_sim2real.ur10e_sim2real.robots.ur10e import UR10e_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.envs.mdp.actions import JointAction, JointEffortAction, JointVelocityAction, JointPositionAction

import gymnasium as gym
import numpy as np
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.reach.scene import ReachSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



@configclass
class ReachEnvCfg(DirectRLEnvCfg):
    # env
    debug_vis = True
    seed = 1234
    decimation = 1
    # episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))
    # episode_length_steps = ceil(3.0 / (1 * 1/125))
    episode_length_s = 3.0 # ~375 steps @ 125 Hz
    # - spaces definition
    action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
    observation_space = gym.spaces.Box(
        low=np.array([
            -1, -1, -1, -1, -1, -1, # 6 joint positions default pos normalized
            -1, -1, -1, -1, -1, -1, # 6 joint velocities limit normalized
            -2000, -2000, -2000, # 3 ee position xyz
            -1, -1, -1, -1, # 4 ee orientation quat
            -2000, -2000, -2000, # 3 target position xyz
            -1, -1, -1, -1, # 4 target orientation quat
            -1, -1, -1, -1, -1, -1 # 6 previous actions
        ]),
        high=np.array([
            1, 1, 1, 1, 1, 1, # 6 joint positions default pos normalized
            1, 1, 1, 1, 1, 1, # 6 joint velocities limit normalized
            2000, 2000, 2000, # 3 ee position xyz
            1, 1, 1, 1, # 4 ee orientation quat
            2000, 2000, 2000, # 3 target position xyz
            1, 1, 1, 1, # 4 target orientation quat
            1, 1, 1, 1, 1, 1 # 6 previous actions
        ]), 
        shape=(32,), dtype=np.float64
    )
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/125,
        render_interval=decimation,
        physx=PhysxCfg(
            # PhysX settings - for stable physics
            bounce_threshold_velocity = 0.2,
            gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity = 16 * 1024,
            friction_correlation_distance = 0.00625
        )
    )

    # scene
    scene: InteractiveSceneCfg = ReachSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=True)
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

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
    distance_tanh_w = 0.1
    distance_l2_w = -0.2
    orientation_error_w = -0.1
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
