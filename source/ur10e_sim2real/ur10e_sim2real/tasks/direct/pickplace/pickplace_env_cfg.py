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
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.pickplace.pickplace_scene import PickPlaceSceneCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.reach.reach_env_cfg import ReachEnvCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.grasp.grasp_env_cfg import GraspEnvCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class PickPlaceEnvCfg(GraspEnvCfg):
    decimation = 1
    seed = None
    # - spaces definition
    action_space = gym.spaces.Box(
        low=np.array([-2.0944, -2.0944, -3.14159, -3.14159, -3.14159, -3.14159]), 
        high=np.array([2.0944, 2.0944, 3.14159, 3.14159, 3.14159, 3.14159]), 
        shape=(6,), dtype=np.float64
    )
    observation_space = gym.spaces.Box(
        low=np.array([
            -6.28319, -6.28319, -6.28319, -6.28319, -6.28319, -6.28319, # 6 joint positions
            -2.0944, -2.0944, -3.14159, -3.14159, -3.14159, -3.14159,    # 6 joint velocities
            -1.3, -1.3, -1.3, # 3 ee position xyz
            -1, -1, -1, -1, # 4 ee orientation quat
            -1.3, -1.3, -1.3, # 3 object position xyz
            -1, -1, -1, -1, # 4 object orientation quat
            -1.3, -1.3, -1.3, # 3 target position xyz
            -1, -1, -1, -1, # 4 target orientation quat
            -2.0944, -2.0944, -3.14159, -3.14159, -3.14159, -3.14159 # 6 previous actions
        ]),
        high=np.array([
            6.28319, 6.28319, 6.28319, 6.28319, 6.28319, 6.28319, # 6 joint positions
            2.0944, 2.0944, 3.14159, 3.14159, 3.14159, 3.14159,    # 6 joint velocities
            1.3, 1.3, 1.3, # 3 ee position xyz
            1, 1, 1, 1, # 4 ee orientation quat
            1.3, 1.3, 1.3, # 3 object position xyz
            1, 1, 1, 1, # 4 object orientation quat
            1.3, 1.3, 1.3, # 3 target position xyz
            1, 1, 1, 1, # 4 target orientation quat
            2.0944, 2.0944, 3.14159, 3.14159, 3.14159, 3.14159 # 6 previous actions
        ]), 
        shape=(39,), dtype=np.float64
    )
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=2,
        physx=PhysxCfg(
            # PhysX settings - for stable physics
            bounce_threshold_velocity = 0.2,
            gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity = 16 * 1024,
            friction_correlation_distance = 0.00625
        )
    )

    episode_length_s = 6.0 # ~750k steps @ 125 Hz

    # scene
    scene: InteractiveSceneCfg = PickPlaceSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=True)

    # linear penalties
    action_l2_w = 0.001
    action_rate_l2_w = 0.005
    joint_pos_limit_w = 1.0
    joint_vel_limit_w = 1.0
    min_link_distance_w = 1.0

    # thresholds
    reach_pos_threshold = 0.02 # 2 cm
    reach_rot_threshold = 0.2 # 11 deg
    grasp_force_threshold = 10.0 # 10 nm
    grasp_width_threshold = 0.005 # 5 mm
    minimal_lift_height = 0.04 # 4 cm
    place_pos_threshold = 0.05 # 5 cm
    transport_pos_threshold = 0.07 # 7 cm
    place_rot_threshold = 0.1  # 5.73 deg
    max_transport_dist = 0.65

    # std
    distance_ee_obj_tanh_std = 0.1 # 90% reward at 1 cm, non-zero from 0.5m
    distance_ee_obj_tanh_fine_std = 0.03 # 90% reward at 0.3cm, non-zero from 15cm
    rot_error_ee_obj_tanh_std = 2.0 # 90% reward at 11.5 degree, non-zero from 567 deg
    grip_width_tanh_std = 0.01  # 90% reward at <1mm error, non-zero from 5 cm
    distance_obj_target_tanh_std = 0.5 # 90% reward at 5 cm, non-zero from 2.5m
    distance_obj_target_tanh_fine_std = 0.1 # 90% reward at 1 cm, non-zero from 0.5m

    # domain randomization
    object_pose_range = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)}
    target_pose_range = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)}
    
@configclass
class PickPlaceEnvPlayCfg(PickPlaceEnvCfg):
    """Configuration for grasp environment during play/testing."""
    episode_length_s = 3.0 # ~375 steps @ 125 Hz
    num_envs = 1
