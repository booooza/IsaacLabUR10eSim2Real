from __future__ import annotations

import gymnasium as gym
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.reach.reach_env_cfg import ReachEnvCfg
from ur10e_sim2real.tasks.direct.isaac_lab_tutorial.isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg
import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, VecEnvObs
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaacsim.core.utils.torch as torch_utils
from isaaclab.managers import SceneEntityCfg

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_apply


class ReachEnv(DirectRLEnv):
    robot: Articulation
    cfg: ReachEnvCfg

    def __init__(self, cfg: ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene.articulations["robot"]
        self.joint_ids, _ = self.robot.find_joints(self.cfg.joint_names)
        
        # Define joint limits
        if self.cfg.action_type == "position":
            self.joint_limits_lower = self.robot.data.joint_pos_limits[:, self.joint_ids, 0].clone()
            self.joint_limits_upper = self.robot.data.joint_pos_limits[:, self.joint_ids, 1].clone()
        elif self.cfg.action_type == "velocity":
            self.joint_limits_lower = -self.robot.data.joint_vel_limits[:, self.joint_ids].clone()
            self.joint_limits_upper = self.robot.data.joint_vel_limits[:, self.joint_ids].clone()
            
        self.action_scale = torch.tensor(self.cfg.action_scale, device=self.device)
        self.joint_ranges = (self.joint_limits_upper - self.joint_limits_lower) * self.action_scale
        
        self.actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self.previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        
        # Define per-joint offset (center positions)
        if self.cfg.use_default_offset:
            if self.cfg.action_type == "position":
                self.joint_offset = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
            if self.cfg.action_type == "velocity":
                self.joint_offset = self.robot.data.default_joint_vel[:, self.joint_ids].clone()    
        else:
            # Midpoint of limits
            self.joint_offset = (self.joint_limits_lower + self.joint_limits_upper) / 2  

        self.pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.quat_w = torch.tensor([[1., 0., 0., 0.]], device=self.device).repeat(self.num_envs, 1)

        # maximum xyz reach
        self.max_reach = torch.tensor([1.3, 1.3, 1.3], device=self.device)

        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.target_cfg)

        # initialize target position w.r.t. robot base
        self.target_pos_low  = torch.tensor([0.30, -0.25, 0.00], device=self.device)
        self.target_pos_high = torch.tensor([0.80,  0.25, 0.50], device=self.device)
        self.target_pos = torch.tensor([[0.55, 0.0, 0.25]], device=self.device).repeat(self.num_envs, 1)
        self.target_quat = torch.tensor([[0.0, -0.70710678, 0.70710678, 0.0]], device=self.device).repeat(self.num_envs, 1)
        self.target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat_w[:, 0] = 1.0 

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # initialize metrics
        self._reset_episode_stats()
        if "log" not in self.extras:
            self.extras["log"] = dict()

    """
    Implementation-specific functions.
    """

    def _setup_scene(self):
        """Setup the scene for the environment.

        This function is responsible for creating the scene objects and setting up the scene for the environment.
        The scene creation can happen through :class:`isaaclab.scene.InteractiveSceneCfg` or through
        directly creating the scene objects and registering them with the scene manager.

        We leave the implementation of this function to the derived classes. If the environment does not require
        any explicit scene setup, the function can be left empty.
        """
        pass

    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before stepping through the physics.

        This function is responsible for pre-processing the actions before stepping through the physics.
        It is called before the physics stepping (which is decimated).

        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, action_dim).
        """
        self.previous_actions = self.actions.clone()
        self.actions = actions.clone()
        
        # Clamp to joint limits
        self.processed_actions = torch.clamp(
            self.actions,
            min=self.joint_limits_lower,
            max=self.joint_limits_upper
        )

    def _apply_action(self):
        """Apply actions to the simulator.

        This function is responsible for applying the actions to the simulator. It is called at each
        physics time-step.
        """
        if self.cfg.action_type == "effort":
            self.robot.set_joint_effort_target(self.processed_actions, joint_ids=self.joint_ids)
        if self.cfg.action_type == "position":
            self.robot.set_joint_position_target(self.processed_actions, joint_ids=self.joint_ids)
        if self.cfg.action_type == "velocity":
            self.robot.set_joint_velocity_target(self.processed_actions, joint_ids=self.joint_ids)

    def _get_observations(self) -> VecEnvObs:
        """Compute and return the observations for the environment.

        Returns:
            The observations for the environment.
        """
        # Get raw joint positions
        joint_positions = self.robot.data.joint_pos[:, self.joint_ids]  # (num_envs, 6)
        joint_velocities = self.robot.data.joint_vel[:, self.joint_ids]  # (num_envs, 6)  joint velocities
        ee_pos = self.scene['ee_frame'].data.target_pos_source[..., 0, :]  # (num_envs, 3) end-effector position xyz
        ee_quat = self.scene['ee_frame'].data.target_quat_source[..., 0, :]  # (num_envs, 4) end-effector orientation quat
        target_pos = self.target_pos  # (num_envs, 3 target position xyz
        target_quat = self.target_quat # (num_envs, 4) target orientation quat
        previous_actions = self.previous_actions  # (num_envs, 6) previous actions

        obs = torch.cat(
            (
                joint_positions,
                joint_velocities,
                ee_pos,
                ee_quat,
                target_pos,
                target_quat,
                previous_actions,
            ),
            dim=-1,
        )

        # logging

        return {"policy": obs}

    def _get_states(self) -> VecEnvObs | None:
        """Compute and return the states for the environment.

        The state-space is used for asymmetric actor-critic architectures. It is configured
        using the :attr:`DirectRLEnvCfg.state_space` parameter.

        Returns:
            The states for the environment. If the environment does not have a state-space, the function
            returns a None.
        """
        return None  # noqa: R501

    def _get_rewards(self) -> torch.Tensor:
        """Compute and return the rewards for the environment.

        Returns:
            The rewards for the environment. Shape is (num_envs,).
        """
        ee_frame = self.scene['ee_frame']
        ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
        ee_quat = ee_frame.data.target_quat_w[..., 0, :]
        target_pos = self.target_pos_w
        target_quat = self.target_quat

        # --- Dense ---
        # Linear L2 distance reward over the full workspace
        distance_l2 = torch.norm(target_pos - ee_pos, p=2, dim=-1)
        # Tanh reward for precision near the goal
        std = 0.2 # start guiding with tanh from 20 cm
        distance_tanh = 1.0 - torch.tanh(distance_l2 / std)
        # Orientation error as the angular error between input quaternions in radians
        orientation_error = quat_error_magnitude(target_quat, ee_quat)
        reach_success: torch.Tensor = ((distance_l2 < self.cfg.reach_pos_threshold) &
                 (orientation_error < self.cfg.reach_rot_threshold))
        
        # --- Penalties ---
        action_l2 = torch.sum(torch.square(self.actions), dim=1)
        action_diff = self.actions - self.previous_actions
        action_rate_l2 = torch.sum(torch.square(action_diff), dim=1)
        action_rate_limit = self._action_rate_limit(self.joint_ids, action_diff, threshold_ratio=0.1)
        joint_vel_l2 = torch.sum(torch.square(self.robot.data.joint_vel[:, self.joint_ids]), dim=1)
        joint_pos_limit = self._joint_pos_limits(self.joint_ids)
        joint_vel_limit = self._joint_vel_limits(self.joint_ids, soft_ratio=1.0)
        min_link_distance = self._minimum_link_distance(min_dist=0.1)

        # --- Sparse ---
        # Update consecutive_successes
        self.consecutive_successes = torch.where(
            reach_success,
            self.consecutive_successes + 1,
            torch.zeros_like(self.consecutive_successes)
        )
        stable_success = self.consecutive_successes >= self.cfg.success_bonus_stable_steps
        success_bonus = torch.where(
            stable_success & (~self.episode_success), # bonus only once per episode if stable for n steps
            torch.ones_like(reach_success),
            torch.zeros_like(reach_success)
        )
        # --- Total weighted reward ---
        reward = (
            # Task rewards
            self.cfg.distance_l2_w * distance_l2 +
            self.cfg.orientation_error_w * orientation_error
            # Regularization penalties
            # self.cfg.action_l2_w * action_l2 +
            # self.cfg.action_rate_l2_w * action_rate_l2 +
            # Safety limits
            # self.cfg.joint_pos_limit_w * joint_pos_limit +
            # self.cfg.joint_vel_limit_w * joint_vel_limit +
            # self.cfg.min_link_distance_w * min_link_distance +
            # Success bonus
            # self.cfg.success_bonus_w * success_bonus
        )

        # --- Logging ---
        self.episode_reward += reward
        self.episode_success = self.episode_success | stable_success
        self.distance_l2 = distance_l2
        self.orientation_error = orientation_error
        
        self.extras["Metrics/success_rate"] = reach_success.float().mean().item()
        self.extras["Metrics/stable_success_rate"] = stable_success.float().mean().item()
        self.extras["Debug/distance_mean"] = distance_l2.mean().item()
        self.extras["Debug/distance_mode"] = distance_l2.mode()
        self.extras["Debug/distance_min"] = distance_l2.min().item()
        self.extras["Debug/distance_max"] = distance_l2.max().item()
        self.extras["Debug/action_mean"] = self.actions.abs().mean().item()
        self.extras["Debug/action_max"] = self.actions.abs().max().item()
        self.extras["Debug/reward"] = reward.mean().item()

        if reach_success.any():
            print(f"Reached goal pose for {reach_success.sum().item()} envs.")
        if stable_success.any():
            print(f"Reached stable goal pose for {stable_success.sum().item()} envs.")

        if self.common_step_counter % 1000 == 0:
            env_id = 0
            
            # Current state
            ee_pos = self.scene['ee_frame'].data.target_pos_w[env_id, 0, :3]
            target_pos = self.target_pos_w[env_id]  # (num_envs, 3 target position xyz
            joint_pos = self.robot.data.joint_pos[env_id, self.joint_ids]
            
            print(f"\n=== Env {env_id} ===")
            print(f"EE pos: {ee_pos}")
            print(f"Target pos: {target_pos}")
            print(f"Distance: {torch.norm(target_pos - ee_pos).item():.4f}m")
            print(f"Joint pos: {joint_pos}")
            print(f"Joint pos (deg): {joint_pos * 180 / 3.14159}")
            
            # Check if any joint near limits
            limits = self.robot.data.soft_joint_pos_limits[env_id, self.joint_ids]
            margin_lower = joint_pos - limits[:, 0]
            margin_upper = limits[:, 1] - joint_pos
            print(f"Margin to lower limits: {margin_lower}")
            print(f"Margin to upper limits: {margin_upper}")
            
            # Check if any joint is limiting progress
            if (margin_lower < 0.2).any() or (margin_upper < 0.2).any():
                print("⚠️ JOINT NEAR LIMIT!")

        return reward

    def _is_danger_of_clamping(self,):
        # Error C403A0: Danger of clamping between the robot lower arm and tool
        # https://forum.universal-robots.com/t/error-c403a0-danger-of-clamping-between-the-robot-lower-arm-and-tool/16142
        forearm_idx = self.robot.find_bodies('forearm_link')[0][0]
        flange_idx = self.robot.find_bodies('flange')[0][0]
        forearm_pos = self.robot.data.body_link_pos_w[:, forearm_idx]  # (N,3)
        flange_pos  = self.robot.data.body_link_pos_w[:, flange_idx]   # (N,3)
        forearm_quat = self.robot.data.body_quat_w[:, forearm_idx]
        # Cylinder around forearm: radius = 0.0375 m
        # Sphere around flange: radius = 0.0375 m
        # Violation when distance < 0.028 m
        FOREARM_R = 0.0375
        FLANGE_R  = 0.0375
        SAFE_DIST = 0.028
        threshold = 0.0375 + 0.0375 + 0.028

        axis_local = torch.tensor([0, 0, 1], dtype=torch.float32, device=forearm_pos.device).expand_as(forearm_pos)
        axis_world = quat_apply(forearm_quat, axis_local)
        # Project flange center onto cylinder axis

        v = flange_pos - forearm_pos
        proj_len = torch.sum(v * axis_world, dim=-1, keepdim=True)
        proj = forearm_pos + proj_len * axis_world

        # Distance between flange center and cylinder axis
        dist = torch.norm(flange_pos - proj, dim=-1)

        threshold = (FOREARM_R + FLANGE_R + SAFE_DIST)
        protective_stop_c403a0 = dist < threshold
    
        return protective_stop_c403a0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return the done flags for the environment.

        Returns:
            A tuple containing the done flags for termination and time-out.
            Shape of individual tensors is (num_envs,).
        """
        # Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        lower = self.robot.data.joint_pos_limits[:, self.joint_ids, 0]
        upper = self.robot.data.joint_pos_limits[:, self.joint_ids, 1]
        # Joint limit violation
        joint_pos_limit_violation = ((self.robot.data.joint_pos[:, self.joint_ids] < lower) |
                                 (self.robot.data.joint_pos[:, self.joint_ids] > upper)).any(dim=1)
        # Abnormal joint velocities
        joint_vel_limit_violation = (self.robot.data.joint_vel.abs() > (self.robot.data.joint_vel_limits * 2)).any(dim=1)
        # Minimum link distance violation
        minimum_link_distance_violation = self._minimum_link_distance(min_dist=0.05).abs() > 0

        died = joint_pos_limit_violation | joint_vel_limit_violation | minimum_link_distance_violation
        
        if died.any():
            # print(f"Episode terminated due to safety limit violation in {died.sum().item()} envs.")
            self.extras["Termination/joint_pos_limit_violation"] = joint_pos_limit_violation.sum().item()
            self.extras["Termination/joint_vel_limit_violation"] = joint_vel_limit_violation.sum().item()
            self.extras["Termination/minimum_link_distance_violation"] = minimum_link_distance_violation.sum().item()

        return False, time_out

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer = VisualizationMarkers(self.cfg.target_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(
            translations=self.target_pos_w,
            orientations=self.target_quat_w,
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self._log_episode_stats(env_ids)
        self._reset_target_pose(env_ids)
        if self.cfg.randomize_joints:
            self.__reset_joints_by_offset(env_ids, position_range=(-1.0, 1.0))
        else:
            self._reset_articulation(env_ids)
        self._reset_episode_stats()

    def _joint_pos_limits(self, joint_ids: list) -> torch.Tensor:
        """Penalize joint positions if they cross the soft limits.

        This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
        """
        # compute out of limits constraints
        out_of_limits = -(
            self.robot.data.joint_pos[:, joint_ids] - self.robot.data.soft_joint_pos_limits[:, joint_ids, 0]
        ).clip(max=0.0)
        out_of_limits += (
            self.robot.data.joint_pos[:, joint_ids] - self.robot.data.soft_joint_pos_limits[:, joint_ids, 1]
        ).clip(min=0.0)
        
        violations = (out_of_limits > 0.01).any(dim=1)
        
        if violations.any():
            num_violations = violations.sum().item()
            # Log statistics
            self.extras["Violations/joint_pos_count"] = num_violations
            self.extras["Violations/joint_pos_mean"] = out_of_limits[violations].mean().item()
            self.extras["Violations/joint_pos_max"] = out_of_limits.max().item()            
            # print(f"⚠️  Joint position limit violations in {num_violations}/{self.num_envs} envs "
            #     f"(max excess: {out_of_limits.max().item():.4f} rad)")
        
        return torch.sum(out_of_limits, dim=1)

    def _joint_vel_limits(self, joint_ids: list, soft_ratio: float) -> torch.Tensor:
        """Penalize joint velocities if they cross the soft limits."""
        out_of_limits = (
            torch.abs(self.robot.data.joint_vel[:, joint_ids])
            - self.robot.data.joint_vel_limits[:, joint_ids] * soft_ratio
        )
        out_of_limits_clip = out_of_limits.clip(min=0.0, max=1.0)
        violations = (out_of_limits_clip > 0.01).any(dim=1)  # 0.01 rad/s threshold
        
        if violations.any():
            num_violations = violations.sum().item()
            self.extras["Violations/joint_vel_count"] = num_violations
            self.extras["Violations/joint_vel_mean"] = out_of_limits_clip[violations].mean().item()
            self.extras["Violations/joint_vel_max"] = out_of_limits_clip.max().item()            
            # print(f"⚠️  Joint velocity limit violations...")
        
        return torch.sum(out_of_limits_clip, dim=1)

    def _minimum_link_distance(
        self, min_dist=0.10,
        safe_links=["shoulder_link", "upper_arm_link", "forearm_link",
                    "wrist_1_link", "wrist_2_link", "wrist_3_link", "flange"]
    ):

        link_idxs = self.robot.find_bodies(safe_links)[0]
        pos = self.robot.data.body_link_pos_w[:, link_idxs]   # (N, K, 3)
        N, K, _ = pos.shape

        diff = pos[:, :, None, :] - pos[:, None, :, :]
        dist = torch.norm(diff, dim=-1)            # (N, K, K)

        # self distances
        mask = torch.eye(K, dtype=torch.bool, device=pos.device)

        # mask adjacent pairs
        adj = torch.zeros((K, K), dtype=torch.bool, device=pos.device)
        for i in range(K - 1):
            adj[i, i+1] = True
            adj[i+1, i] = True

        # combine masks
        mask = mask | adj

        dist_masked = dist.masked_fill(mask, float('inf'))

        violations = dist_masked < min_dist
        per_env_violation = violations.any(dim=(1, 2))
        num_violations_each_env = violations.sum(dim=(1, 2))

        if per_env_violation.any():
            num_envs_violating = per_env_violation.sum().item()

            # statistics only on violating environments
            violating_dists = dist_masked[violations]

            self.extras["Violations/link_dist_env_count"] = num_envs_violating
            self.extras["Violations/link_dist_mean"] = violating_dists.mean().item()
            self.extras["Violations/link_dist_min"] = violating_dists.min().item()
            self.extras["Violations/link_dist_max"] = violating_dists.max().item()
            self.extras["Violations/link_dist_total_pairs"] = violations.sum().item()

        # penalty = number of violating pairs
        return num_violations_each_env.float()
    
    def _action_rate_limit(self, joint_ids: list, action_diff: torch.Tensor, threshold_ratio: float = 0.1,
    ) -> torch.Tensor:
        """Penalize action rate changes exceeding a threshold ratio of joint velocity limits.
        
        Small oscillations within the threshold are acceptable and not penalized.
        
        Args:
            threshold_ratio: Ratio of joint velocity limits to use as threshold (default: 0.1 = 10%)
        """
        # Get joint velocity limits from the asset (for the controlled joints)
        joint_vel_limits = self.robot.data.joint_vel_limits[:, joint_ids]
        
        # Calculate threshold (10% of velocity limits)
        threshold = joint_vel_limits * threshold_ratio
        
        # Calculate excess beyond threshold (zero if within threshold)
        excess = torch.abs(action_diff) - threshold
        excess = torch.clamp(excess, min=0.0)
        
        # Return squared penalty on excess
        return torch.sum(torch.square(excess), dim=1)
        
    def _reset_target_pose(self, env_ids):
        # update goal pose and markers within workspace bounds

        # sample positions
        rand_pos = torch.rand((len(env_ids), 3), device=self.device)
        self.target_pos[env_ids] = rand_pos * (self.target_pos_high - self.target_pos_low) + self.target_pos_low

        # sample orientations with random yaw around z
        yaw = (torch.rand(len(env_ids), device=self.device) - 0.5) * 2 * torch.pi
        roll = torch.ones_like(yaw) * torch.pi     # Z down
        pitch = torch.zeros_like(yaw)

        self.target_quat[env_ids] = torch_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # convert to world frame
        base_pos_w = self.robot.data.root_pos_w[env_ids]
        base_quat_w = self.robot.data.root_quat_w[env_ids]
        self.target_pos_w[env_ids] = torch_utils.quat_apply(base_quat_w, self.target_pos[env_ids]) + base_pos_w
        self.target_quat_w[env_ids] = torch_utils.quat_mul(base_quat_w, self.target_quat[env_ids])

    def __reset_joints_by_offset(
        self,
        env_ids: torch.Tensor,
        position_range: tuple[float, float] = (0.0, 0.0),
        velocity_range: tuple[float, float] = (0.0, 0.0),
    ):
        """Reset the robot joints with offsets around the default position and velocity by the given ranges.

        This function samples random values from the given ranges and biases the default joint positions and velocities
        by these values. The biased values are then set into the physics simulation.
        """
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self.robot

        # cast env_ids to allow broadcasting
        if self.joint_ids != slice(None):
            iter_env_ids = env_ids[:, None]
        else:
            iter_env_ids = env_ids

        # get default joint state
        joint_pos = asset.data.default_joint_pos[iter_env_ids, self.joint_ids].clone()
        joint_vel = asset.data.default_joint_vel[iter_env_ids, self.joint_ids].clone()

        # bias these values randomly
        joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
        joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

        # clamp joint pos to limits
        joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, self.joint_ids]
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

        # clamp joint vel to limits
        joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, self.joint_ids]
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        # set into the physics simulation
        asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self.joint_ids, env_ids=env_ids)
        
    def _reset_articulation(self, env_ids):
        # reset articulation assets to_default
        for articulation_asset in self.scene.articulations.values():
            # obtain default and deal with the offset for env origins
            default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:3] += self.scene.env_origins[env_ids]
            # set into the physics simulation
            articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
            articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
            # obtain default joint positions
            default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
            default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
            # set into the physics simulation
            articulation_asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target(default_joint_vel, env_ids=env_ids)
            articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

    def _log_episode_stats(self, env_ids):
        self.extras["Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["Stats/reward"] = self.episode_reward.mean().item()
        self.extras["Stats/success"] = self.episode_success.float().mean().item()
        self.extras["Stats/consecutive_successes"] = self.consecutive_successes.float().mean().item()
        self.extras["Stats/distance_l2"] = self.distance_l2.mean().item()
        self.extras["Stats/orientation_error"] = self.orientation_error.mean().item()

    def _reset_episode_stats(self):
        self.episode_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.distance_l2 = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.orientation_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
