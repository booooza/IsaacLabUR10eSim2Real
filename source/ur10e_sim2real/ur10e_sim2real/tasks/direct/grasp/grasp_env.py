from __future__ import annotations

import gymnasium as gym
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.reach.reach_env_cfg import ReachEnvCfg
from ur10e_sim2real.tasks.direct.isaac_lab_tutorial.isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg
import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, VecEnvObs
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaacsim.core.utils.torch as torch_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_error_magnitude, quat_mul, quat_inv
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.math import subtract_frame_transforms
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.reach.reach_env import ReachEnv
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.grasp.grasp_env_cfg import GraspEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class GraspEnv(ReachEnv):
    robot: Articulation
    cfg: GraspEnvCfg

    def __init__(self, cfg: GraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.robot = self.scene.articulations["robot"]
        # Arm joints (6)
        self.joint_ids, _ = self.robot.find_joints(self.cfg.joint_names)
        
        # Gripper joints (2)
        self.gripper_joint_ids, _ = self.robot.find_joints(self.cfg.gripper_joint_names)

        self.lower_joint_pos_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        self.upper_joint_pos_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]

        gripper_limits = self.robot.data.joint_pos_limits[:, self.gripper_joint_ids, :]  # (num_envs, 2, 2)
        self.gripper_max_width = gripper_limits[0, 0, 1].item()
        self.gripper_default_width = self.robot.data.default_joint_pos[0, self.gripper_joint_ids].sum().item()

        self.actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self.previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        self.pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.quat_w = torch.tensor([[1., 0., 0., 0.]], device=self.device).repeat(self.num_envs, 1)

        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.target_cfg)

        # initialize target position w.r.t. robot base
        self.target_pos_low  = torch.tensor([0.40, -0.25, 0.20], device=self.device)
        self.target_pos_high = torch.tensor([0.70,  0.25, 0.50], device=self.device)
        self.target_pos = torch.tensor([[0.55, 0.0, 0.25]], device=self.device).repeat(self.num_envs, 1)
        self.target_quat = torch.tensor([[0.0, -0.70710678, 0.70710678, 0.0]], device=self.device).repeat(self.num_envs, 1)
        self.target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_quat_w[:, 0] = 1.0 

        

        self.obj_width = 0.03
        self.gripper_max_width = 0.05 # fixed
        self.table_height = 0.015  # From init state
        self.pre_grasp_offset_z = 0.05  # 5cm above object



        # Initialize grasp orientation (top-down)
        object_frame = self.scene['object_frame']
        object_pos_w = object_frame.data.target_pos_w[..., 0, :]
        object_quat_w = object_frame.data.target_quat_w[..., 0, :]
        _, _, obj_yaw = math_utils.euler_xyz_from_quat(object_quat_w) 
        self.grasp_offset = torch.zeros_like(object_pos_w)
        self.grasp_offset[:, 2] = self.pre_grasp_offset_z
        self.grasp_pos_w = object_pos_w + self.grasp_offset
        self.grasp_quat_w = torch_utils.quat_from_euler_xyz(
            torch.ones_like(obj_yaw) * torch.pi, # Roll = π (gripper upside down, Z pointing down)
            torch.zeros_like(obj_yaw), # Pitch = 0
            obj_yaw, # Match object yaw
        )

        self.target_width_open_normalized = 1.0
        self.target_width_closed_normalized = 2.0 * (self.obj_width / self.gripper_max_width) - 1.0
        self.target_width = torch.full((self.num_envs, 1), self.target_width_closed_normalized, device=self.device)

        # Initialize success tracking tensors
        self.reached_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasped_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Episode tracking
        self.episode_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Overall metrics
        self.best_reach_rate = 0.0
        self.best_grasp_rate = 0.0
        self.best_lift_rate = 0.0
        self.best_episode_reward = float('-inf')

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
        
        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, 7).
                    First 6 are arm joints, 7th is gripper width command.
        """
        self.previous_actions = self.actions.clone()
        self.actions = actions.clone()
        
        # Split actions into arm (6) and gripper (1)
        arm_actions = self.actions[:, :6]  # (num_envs, 6)
        gripper_action = self.actions[:, 6:7]  # (num_envs, 1)

        processed_arm_actions = torch.clamp(
            arm_actions,
            min=self.joint_limits_lower,
            max=self.joint_limits_upper
        )

        # Clamp gripper action to [-1, 1]
        gripper_action = torch.clamp(gripper_action, min=-1.0, max=1.0)
        # Process gripper action (map from [-1, 1] to actual width, then split between fingers)
        gripper_max_width = 0.05
        # Map action from [-1, 1] to [0, gripper_max_width]
        target_width = (gripper_action + 1.0) / 2.0 * gripper_max_width  # (num_envs, 1)
        # Each finger should be at half the total width
        finger_position = target_width / 2.0  # (num_envs, 1)
        
        # Combine processed actions: 6 arm joints + 2 gripper fingers (mirrored)
        self.processed_actions = torch.cat(
            (
                processed_arm_actions,  # (num_envs, 6)
                finger_position,        # (num_envs, 1) - left finger
                finger_position,        # (num_envs, 1) - right finger (mimic)
            ),
            dim=-1
        )  # (num_envs, 8)

    def _apply_action(self):
        """Apply actions to the simulator.

        This function is responsible for applying the actions to the simulator. It is called at each
        physics time-step.
        """
        # Arm control: velocity control
        self.robot.set_joint_velocity_target(self.processed_actions[:, :6], joint_ids=self.joint_ids)
        # Gripper control: position control
        self.robot.set_joint_position_target(self.processed_actions[:, 6:], joint_ids=self.gripper_joint_ids)

    def _get_observations(self) -> VecEnvObs:
        """Compute and return the observations for the environment.
        Returns:
            The observations for the environment.
        """        
        joint_positions = self.robot.data.joint_pos[:, self.joint_ids]  # (num_envs, 6)
        joint_velocities = self.robot.data.joint_vel[:, self.joint_ids]  # (num_envs, 6)  joint velocities
        
        # End-effector pose
        ee_pos = self.scene['ee_frame'].data.target_pos_source[..., 0, :]  # (num_envs, 3)
        ee_quat = self.scene['ee_frame'].data.target_quat_source[..., 0, :]  # (num_envs, 4)

        # Object pose
        obj_pos = self.scene['object_frame'].data.target_pos_source[..., 0, :]  # (num_envs, 3)
        obj_quat = self.scene['object_frame'].data.target_quat_source[..., 0, :]  # (num_envs, 4)
        
        # Target pose
        target_pos = self.target_pos   # (num_envs, 3)
        target_quat = self.target_quat  # (num_envs, 4)
        
        # Normalize target width to [-1, 1]
        gripper_max_width = 0.05
        target_width = 0.03
        target_width_normalized = 2.0 * (target_width / gripper_max_width) - 1.0  # Result: 0.2
        target_width_tensor = torch.full((self.num_envs, 1), target_width_normalized, device=self.device)
        
       # Get current gripper width (sum of both finger positions)
        gripper_positions = self.robot.data.joint_pos[:, self.gripper_joint_ids]  # (num_envs, 2)
        current_gripper_width = gripper_positions.sum(dim=-1)  # (num_envs,)
        current_width_normalized = 2.0 * (current_gripper_width / self.gripper_max_width) - 1.0
        gripper_width = current_width_normalized.unsqueeze(-1)  # (num_envs, 1)
        target_gripper_width = self.target_width

        # Gripper contact detection
        contact_forces = self.scene._sensors['contact_sensor']._data.net_forces_w # (num_envs, 2, 3)
        normal_forces = contact_forces[:, :, 1].abs()
        min_normal_force = torch.min(normal_forces, dim=-1).values # both fingers need y-forces
        is_gripping = min_normal_force > 10.0 # Force threshold (nm)

        obj_pos_rel = obj_pos - ee_pos
        obj_quat_rel = quat_mul(quat_inv(ee_quat), obj_quat)

        full_obs = torch.cat(
            (
                joint_positions,
                joint_velocities,
                ee_pos,
                ee_quat,
                obj_pos_rel,
                obj_quat_rel,
                gripper_width,
                target_gripper_width,
                self.previous_actions,
            ),
            dim=-1,
        )
        actor_obs = torch.cat(
            (
                ee_pos,
                ee_quat,
                obj_pos_rel,
                obj_quat_rel,
                self.previous_actions,
            ),
            dim=-1,
        )
        critic_obs = full_obs

        # logging
        if self.cfg.observations == "symmetric":
            return {"policy": full_obs, "critic": full_obs}
        else:
            return {"policy": actor_obs, "critic": critic_obs}  

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
        object_frame = self.scene['object_frame']
        object_pos = object_frame.data.target_pos_w[..., 0, :]
        object_quat = object_frame.data.target_quat_w[..., 0, :]

        all_joint_ids = self.joint_ids.copy()
        all_joint_ids.append(self.gripper_joint_ids[0])

        # --- Dense ---
        # Linear L2 distance EE to obj
        distance_l2 = torch.norm(ee_pos - object_pos, dim=-1)
        orientation_error = quat_error_magnitude(self.grasp_quat_w, ee_quat)

        contact_forces = self.scene._sensors['contact_sensor']._data.net_forces_w # (num_envs, 2, 3)
        normal_forces = contact_forces[:, :, 1].abs()
        min_normal_force = torch.min(normal_forces, dim=-1).values # both fingers need y-forces
        gripper_contact_detected = min_normal_force > self.cfg.grasp_force_threshold
        gripper_positions = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        current_gripper_width = gripper_positions.sum(dim=-1)
        grip_width_l2 = torch.abs(current_gripper_width - self.obj_width)

        object_height = object_pos[:, 2]
        lift_height = object_height - self.table_height

        # Phase success / bonus
        reached: torch.Tensor = ((distance_l2 < self.cfg.reach_pos_threshold) &
                 (orientation_error < self.cfg.reach_rot_threshold))
        grasped = gripper_contact_detected
        lifted = lift_height > self.cfg.minimal_lift_height

        # Compute bonuses before updating success flags
        reach_bonus = torch.where(reached, 1.0, 0.0)
        grasp_bonus = torch.where(reached & grasped, 1.0, 0.0)
        lift_bonus = torch.where(grasped & lifted, 1.0, 0.0)
        success_bonus = torch.where(
            (reached & grasped) & ~self.grasped_object, 
            1.0, 
            0.0
        )

        # Update flags
        self.reached_object |= reached
        self.grasped_object |= (self.reached_object & grasped)  # Only successful if reached already in this episode
        self.lifted_object |= (self.grasped_object & lifted)    # Only successful if grasped already in this episode

        # Distance target to obj
        distance_obj_target_l2 = torch.norm(target_pos - object_pos, dim=1)
        distance_obj_target_tanh =  1 - torch.tanh(distance_obj_target_l2 / 0.3)
        distance_obj_target_tanh_fine =  1 - torch.tanh(distance_obj_target_l2 / 0.05)
        
        # --- Penalties ---
        action_l2 = torch.sum(torch.square(self.actions), dim=1)
        action_diff = self.actions - self.previous_actions
        action_rate_l2 = torch.sum(torch.square(action_diff), dim=1)
        action_rate_limit = self._action_rate_limit(all_joint_ids, action_diff, threshold_ratio=0.1)
        joint_vel_l2 = torch.sum(torch.square(self.robot.data.joint_vel[:, self.joint_ids]), dim=1)
        joint_pos_limit = self._joint_pos_limits(all_joint_ids)
        joint_vel_limit = self._joint_vel_limits(all_joint_ids, soft_ratio=1.0)
        min_link_distance = self._minimum_link_distance(min_dist=0.1)

        # Phase 1: Reach (always active)
        reach_reward = -0.2 * distance_l2 - 0.1 * orientation_error + torch.where(reached, 1.0, 0.0)
        # Phase 2: Grasp maintenance
        grasp_reward = torch.where(grasped, 1.0, 0.0)
        lift_reward = torch.where(lifted, 1.0, 0.0)
        # --- Total weighted reward ---
        task_reward = (
            # Task rewards
            self.cfg.distance_l2_w * distance_l2 +
            self.cfg.orientation_error_w * orientation_error +
            self.cfg.reach_bonus_w * reach_bonus +
            self.cfg.grasp_bonus_w * grasp_bonus +
            # Success bonus
            self.cfg.success_bonus_w * success_bonus
        )

        penalty = (        
            # Regularization penalties
            self.cfg.action_l2_w * action_l2 +
            # self.cfg.action_rate_l2_w * action_rate_limit +
            # Safety limits
            self.cfg.joint_pos_limit_w * joint_pos_limit +
            self.cfg.joint_vel_limit_w * joint_vel_limit
        )
        reward = task_reward - penalty

        # --- Logging ---
        self.episode_reward += reward
        self.episode_success = self.episode_success | grasped
        self.distance_l2 = distance_l2
        self.target_l2 = distance_obj_target_l2
        self.grip_width_l2 = grip_width_l2
        self.lift_height = lift_height

        self.extras["log"]["success_rate"] = grasped.float().mean().item()
        self.extras["Metrics/success_rate"] = reached.float().mean().item()
        self.extras["Debug/distance_mean"] = distance_l2.mean().item()
        self.extras["Debug/distance_mode"] = distance_l2.mode()
        self.extras["Debug/distance_min"] = distance_l2.min().item()
        self.extras["Debug/distance_max"] = distance_l2.max().item()
        self.extras["Debug/action_mean"] = self.actions.abs().mean().item()
        self.extras["Debug/action_max"] = self.actions.abs().max().item()
        self.extras["Debug/reward"] = reward.mean().item()

        print("--- Reward Debug ---")
        print(f"Common Step Counter: {self.common_step_counter}")
        print(f"Reach bonus: mean={reach_bonus.mean().item():.2f}, min={reach_bonus.min().item():.2f}, max={reach_bonus.max().item():.2f}")
        print(f"Grasp bonus: mean={grasp_bonus.mean().item():.2f}, min={grasp_bonus.min().item():.2f}, max={grasp_bonus.max().item():.2f}")
        print(f"Lift bonus: mean={lift_bonus.mean().item():.2f}, min={lift_bonus.min().item():.2f}, max={lift_bonus.max().item():.2f}")
        print(f"Success bonus: mean={success_bonus.mean().item():.2f}, min={success_bonus.min().item():.2f}, max={success_bonus.max().item():.2f}")
        
        print(f"Reach reward: mean={reach_reward.mean().item():.2f}, min={reach_reward.min().item():.2f}, max={reach_reward.max().item():.2f}")
        print(f"Grasp reward: mean={grasp_reward.mean().item():.2f}, min={grasp_reward.min().item():.2f}, max={grasp_reward.max().item():.2f}")
        
        print(f"Task reward: mean={task_reward.mean().item():.2f}, min={task_reward.min().item():.2f}, max={task_reward.max().item():.2f}")
        print(f"Penalty: mean={penalty.mean().item():.2f}, min={penalty.min().item():.2f}, max={penalty.max().item():.2f}")
        print(f"Total reward: mean={reward.mean().item():.2f}, min={reward.min().item():.2f}, max={reward.max().item():.2f}")
        
        print(f"Reach distance: mean={distance_l2.mean().item():.2f}, min={distance_l2.min().item():.2f}, max={distance_l2.max().item():.2f}")
        print(f"Reach rotation error: mean={orientation_error.mean().item():.2f}, min={orientation_error.min().item():.2f}, max={orientation_error.max().item():.2f}")
        print(f"Lift height: mean={lift_height.mean().item():.2f}, min={lift_height.min().item():.2f}, max={lift_height.max().item():.2f}")
        
        print(f"Reached object? {self.reached_object.float().mean().item():.2f}")
        print(f"Grasped object? {self.grasped_object.float().mean().item():.2f}")
        print(f"Lifted object? {self.lifted_object.float().mean().item():.2f}")
        print("-------------------")
        

        return reward

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

        out_of_bound = self._out_of_bound(
            in_bound_range={
                "x": (0.30, 1.0), 
                "y": (-0.5, 0.5), 
                "z": (0.0, 1.0),
                "roll": (-0.785, 0.785),   # ±45° tilt
                "pitch": (-0.785, 0.785),  # ±45° tilt
            }
        )

        died = out_of_bound | joint_pos_limit_violation | minimum_link_distance_violation
        
        if died.any():
            # print(f"Episode terminated due to safety limit violation in {died.sum().item()} envs.")
            self.extras["Termination/out_of_bound"] = out_of_bound.sum().item()
            self.extras["Termination/joint_pos_limit_violation"] = joint_pos_limit_violation.sum().item()
            self.extras["Termination/joint_vel_limit_violation"] = joint_vel_limit_violation.sum().item()
            self.extras["Termination/minimum_link_distance_violation"] = minimum_link_distance_violation.sum().item()

        return died, time_out
    
    def _out_of_bound(
        self,
        in_bound_range: dict[str, tuple[float, float]] = {},
    ) -> torch.Tensor:
        """Termination condition for the object falls out of bound or tips over.

        Args:
            in_bound_range: The range in x, y, z, roll, pitch, yaw such that the object is considered in bounds.
                        Keys: "x", "y", "z", "roll", "pitch", "yaw"
                        Values: (min, max) tuples
                        Missing keys default to no constraint (very large range)
        """
        object_frame = self.scene['object_frame']
        object_pos_local = object_frame.data.target_pos_source[..., 0, :]
        object_quat_local = object_frame.data.target_quat_source[..., 0, :]

        # Get position and orientation
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(object_quat_local)
        pose = torch.stack([object_pos_local[:, 0], object_pos_local[:, 1], object_pos_local[:, 2], 
                            roll, pitch, yaw], dim=1)  # (num_envs, 6)

        # Check bounds (default to very large range = no constraint)
        range_list = [in_bound_range.get(key, (-1e6, 1e6)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)

        outside_bounds = ((pose < ranges[:, 0]) | (pose > ranges[:, 1])).any(dim=1)

        
        return outside_bounds
        
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # Log before reset
        self._log_episode_stats(env_ids)
        
        # Call grandparent reset
        DirectRLEnv._reset_idx(self, env_ids)
        
        # Reset success flags for the reset environments
        self.reached_object[env_ids] = False
        self.grasped_object[env_ids] = False
        self.lifted_object[env_ids] = False
        
        # Reset episode reward for the reset environments
        self.episode_reward[env_ids] = 0.0
        
        # Reset per-step tracking vars for the reset environments
        self.target_l2[env_ids] = 0.0
        self.grip_width_l2[env_ids] = 0.0
        self.lift_height[env_ids] = 0.0
        
        if self.cfg.randomize_joints:
            self._reset_joints_by_offset(env_ids, position_range=self.cfg.joint_pos_range)
        else:
            self._reset_articulation(env_ids)
        
        self._reset_target_pose(env_ids)
        self._reset_object_uniform(env_ids, self.scene.rigid_objects['object'], pose_range=self.cfg.object_pose_range, velocity_range={})

    def _log_episode_stats(self, env_ids):
        super()._log_episode_stats(env_ids)
        self.extras["log"]["Episode/target_l2"] = self.target_l2.mean().item()
        self.extras["log"]["Episode/grip_width_l2"] = self.grip_width_l2.mean().item()
        self.extras["log"]["Episode/lift_height"] = self.lift_height.mean().item()

        # Compute success rates for the environments being reset
        reach_rate = self.reached_object[env_ids].float().mean().item()
        grasp_rate = self.grasped_object[env_ids].float().mean().item()
        lift_rate = self.lifted_object[env_ids].float().mean().item()
        
        # Compute mean episode reward for the environments being reset
        mean_reward = self.episode_reward[env_ids].mean().item()
        self.best_reach_rate = max(self.best_reach_rate, reach_rate)
        self.best_grasp_rate = max(self.best_grasp_rate, grasp_rate)
        self.best_lift_rate = max(self.best_lift_rate, lift_rate)
        self.best_episode_reward = max(self.best_episode_reward, mean_reward)
        
        # Log current episode metrics
        self.extras["episode/reach_rate"] = reach_rate
        self.extras["episode/grasp_rate"] = grasp_rate
        self.extras["episode/lift_rate"] = lift_rate
        self.extras["episode/mean_reward"] = mean_reward
        
        # Log best metrics
        self.extras["best/reach_rate"] = self.best_reach_rate
        self.extras["best/grasp_rate"] = self.best_grasp_rate
        self.extras["best/lift_rate"] = self.best_lift_rate
        self.extras["best/episode_reward"] = self.best_episode_reward
        
        # Log number of episodes reset
        self.extras["episode/reset_count"] = len(env_ids)


    def _reset_episode_stats(self):
        super()._reset_episode_stats()
        self.target_l2 = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.grip_width_l2 = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.lift_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _reset_rigid_objects(self, env_ids):
        # reset rigid objects to_default
        for rigid_object in self.scene.rigid_objects.values():
            # obtain default and deal with the offset for env origins
            default_root_state = rigid_object.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:3] += self.scene.env_origins[env_ids]
            # set into the physics simulation
            rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    def _reset_object_uniform(
        self,
        env_ids: torch.Tensor,
        asset: RigidObject,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
    ):
        """Reset the asset root state to a random position and velocity uniformly within the given ranges.

        This function randomizes the root position and velocity of the asset.

        * It samples the root position from the given ranges and adds them to the default root position, before setting
        them into the physics simulation.
        * It samples the root orientation from the given ranges and sets them into the physics simulation.
        * It samples the root velocity from the given ranges and sets them into the physics simulation.

        The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
        dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
        ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
        """
        # get default root state
        root_states = asset.data.default_root_state[env_ids].clone()

        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        positions = root_states[:, 0:3] + self.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
