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
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.markers.config import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.math import subtract_frame_transforms
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.reach.reach_env import ReachEnv
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.grasp.grasp_env import GraspEnv
from source.ur10e_sim2real.ur10e_sim2real.tasks.direct.pickplace.pickplace_env_cfg import PickPlaceEnvCfg
from isaaclab.utils.math import euler_xyz_from_quat

class PickPlaceEnv(GraspEnv):
    robot: Articulation
    cfg: PickPlaceEnvCfg

    def __init__(self, cfg: PickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        self.robot = self.scene.articulations["robot"]
        self.joint_ids, _ = self.robot.find_joints(self.cfg.joint_names)
        self.gripper_joint_ids, _ = self.robot.find_joints(self.cfg.gripper_joint_names)
        self.all_joint_ids = self.joint_ids.copy()
        self.all_joint_ids.append(self.gripper_joint_ids[0])

        # Define joint limits
        if self.cfg.action_type == "position":
            self.joint_limits_lower = self.robot.data.joint_pos_limits[:, self.joint_ids, 0].clone()
            self.joint_limits_upper = self.robot.data.joint_pos_limits[:, self.joint_ids, 1].clone()
        elif self.cfg.action_type == "velocity":
            self.joint_limits_lower = -self.robot.data.joint_vel_limits[:, self.joint_ids].clone()
            self.joint_limits_upper = self.robot.data.joint_vel_limits[:, self.joint_ids].clone()

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

        self.obj_width = 0.03 # todo: later get this from object
        self.gripper_max_width = 0.05 # fixed
        self.table_height = 0.015  # From init state
        self.max_lift_height = 0.10 # 10cm above table
        self.pre_grasp_offset_z = 0.05  # 5cm above object
        self.curriculum_update_interval = 750 # every 1 full episodes

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

        # Initialize dynamic curriculum
        # Stage Tracking (per env)
        self.current_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.furthest_stage_reached = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Weight adaptation hyperparams (global)
        self.K = 1.0 # Multiplier
        self.mu_min = 0.1
        self.mu_max = 1.0

        self.stages = {
            0: "Reaching",
            1: "Grasping",
            2: "Transporting",
            3: "Placing"
        }
        self.stage_rewards = {
            0: ['distance_ee_pregrasp_l2', 'rot_error_ee_obj_rad'],
            1: ['distance_ee_obj_l2', 'grip_error', 'lift_error'],
            2: ['distance_obj_target_l2', 'rot_error_obj_target_rad'],
            3: ['success_bonus'],
        }
        self.reward_weights = {
            'distance_ee_pregrasp_l2': 0.1, # Stage 0
            'rot_error_ee_obj_rad': 0.1, # Stage 0
            'distance_ee_obj_l2': 0.1,  # Stage 1
            'grip_error': 0.2, # Stage 1
            'lift_error': 2.5, # Stage 1
            'distance_obj_target_l2': 0.2,  # Stage 2
            'rot_error_obj_target_rad': 0.1, # Stage 2
            'success_bonus': 1.0,  # Stage 3
        }

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # initialize metrics
        self._reset_episode_stats()
        if "log" not in self.extras:
            self.extras["log"] = dict()

    def _pre_physics_step(self, actions: torch.Tensor):
        """Pre-process actions before stepping through the physics.
        
        Args:
            actions: The actions to apply on the environment. Shape is (num_envs, 7).
                    First 6 are arm joints, 7th is gripper width command.
        """
        self.previous_actions = self.actions.clone()
        self.actions = actions.clone()

        # Get current state
        ee_frame = self.scene['ee_frame']
        ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
        ee_quat = ee_frame.data.target_quat_w[..., 0, :]
        object_frame = self.scene['object_frame']
        object_pos = object_frame.data.target_pos_w[..., 0, :]
        obj_quat = self.scene['object_frame'].data.target_quat_source[..., 0, :]  # (num_envs, 4)
        target_pos = self.scene['target_frame'].data.target_pos_source[..., 0, :]  # (num_envs, 3)
        target_quat = self.scene['target_frame'].data.target_quat_source[..., 0, :]  # (num_envs, 4)
        
        distance_ee_obj_l2 = torch.norm(object_pos - ee_pos, p=2, dim=-1)
        rot_error_ee_obj_rad = quat_error_magnitude(self.grasp_quat_w, ee_quat)
        reached: torch.Tensor = (distance_ee_obj_l2 < self.cfg.reach_pos_threshold) & \
                  (rot_error_ee_obj_rad < self.cfg.reach_rot_threshold)
        distance_obj_target_l2 = torch.norm(target_pos - object_pos, dim=1)
        rot_error_obj_target_rad = quat_error_magnitude(target_quat, obj_quat)
        transported = (distance_obj_target_l2 < self.cfg.transport_pos_threshold) & \
                  (rot_error_obj_target_rad < self.cfg.transport_rot_threshold)

        # Check if object is grasped
        gripper_positions = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        current_gripper_width = gripper_positions.sum(dim=-1)
        gripper_closed = torch.abs(current_gripper_width - self.obj_width) < self.cfg.grasp_width_threshold
        
        contact_forces = self.scene._sensors['contact_sensor']._data.net_forces_w
        normal_forces = contact_forces[:, :, 1].abs()
        min_normal_force = torch.min(normal_forces, dim=-1).values
        has_contact = min_normal_force > self.cfg.grasp_force_threshold
        
        object_grasped = has_contact #& gripper_closed
        
        # Split actions into arm (6) and gripper (1)
        arm_actions = self.actions[:, :6]  # (num_envs, 6)
        gripper_action = self.actions[:, 6:7]  # (num_envs, 1)

        processed_arm_actions = torch.clamp(
            arm_actions,
            min=self.joint_limits_lower,
            max=self.joint_limits_upper
        )

        # If close to object but not yet grasped: freeze arm
        attempting_grasp = reached & ~object_grasped

        # processed_arm_actions[attempting_grasp] = 0.0

        # Gripper control (scripted)
        gripper_open_pos = self.gripper_max_width / 2  # Each finger at max
        gripper_closed_pos = self.obj_width / 2  # Each finger at object width/2

        # If grasped OR attempting grasp: close gripper
        # Otherwise: open gripper
        should_close_gripper = (object_grasped | attempting_grasp) & ~transported
        finger_position = torch.where(
            should_close_gripper.unsqueeze(-1),
            torch.full((self.num_envs, 1), 0.0, device=self.device),
            torch.full((self.num_envs, 1), gripper_open_pos, device=self.device)
        )

        # # Clamp gripper action to [-1, 1]
        # gripper_action = torch.clamp(gripper_action, min=-1.0, max=1.0)
        # # Process gripper action (map from [-1, 1] to actual width, then split between fingers)
        # gripper_max_width = 0.05
        # # Map action from [-1, 1] to [0, gripper_max_width]
        # target_width = (gripper_action + 1.0) / 2.0 * gripper_max_width  # (num_envs, 1)
        # # Each finger should be at half the total width
        # finger_position = target_width / 2.0  # (num_envs, 1)
        
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
        joint_velocities = self.robot.data.joint_vel[:, self.joint_ids]  # (num_envs, 6)
        
        # End-effector pose
        ee_pos = self.scene['ee_frame'].data.target_pos_source[..., 0, :]  # (num_envs, 3)
        ee_quat = self.scene['ee_frame'].data.target_quat_source[..., 0, :]  # (num_envs, 4)

        # Object pose
        obj_pos = self.scene['object_frame'].data.target_pos_source[..., 0, :]  # (num_envs, 3)
        obj_quat = self.scene['object_frame'].data.target_quat_source[..., 0, :]  # (num_envs, 4)
        
        # Target pose
        target_pos = self.scene['target_frame'].data.target_pos_source[..., 0, :]  # (num_envs, 3)
        target_quat = self.scene['target_frame'].data.target_quat_source[..., 0, :]  # (num_envs, 4)
        
        # Normalize target width to [-1, 1]
        target_width_normalized = 2.0 * (self.obj_width / self.gripper_max_width) - 1.0
        target_width = torch.full((self.num_envs, 1), target_width_normalized, device=self.device)
        
        # Get current gripper width (sum of both finger positions)
        gripper_positions = self.robot.data.joint_pos[:, self.gripper_joint_ids]  # (num_envs, 2)
        current_gripper_width = gripper_positions.sum(dim=-1)  # (num_envs,)
        current_width_normalized = 2.0 * (current_gripper_width / self.gripper_max_width) - 1.0
        gripper_width = current_width_normalized.unsqueeze(-1)  # (num_envs, 1)
        
        obs = torch.cat(
            (
                joint_positions,          # (num_envs, 6) [-1, 1]
                joint_velocities,         # (num_envs, 6) [-1, 1]
                ee_pos,                   # (num_envs, 3)
                ee_quat,                  # (num_envs, 4)
                obj_pos,                   # (num_envs, 3)
                obj_quat,                  # (num_envs, 4)
                target_pos,               # (num_envs, 3)
                target_quat,              # (num_envs, 4)
                gripper_width,              # (num_envs, 1) [-1, 1]
                target_width,      # (num_envs, 1) [-1, 1]
                self.previous_actions,    # (num_envs, 7)
            ),
            dim=-1,
        )
        
        return {"policy": obs}
            
    def _get_rewards(self) -> torch.Tensor:
        # get scene entities
        ee_frame = self.scene['ee_frame']
        ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
        ee_quat = ee_frame.data.target_quat_w[..., 0, :]
        target_frame = self.scene['target_frame']
        target_pos = target_frame.data.target_pos_w[..., 0, :]
        target_quat = target_frame.data.target_quat_w[..., 0, :]
        object_frame = self.scene['object_frame']
        object_pos = object_frame.data.target_pos_w[..., 0, :]
        object_quat = object_frame.data.target_quat_w[..., 0, :]
        # Pre-grasp position (hover above object)
        self.grasp_pos_w = object_pos + self.grasp_offset
        _, _, obj_yaw = math_utils.euler_xyz_from_quat(object_quat)
        self.grasp_quat_w = torch_utils.quat_from_euler_xyz(
            torch.ones_like(obj_yaw) * torch.pi,  # Roll = π (Z down)
            torch.zeros_like(obj_yaw),             # Pitch = 0
            obj_yaw                                # Yaw = match current object
        )
        gripper_positions = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        current_gripper_width = gripper_positions.sum(dim=-1)
        object_height = object_pos[:, 2]

        # compute rewards terms
        distance_ee_pregrasp_l2 = torch.norm(self.grasp_pos_w - ee_pos, p=2, dim=-1)
        distance_ee_obj_l2 = torch.norm(object_pos - ee_pos, p=2, dim=-1)
        distance_ee_obj_tanh = 1.0 - torch.tanh(torch.norm(object_pos - ee_pos, p=2, dim=-1) / self.cfg.distance_ee_obj_tanh_std)
        distance_ee_obj_tanh_fine = 1.0 - torch.tanh(distance_ee_obj_l2 / self.cfg.distance_ee_obj_tanh_fine_std)
        rot_error_ee_obj_rad = quat_error_magnitude(self.grasp_quat_w, ee_quat)
        rot_error_ee_obj_tanh = 1.0 - torch.tanh(rot_error_ee_obj_rad / self.cfg.rot_error_ee_obj_tanh_std)
        grip_width_l2 = torch.abs(current_gripper_width - self.obj_width)
        grip_width_tanh = 1.0 - torch.tanh(grip_width_l2 / self.cfg.grip_width_tanh_std)
        # 0 if error is 0, 1 if error is maximal
        grip_error = torch.clamp(grip_width_l2 / self.gripper_max_width, min=0.0, max=1.0)
        # Only reward gripper closing when close to object
        close_to_object = distance_ee_obj_l2 < (self.obj_width / 2)  # (assuming cube)
        grip_error_conditional = torch.where(
            close_to_object,
            grip_width_tanh,
            torch.zeros_like(grip_width_tanh)
        )
        # 0 if not lifted, 1 if lifted to target
        lift_height = object_height - self.table_height
        lift_error_l2 = self.max_lift_height - lift_height
        lift_error = torch.clamp(lift_error_l2 / self.max_lift_height, min=0.0, max=1.0)
        distance_obj_target_l2 = torch.norm(target_pos - object_pos, dim=1)
        distance_obj_target_tanh = 1 - torch.tanh(distance_obj_target_l2 / self.cfg.distance_obj_target_tanh_std)
        distance_obj_target_tanh_fine = 1 - torch.tanh(distance_obj_target_l2 / self.cfg.distance_obj_target_tanh_fine_std)
        rot_error_obj_target_rad = quat_error_magnitude(target_quat, object_quat)
        obj_lin_vel = torch.norm(self.scene.rigid_objects['object'].data.root_com_lin_vel_w , dim=1)
        obj_ang_vel = torch.norm(self.scene.rigid_objects['object'].data.root_com_ang_vel_w , dim=1)

        # compute penalty terms
        action_l2 = torch.sum(torch.square(self.actions), dim=1)
        action_rate_l2 = torch.sum(torch.square(self.actions - self.previous_actions), dim=1)
        joint_pos_limit = self._joint_pos_limits(self.joint_ids)
        joint_vel_limit = self._joint_vel_limits(self.joint_ids, soft_ratio=1.0)
        min_link_distance = self._minimum_link_distance(min_dist=0.1)

        # Stage Transitions
        # S0 -> S1: Reached object within threshold
        reached: torch.Tensor = (distance_ee_obj_l2 < self.cfg.reach_pos_threshold) & \
                  (rot_error_ee_obj_rad < self.cfg.reach_rot_threshold)
        if reached.any():
            print(f"Reached for {reached.sum().item()} envs.")

        # S1 → S2: Grasped object and lifted to minimal height
        gripper_closed_correctly = grip_width_l2 < self.cfg.grasp_width_threshold
        contact_forces = self.scene._sensors['contact_sensor']._data.net_forces_w # (num_envs, 2, 3)
        normal_forces = contact_forces[:, :, 1].abs()
        min_normal_force = torch.min(normal_forces, dim=-1).values # both fingers need y-forces
        gripper_contact_detected = min_normal_force > self.cfg.grasp_force_threshold # Force threshold (nm)
        lifted = lift_height > self.cfg.minimal_lift_height 
        if lifted.any():
            print(f"Lifted for {lifted.sum().item()} envs.")

        grasped = gripper_contact_detected # & gripper_closed_correctly
        if grasped.any():
            print(f"Grasped for {grasped.sum().item()} envs.")

        # S2 → S3: Transported object
        transported = (distance_obj_target_l2 < self.cfg.transport_pos_threshold) & \
                  (rot_error_obj_target_rad < self.cfg.transport_rot_threshold)
        
        if transported.any():
            print(f"Transported for {transported.sum().item()} envs.")

        # S4 → Success: Placed object (released + stable + at goal)
        placed = (distance_obj_target_l2 < self.cfg.place_pos_threshold) & \
                  (rot_error_obj_target_rad < self.cfg.place_rot_threshold) & \
                 (grip_width_l2 > self.obj_width) & \
                 ~gripper_contact_detected & \
                 (obj_lin_vel < 0.1) & (obj_ang_vel < 0.1)

        # Update stages
        self._progress_stages_vectorized(reached, grasped, transported, placed)

        reach_bonus = torch.where(
            reached & (~self.reach_success),
            torch.ones_like(grasped, dtype=torch.float),
            torch.zeros_like(grasped, dtype=torch.float)
        )

        grasp_bonus = torch.where(
            grasped & (~self.grasp_success),
            torch.ones_like(grasped, dtype=torch.float),
            torch.zeros_like(grasped, dtype=torch.float)
        )

        lift_bonus = torch.where(
            lifted & (~self.lift_success),
            torch.ones_like(lifted, dtype=torch.float),
            torch.zeros_like(lifted, dtype=torch.float)
        )

        transport_bonus = torch.where(
            transported & (~self.transport_success),
            torch.ones_like(transported, dtype=torch.float),
            torch.zeros_like(transported, dtype=torch.float)
        )

        # Compute success bonus (only once per episode)
        success_bonus = torch.where(
            placed & (~self.place_success),
            torch.ones_like(placed, dtype=torch.float),
            torch.zeros_like(placed, dtype=torch.float)
        )
        self.place_success = self.place_success | placed
        
        # Task Reward with dynamic weights
        # task_reward = (
        #     - self.reward_weights['distance_ee_pregrasp_l2'] * distance_ee_obj_l2 + 
        #     - self.reward_weights['rot_error_ee_obj_rad'] * rot_error_ee_obj_rad +
        #     - self.reward_weights['distance_ee_obj_l2'] * distance_ee_obj_l2 +  
        #     - self.reward_weights['grip_error'] * grip_error_conditional +
        #     - self.reward_weights['lift_error'] * lift_error +
        #     - self.reward_weights['distance_obj_target_l2'] * distance_obj_target_l2 +
        #     # - self.reward_weights['rot_error_obj_target_rad'] * rot_error_obj_target_rad +
        #     # 10.0 * reach_bonus +
        #     # 20.0 * grasp_bonus +
        #     # 30.0 * lift_bonus +
        #     # 40.0 * transport_bonus +
        #     1000.0 * success_bonus
        # )
        # Only reward gripper closing when close to object
        grip_width_reward = torch.where(
            reached,
            grip_width_tanh,
            torch.zeros_like(grip_width_tanh)
        )
        lift_reward = torch.clamp(lift_height / self.max_lift_height, min=0.0, max=1.0)
        task_reward = (
            -2.0 * distance_ee_obj_l2 +
            -1.0 * rot_error_ee_obj_rad +
            # -2.0 * distance_obj_target_l2 +
            # -1.0 * rot_error_obj_target_rad +
            2.0 * grip_width_reward + 
            5.0 * lift_reward +
            100.0 * reach_bonus
            # 200.0 * grasp_bonus +
            # 300.0 * lift_bonus + 
            # 400.0 * transport_bonus
        )
        
        # Penalties
        penalty = (
            # Regularization penalties
            - self.cfg.action_l2_w * action_l2 +
            - self.cfg.action_rate_l2_w * action_rate_l2 +
            # Safety limits
            - self.cfg.joint_pos_limit_w * joint_pos_limit +
            - self.cfg.joint_vel_limit_w * joint_vel_limit +
            - self.cfg.min_link_distance_w * min_link_distance
        )
        
        reward = task_reward + penalty

        if self.common_step_counter % self.curriculum_update_interval == 0 and self.common_step_counter > 0:
            self._adapt_weights(self.robot._ALL_INDICES)
            self._print_stage_summary(self.robot._ALL_INDICES)

        # Log stage distribution
        for stage_id, stage_name in self.stages.items():
            count = (self.current_stage == stage_id).sum().item()
            self.extras[f"Stage/current_{stage_name}"] = count
            
        for stage_id, stage_name in self.stages.items():
            count = (self.furthest_stage_reached == stage_id).sum().item()
            self.extras[f"Stage/furthest_{stage_name}"] = count
        
        # Log current weights
        for key, weight in self.reward_weights.items():
            self.extras[f"Weights/{key}"] = weight

        # Log Rewards
        self.extras["Rewards/task"] = task_reward.mean().item()
        self.extras["Rewards/penalty"] = penalty.mean().item()
        self.extras["Rewards/total"] = reward.mean().item()

        # Debug logs
        self.extras["Debug/distance_ee_pregrasp_l2"] = distance_ee_pregrasp_l2.mean().item()
        self.extras["Debug/distance_ee_obj_l2"] = distance_ee_obj_l2.mean().item()
        self.extras["Debug/rot_error_ee_obj_rad"] = rot_error_ee_obj_rad.mean().item()
        self.extras["Debug/distance_obj_target_l2"] = distance_obj_target_l2.mean().item()
        self.extras["Debug/grip_width_l2"] = grip_width_l2.mean().item()
        self.extras["Debug/lift_error_l2"] = lift_error_l2.mean().item()
        self.extras["Debug/action_magnitude"] = self.actions.abs().mean().item()
        self.extras["Debug/max_joint_velocity"] = self.robot.data.joint_vel[:, self.joint_ids].abs().max().item()

        # --- Logging ---
        self.episode_reward += reward
        self.distance_l2 = distance_ee_obj_l2
        self.target_l2 = distance_obj_target_l2
        self.grip_width_l2 = grip_width_l2
        self.lift_height = lift_height

        return reward

    def _progress_stages_vectorized(self, reached, grasped, transported, placed):
        """
        Vectorized stage progression for multiple environments.
        Inputs are tensors of shape [num_envs] with boolean values.
        """
        current = self.current_stage  # shape: [num_envs]

        # Stage 0 → 1 (Reach → Grasp)
        mask0 = (current == 0) & reached
        self.current_stage[mask0] = 1

        # Stage 1 → 2 (Grasp → Transport) or back to 1 if dropped
        mask1_up = (current == 1) & grasped
        mask1_down = (current == 1) & (~grasped)
        self.current_stage[mask1_up] = 2
        self.current_stage[mask1_down] = 1

        # Stage 2 → 3 (Transport → Place) or back to 1 if dropped
        mask2_up = (current == 2) & transported
        mask2_down = (current == 2) & (~grasped)
        self.current_stage[mask2_up] = 3
        self.current_stage[mask2_down] = 1

        # Stage 3 → check placed, or back to 1 if dropped
        mask3_done = (current == 3) & placed
        mask3_down = (current == 3) & (~grasped)
        # self.current_stage[mask3_done] stays at 3 (success!)
        self.current_stage[mask3_down] = 1

        # Track furthest stage reached (never goes backward)
        self.furthest_stage_reached = torch.maximum(
            self.furthest_stage_reached,
            self.current_stage
        )

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
        # spawn position: 0.74687, 0.17399, 0.015
        # then randomized x": (-0.1, 0.1), "y": (-0.1, 0.1)
        out_of_bound = self._out_of_bound(in_bound_range={"x": (0.30, 1.0), "y": (-0.5, 0.5), "z": (0.0, 1.0)})

        died = out_of_bound | joint_pos_limit_violation | minimum_link_distance_violation
        
        if died.any():
            # print(f"Episode terminated due to safety limit violation in {died.sum().item()} envs.")
            self.extras["Termination/out_of_bound"] = out_of_bound.sum().item()
            self.extras["Termination/joint_pos_limit_violation"] = joint_pos_limit_violation.sum().item()
            self.extras["Termination/joint_vel_limit_violation"] = joint_vel_limit_violation.sum().item()
            self.extras["Termination/minimum_link_distance_violation"] = minimum_link_distance_violation.sum().item()
        
        self.extras["Termination/time_out"] = time_out.sum().item()

        return died, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._log_episode_stats(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)
        # Reset parent (robot, articulation, stats, etc.)
        self._reset_episode_stats()

        if self.cfg.randomize_joints:
            self.__reset_joints_by_offset(env_ids, position_range=(-1.0, 1.0))
        else:
            self._reset_articulation(env_ids)

        # Reset the object
        self._reset_object_uniform(env_ids, self.scene.rigid_objects['object'], pose_range=self.cfg.object_pose_range, velocity_range={})
        # Reset the target
        self._reset_object_uniform(env_ids, self.scene.rigid_objects['target'], pose_range=self.cfg.target_pose_range, velocity_range={})

        # Reset grasp pos/quat
        object_frame = self.scene['object_frame']
        object_pos_w = object_frame.data.target_pos_w[..., 0, :]
        object_quat_w = object_frame.data.target_quat_w[..., 0, :]
        _, _, obj_yaw = math_utils.euler_xyz_from_quat(object_quat_w) 
        self.grasp_pos_w = object_pos_w + self.grasp_offset
        self.grasp_quat_w = torch_utils.quat_from_euler_xyz(
            torch.ones_like(obj_yaw) * torch.pi, # Roll = π (gripper upside down, Z pointing down)
            torch.zeros_like(obj_yaw), # Pitch = 0
            obj_yaw, # Match object yaw
        )


        # --- Stage summary print ---

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
    
    def _out_of_bound(
        self,
        in_bound_range: dict[str, tuple[float, float]] = {},
    ) -> torch.Tensor:
        """Termination condition for the object falls out of bound.

        Args:
            in_bound_range: The range in x, y, z such that the object is considered in range
        """
        object_frame = self.scene['object_frame']
        object_pos_local = object_frame.data.target_pos_source[..., 0, :]

        range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.device)

        outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
        return outside_bounds
    
    def _print_stage_summary(self, env_ids: Sequence[int]):
        furthest_stages = self.furthest_stage_reached[env_ids].clone()
        current_stages = self.current_stage[env_ids].clone()

        # Count how many envs reached each stage
        counts = torch.bincount(furthest_stages, minlength=len(self.stages))
        print("=== Stage Summary for Reset ===")
        for i, count in enumerate(counts):
            if count > 0:
                print(f"🚀 {count.item()} env(s) reached stage {i} ({self.stages[i]})")

        # Furthest stage reached
        max_stage = furthest_stages.max().item()
        print(f"🏁 Furthest stage reached this episode: {max_stage} ({self.stages[max_stage]})")

        # Backwards drops (envs that went back at least one stage)
        drops = furthest_stages > current_stages
        if drops.any():
            print("⚠️ Backward drops during episode:")
            for stage in range(len(self.stages)):
                stage_mask = (furthest_stages == stage) & (current_stages < stage)
                if stage_mask.any():
                    num_dropped = stage_mask.sum().item()
                    print(f"   ⚠️ {num_dropped} env(s) dropped to stage {current_stages[stage_mask][0].item()} → {stage} ({self.stages[stage]})")

    def _adapt_weights(self, env_ids: Sequence[int] | None):
        # What stage did most envs reach?
        stages = self.furthest_stage_reached[env_ids]

        if len(env_ids) == 0:
            return
        
        # Find most common stage reached
        most_common_stage = torch.mode(stages).values.item()

        # Log before adaptation
        self.extras["Curriculum/stage_transitions"] = torch.bincount(self.current_stage, minlength=5).cpu().tolist()
        self.extras["Curriculum/stage_mode"] = most_common_stage
        self.extras["Curriculum/stage_mean"] = stages.float().mean().item()
        
        # Adapt weights
        all_reward_keys = set(self.reward_weights.keys())
        stage_keys = set(self.stage_rewards.get(most_common_stage, []))
        
        # Increase weights for this stage's rewards
        for key in stage_keys:
            old_weight = self.reward_weights[key]
            new_weight = min(old_weight * self.K, self.mu_max)
            self.reward_weights[key] = new_weight
            
            if new_weight != old_weight:
                print(f"  ⬆️ {key}: {old_weight:.3f} → {new_weight:.3f}")
        
        # Decrease weights for other stages
        for key in (all_reward_keys - stage_keys):
            old_weight = self.reward_weights[key]
            new_weight = max(old_weight / self.K, self.mu_min)
            self.reward_weights[key] = new_weight
            
            if new_weight != old_weight:
                print(f"  ⬇️ {key}: {old_weight:.3f} → {new_weight:.3f}")

        # Reset stage tracking
        self.current_stage[env_ids] = 0
        self.furthest_stage_reached[env_ids] = 0

    def _reset_episode_stats(self):
        """Reset episode-level tracking variables."""
        super()._reset_episode_stats() 
        self.reach_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasp_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.transport_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.place_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _log_episode_stats(self, env_ids):
        self.extras["Episode/distance_l2"] = self.distance_l2.mean().item()
        self.extras["Episode/target_l2"] = self.target_l2.mean().item()
        self.extras["Episode/grip_width_l2"] = self.grip_width_l2.mean().item()
        self.extras["Episode/lift_height"] = self.lift_height.mean().item()
        self.extras["Episode/reach_success"] = self.reach_success.float().mean().item()
        self.extras["Episode/grasp_success"] = self.grasp_success.float().mean().item()
        self.extras["Episode/lift_success"] = self.lift_success.float().mean().item()
        self.extras["Episode/transport_success"] = self.transport_success.float().mean().item()
        self.extras["Episode/place_success"] = self.place_success.float().mean().item()


    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(
            translations=self.grasp_pos_w,
            orientations=self.grasp_quat_w,
        )
