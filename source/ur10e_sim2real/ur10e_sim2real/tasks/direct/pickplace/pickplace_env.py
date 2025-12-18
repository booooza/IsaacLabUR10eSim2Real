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
        self.all_joint_ids.extend(self.gripper_joint_ids)

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
        self.pre_grasp_offset_z = 0.05  # 5cm above object
        self.curriculum_update_interval = 350 # every half episodes

        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.target_cfg)

        # initialize target position w.r.t. robot base
        self.target_pos_low  = torch.tensor([0.50,  0.15, 0.015], device=self.device)
        self.target_pos_high = torch.tensor([0.70,  0.35, 0.015], device=self.device)
        # # Front-right of robot (60cm in front of robot base)
        self.target_pos = torch.tensor([[0.60, 0.25, 0.015]], device=self.device).repeat(self.num_envs, 1)
        self.target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1)  # Identity quat (Z up)
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
        # Initialize success tracking tensors
        self.reached_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasped_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.transported_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.placed_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Episode tracking
        self.episode_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Overall metrics
        self.best_reach_rate = 0.0
        self.best_grasp_rate = 0.0
        self.best_lift_rate = 0.0
        self.best_transport_rate = 0.0
        self.best_place_rate = 0.0
        self.best_episode_reward = float('-inf')
        
        # Initialize dynamic curriculum
        # Stage Tracking (per env)
        self.current_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.furthest_stage_reached = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Weight adaptation hyperparams (global)
        self.K = 1.0 # Multiplier
        self.mu_min = 0.1
        self.mu_max = 100.0

        self.stages = {
            0: "Picking",
            1: "Lifting",
            2: "Placing",
            3: "Done",
        }
        self.stage_rewards = {
            0: ['distance_ee_obj_l2', 'rot_error_ee_obj_rad', 'reach_reward', 'reach_bonus'],
            1: ['lift_reward', 'lift_bonus'],
            2: ['distance_obj_target_l2', 'rot_error_obj_target_rad', 'place_bonus'],
            3: [],
        }
        self.reward_weights = {
            'distance_ee_obj_l2': 0.2,  # Stage 0
            'rot_error_ee_obj_rad': 0.1, # Stage 0
            'reach_reward': 1.0,          # Stage 0
            'reach_bonus': 10.0,          # Stage 0
            'lift_reward': 1.0,        # Stage 1
            'lift_bonus': 10.0,        # Stage 1
            'distance_obj_target_l2': 0.2,  # Stage 2
            'rot_error_obj_target_rad': 0.1, # Stage 2
            'place_bonus': 100.0,         # Stage 2
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
        obj_quat = self.scene['object_frame'].data.target_quat_w[..., 0, :]  # (num_envs, 4)
        # target_pos = self.scene['target_frame'].data.target_pos_source[..., 0, :]  # (num_envs, 3)
        # target_quat = self.scene['target_frame'].data.target_quat_source[..., 0, :]  # (num_envs, 4)
        target_pos = self.target_pos_w  # (num_envs, 3)
        target_quat = self.target_quat_w  # (num_envs, 4)
        
        distance_ee_obj_l2 = torch.norm(object_pos - ee_pos, p=2, dim=-1)
        rot_error_ee_obj_rad = quat_error_magnitude(self.grasp_quat_w, ee_quat)
        reached: torch.Tensor = (distance_ee_obj_l2 < self.cfg.reach_pos_threshold) & \
                  (rot_error_ee_obj_rad < self.cfg.reach_rot_threshold)
        distance_obj_target_l2 = torch.norm(target_pos - object_pos, dim=1)
        rot_error_obj_target_rad = quat_error_magnitude(target_quat, obj_quat)

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
        # gripper_action = self.actions[:, 6:7]  # (num_envs, 1)

        processed_arm_actions = torch.clamp(
            arm_actions,
            min=self.joint_limits_lower,
            max=self.joint_limits_upper
        )

        # If close to object but not yet grasped
        attempting_grasp = reached & ~object_grasped
        
        # Gripper control (scripted)
        gripper_open_pos = self.gripper_max_width / 2  # Each finger at max
        gripper_closed_pos = self.obj_width / 2  # Each finger at object width/2

        # If the object is close to the target AND currently grasped: open gripper
        should_open_gripper = (
            (distance_obj_target_l2 < self.cfg.place_pos_threshold) &
            object_grasped  # Changed from self.grasped_object
        )

        # If grasped OR attempting grasp: close gripper
        # Otherwise: open gripper
        should_close_gripper = (object_grasped | attempting_grasp) & ~should_open_gripper
        finger_position = torch.where(
            should_close_gripper.unsqueeze(-1),
            torch.full((self.num_envs, 1), 0.0, device=self.device),
            torch.full((self.num_envs, 1), gripper_open_pos, device=self.device)
        )

        # Binary control
        # should_close_gripper = gripper_action > 0 # (num_envs, 1)
        # finger_position = torch.where(
        #     should_close_gripper,
        #     torch.full((self.num_envs, 1), 0.0, device=self.device),
        #     torch.full((self.num_envs, 1), gripper_open_pos, device=self.device)
        # )

        # # Clamp gripper action to [-1, 1]
        # gripper_action = torch.clamp(gripper_action, min=-1.0, max=1.0)
        # # Process gripper action (map from [-1, 1] to actual width, then split between fingers)
        # gripper_max_width = 0.05
        # # Map action from [-1, 1] to [0, gripper_max_width]
        # target_width = (gripper_action + 1.0) / 2.0 * gripper_max_width  # (num_envs, 1)
        # # Each finger should be at half the total width
        # finger_position = target_width / 2.0  # (num_envs, 1)
        
        # If object is at target position, override actions to safe home position
        transport_dist = torch.norm(object_pos - self.target_pos_w, dim=-1)
        placed = (transport_dist < self.cfg.place_pos_threshold) & ~object_grasped
        
        if placed.any():
            # Freeze arm and close gripper
            placed_env_ids = torch.where(placed)[0]
            processed_arm_actions[placed] = 0.0
            finger_position[placed] = gripper_closed_pos
            
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
        # target_pos = self.scene['target_frame'].data.target_pos_source[..., 0, :]  # (num_envs, 3)
        # target_quat = self.scene['target_frame'].data.target_quat_source[..., 0, :]  # (num_envs, 4)
        target_pos = self.target_pos  # (num_envs, 3)
        target_quat = self.target_quat  # (num_envs, 4)
        
        # Normalize target width to [-1, 1]
        target_width_normalized = 2.0 * (self.obj_width / self.gripper_max_width) - 1.0
        target_width = torch.full((self.num_envs, 1), target_width_normalized, device=self.device)
        
        # Get current gripper width (sum of both finger positions)
        gripper_positions = self.robot.data.joint_pos[:, self.gripper_joint_ids]  # (num_envs, 2)
        current_gripper_width = gripper_positions.sum(dim=-1)  # (num_envs,)
        current_width_normalized = 2.0 * (current_gripper_width / self.gripper_max_width) - 1.0
        gripper_width = current_width_normalized.unsqueeze(-1)  # (num_envs, 1)
        
        obj_pos_rel = obj_pos - ee_pos
        obj_quat_rel = quat_mul(quat_inv(ee_quat), obj_quat)
        target_pos_rel = target_pos - obj_pos
        target_quat_rel = quat_mul(quat_inv(obj_quat), target_quat)

        obs = torch.cat(
            (
                joint_positions,          # (num_envs, 6) [-1, 1]
                joint_velocities,           # (num_envs, 6) 
                ee_pos,                   # (num_envs, 3) try: leave away
                ee_quat,                  # (num_envs, 4) try: leave away
                obj_pos_rel,          # (num_envs, 3)
                obj_quat_rel,                  # (num_envs, 4)
                target_pos_rel,      # (num_envs, 3)
                target_quat_rel,              # (num_envs, 4)
                self.previous_actions,    # (num_envs, 7)
            ),
            dim=-1,
        )
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_frame = self.scene['ee_frame']
        ee_pos = ee_frame.data.target_pos_w[..., 0, :3]
        ee_quat = ee_frame.data.target_quat_w[..., 0, :]
        object_frame = self.scene['object_frame']
        object_pos = object_frame.data.target_pos_w[..., 0, :]

        # Calculations
        reach_dist = torch.norm(ee_pos - object_pos, dim=-1)
        reach_rot_error = quat_error_magnitude(self.grasp_quat_w, ee_quat)

        contact_forces = self.scene._sensors['contact_sensor']._data.net_forces_w # (num_envs, 2, 3)
        normal_forces = contact_forces[:, :, 1].abs()
        min_normal_force = torch.min(normal_forces, dim=-1).values # both fingers need y-forces
        gripper_contact_detected = min_normal_force > self.cfg.grasp_force_threshold
        
        object_height = object_pos[:, 2]
        lift_height = object_height - self.table_height

        transport_dist = torch.norm(object_pos - self.target_pos_w, dim=-1)
        transport_dist_clamped = torch.clamp(transport_dist, 0.0, self.cfg.max_transport_dist)
        # Use XY distance only for transport
        transport_dist_xy = torch.norm(object_pos[:, :2] - self.target_pos_w[:, :2], dim=-1)
        transport_dist_xy_clamped = torch.clamp(transport_dist_xy, 0.0, self.cfg.max_transport_dist)

        # Phase success / bonus
        reached = (reach_dist < self.cfg.reach_pos_threshold) & (reach_rot_error < self.cfg.reach_rot_threshold)
        grasped = gripper_contact_detected
        lifted = lift_height > self.cfg.minimal_lift_height
        # transported = (transport_dist_xy < self.cfg.transport_pos_threshold)
        
        # Object must be stable (low velocity)
        object_vel = self.scene.rigid_objects['object'].data.root_lin_vel_w
        is_stable = torch.norm(object_vel, dim=-1) < 0.05
        placed = (transport_dist_xy < self.cfg.place_pos_threshold) & is_stable & ~grasped

        # Compute bonuses before updating success flags
        reach_bonus = torch.where(reached & ~self.reached_object, 1.0, 0.0)
        grasp_bonus = torch.where(grasped & ~self.grasped_object, 1.0, 0.0)
        lift_bonus = torch.where(lifted & ~self.lifted_object, 1.0, 0.0)
        # transport_bonus = torch.where(transported & ~self.transported_object, 1.0, 0.0)
        place_bonus = torch.where(placed, 100.0, 0.0)

        # Update flags
        self.reached_object |= reached
        self.grasped_object |= grasped
        self.lifted_object |= lifted
        # self.transported_object |= transported
        self.placed_object |= placed

        # Phase 1: Reach (always active)
        reach_reward = -0.2 * reach_dist - 0.1 * reach_rot_error + torch.where(reached, 0.5, 0.0)
        # Phase 2: Grasp maintenance
        grasp_reward = torch.where(grasped, 0.5, 0.0)
        # Phase 3: Lift to target height when grasped and not transported yet
        lift_reward = torch.where(grasped & lifted & (transport_dist_xy > 0.10), 0.1, 0.0)
        # Phase 4: Transport (active only when grasping)
        transport_reward = torch.where(
            grasped & lifted,
            5.0 * (self.cfg.max_transport_dist - transport_dist_clamped) / self.cfg.max_transport_dist,
            0.0
        )
        # Total task reward
        task_reward = (reach_reward + reach_bonus + 
              grasp_reward + grasp_bonus + 
              lift_bonus + transport_reward + place_bonus)
        
        # Penalty terms
        action_l2 = torch.sum(torch.square(self.actions), dim=1)
        action_rate_l2 = torch.sum(torch.square(self.actions - self.previous_actions), dim=1)
        joint_pos_limit = self._joint_pos_limits(self.joint_ids)
        joint_vel_limit = self._joint_vel_limits(self.joint_ids, soft_ratio=1.0)
        min_link_distance = self._minimum_link_distance(min_dist=0.1)
        floor_collision_penalty = torch.where(self._check_robot_floor_collision(floor_threshold=self.table_height), 1.0, 0.0)
        
        penalty = (
            # Regularization penalties
            0.001 * action_l2 +
            0.005 * action_rate_l2 +
            # Safety limits
            0.1 * joint_pos_limit +
            0.1 * joint_vel_limit +
            0.1 * min_link_distance +
            0.1 * floor_collision_penalty
        )
        reward = task_reward - penalty

        print("--- Reward Debug ---")
        print(f"Reach bonus: mean={reach_bonus.mean().item():.2f}, min={reach_bonus.min().item():.2f}, max={reach_bonus.max().item():.2f}")
        print(f"Grasp bonus: mean={grasp_bonus.mean().item():.2f}, min={grasp_bonus.min().item():.2f}, max={grasp_bonus.max().item():.2f}")
        print(f"Lift bonus: mean={lift_bonus.mean().item():.2f}, min={lift_bonus.min().item():.2f}, max={lift_bonus.max().item():.2f}")
        # print(f"Transport bonus: mean={transport_bonus.mean().item():.2f}, min={transport_bonus.min().item():.2f}, max={transport_bonus.max().item():.2f}")
        print(f"Place bonus: mean={place_bonus.mean().item():.2f}, min={place_bonus.min().item():.2f}, max={place_bonus.max().item():.2f}")
        
        print(f"Reach reward: mean={reach_reward.mean().item():.2f}, min={reach_reward.min().item():.2f}, max={reach_reward.max().item():.2f}")
        print(f"Grasp reward: mean={grasp_reward.mean().item():.2f}, min={grasp_reward.min().item():.2f}, max={grasp_reward.max().item():.2f}")
        print(f"Lift reward: mean={lift_reward.mean().item():.2f}, min={lift_reward.min().item():.2f}, max={lift_reward.max().item():.2f}")
        # print(f"Transport reward: mean={transport_xy_reward.mean().item():.2f}, min={transport_xy_reward.min().item():.2f}, max={transport_xy_reward.max().item():.2f}")
        # print(f"Place reward: mean={place_reward.mean().item():.2f}, min={place_reward.min().item():.2f}, max={place_reward.max().item():.2f}")
        
        print(f"Task reward: mean={task_reward.mean().item():.2f}, min={task_reward.min().item():.2f}, max={task_reward.max().item():.2f}")
        print(f"Penalty: mean={penalty.mean().item():.2f}, min={penalty.min().item():.2f}, max={penalty.max().item():.2f}")
        print(f"Total reward: mean={reward.mean().item():.2f}, min={reward.min().item():.2f}, max={reward.max().item():.2f}")
        
        print(f"Reach distance: mean={reach_dist.mean().item():.2f}, min={reach_dist.min().item():.2f}, max={reach_dist.max().item():.2f}")
        print(f"Reach rotation error: mean={reach_rot_error.mean().item():.2f}, min={reach_rot_error.min().item():.2f}, max={reach_rot_error.max().item():.2f}")
        print(f"Lift height: mean={lift_height.mean().item():.2f}, min={lift_height.min().item():.2f}, max={lift_height.max().item():.2f}")
        print(f"Transport distance: mean={transport_dist.mean().item():.2f}, min={transport_dist.min().item():.2f}, max={transport_dist.max().item():.2f}")
        
        print(f"Reached object? {self.reached_object.float().mean().item():.2f}")
        print(f"Grasped object? {self.grasped_object.float().mean().item():.2f}")
        print(f"Lifted object? {self.lifted_object.float().mean().item():.2f}")
        # print(f"Transported object? {self.transported_object.float().mean().item():.2f}")
        print(f"Placed object? {self.placed_object.float().mean().item():.2f} {f'🔥🔥🔥🔥 {placed.sum().item()} envs' if placed.any() else ''}")
        print("-------------------")
        
        self.episode_reward += reward
        
        return reward

    def _progress_stages_vectorized(self, grasped, lifted, placed):
        """
        Vectorized stage progression for multiple environments.
        Inputs are tensors of shape [num_envs] with boolean values.
        """
        current = self.current_stage  # shape: [num_envs]

        # Stage 0 → 1: Grasped
        mask0 = (current == 0) & grasped
        self.current_stage[mask0] = 1

        # Stage 1 → 2: Lifted
        mask1_up = (current == 1) & lifted
        mask1_down = (current == 1) & (~grasped)
        self.current_stage[mask1_up] = 2
        self.current_stage[mask1_down] = 0

        # Stage 2 → 3: Placed
        mask2_up = (current == 2) & placed
        mask2_down = (current == 2) & (~placed) & (~grasped | ~lifted)
        self.current_stage[mask2_up] = 3
        self.current_stage[mask2_down] = 0

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
        out_of_bound = self._out_of_bound(
            in_bound_range={
                "x": (0.30, 1.0), 
                "y": (-0.5, 0.5), 
                "z": (0.0, 1.0),
                "roll": (-0.785, 0.785),   # ±45° tilt
                "pitch": (-0.785, 0.785),  # ±45° tilt
            }
        )
        # Robot-floor collision
        floor_collision = self._check_robot_floor_collision()
        is_danger_of_clamping = self._is_danger_of_clamping()

        died = out_of_bound | joint_pos_limit_violation | minimum_link_distance_violation #| floor_collision | is_danger_of_clamping
        
        if died.any():
            # print(f"Episode terminated due to safety limit violation in {died.sum().item()} envs.")
            self.extras["Termination/out_of_bound"] = out_of_bound.sum().item()
            self.extras["Termination/joint_pos_limit_violation"] = joint_pos_limit_violation.sum().item()
            self.extras["Termination/joint_vel_limit_violation"] = joint_vel_limit_violation.sum().item()
            self.extras["Termination/minimum_link_distance_violation"] = minimum_link_distance_violation.sum().item()
            self.extras["Termination/floor_collision"] = floor_collision.sum().item()
            self.extras["Termination/is_danger_of_clamping"] = is_danger_of_clamping.sum().item()
        
        self.extras["Termination/time_out"] = time_out.sum().item()

        return died, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        self._log_episode_stats(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)
        # Reset parent (robot, articulation, stats, etc.)
        self._reset_episode_stats()
        # Reset success flags for the reset environments
        self.reached_object[env_ids] = False
        self.grasped_object[env_ids] = False
        self.lifted_object[env_ids] = False
        self.transported_object[env_ids] = False
        self.placed_object[env_ids] = False
        
        if self.cfg.randomize_joints:
            self.__reset_joints_by_offset(env_ids, position_range=(-1.0, 1.0))
        else:
            self._reset_articulation(env_ids)

        # Reset the object
        self._reset_object_uniform(env_ids, self.scene.rigid_objects['object'], pose_range=self.cfg.object_pose_range, velocity_range={})
        # Reset the target
        # self._reset_object_uniform(env_ids, self.scene.rigid_objects['target'], pose_range=self.cfg.target_pose_range, velocity_range={})
        # self._reset_target_pose(env_ids)
        self._reset_target_uniform(env_ids, self.cfg.target_pose_range)

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
        self.place_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _log_episode_stats(self, env_ids):
        # Compute success rates for the environments being reset
        reach_rate = self.reached_object[env_ids].float().mean().item()
        grasp_rate = self.grasped_object[env_ids].float().mean().item()
        lift_rate = self.lifted_object[env_ids].float().mean().item()
        transport_rate = self.transported_object[env_ids].float().mean().item()
        place_rate = self.placed_object[env_ids].float().mean().item()
        
        # Compute mean episode reward for the environments being reset
        mean_reward = self.episode_reward[env_ids].mean().item()
        self.best_reach_rate = max(self.best_reach_rate, reach_rate)
        self.best_grasp_rate = max(self.best_grasp_rate, grasp_rate)
        self.best_lift_rate = max(self.best_lift_rate, lift_rate)
        self.best_transport_rate = max(self.best_transport_rate, transport_rate)
        self.best_place_rate = max(self.best_place_rate, place_rate)
        self.best_episode_reward = max(self.best_episode_reward, mean_reward)
        
        # Log current episode metrics
        self.extras["episode/reach_rate"] = reach_rate
        self.extras["episode/grasp_rate"] = grasp_rate
        self.extras["episode/lift_rate"] = lift_rate
        self.extras["episode/transport_rate"] = transport_rate
        self.extras["episode/place_rate"] = place_rate
        self.extras["episode/mean_reward"] = mean_reward
        
        # Log best metrics
        self.extras["best/reach_rate"] = self.best_reach_rate
        self.extras["best/grasp_rate"] = self.best_grasp_rate
        self.extras["best/lift_rate"] = self.best_lift_rate
        self.extras["best/transport_rate"] = self.best_transport_rate
        self.extras["best/place_rate"] = self.best_place_rate
        self.extras["best/episode_reward"] = self.best_episode_reward
        
        # Log number of episodes reset
        self.extras["episode/reset_count"] = len(env_ids)


    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(
            translations=self.target_pos_w,
            orientations=self.target_quat_w,
        )

    def _reset_target_pose(self, env_ids):
        # update goal pose and markers within workspace bounds

        # sample positions
        rand_pos = torch.rand((len(env_ids), 3), device=self.device)
        self.target_pos[env_ids] = rand_pos * (self.target_pos_high - self.target_pos_low) + self.target_pos_low

        # sample orientations with random yaw around z
        yaw = (torch.rand(len(env_ids), device=self.device) - 0.5) * 2 * torch.pi
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)

        self.target_quat[env_ids] = torch_utils.quat_from_euler_xyz(roll, pitch, yaw)

        # convert to world frame
        base_pos_w = self.robot.data.root_pos_w[env_ids]
        base_quat_w = self.robot.data.root_quat_w[env_ids]
        self.target_pos_w[env_ids] = torch_utils.quat_apply(base_quat_w, self.target_pos[env_ids]) + base_pos_w
        self.target_quat_w[env_ids] = torch_utils.quat_mul(base_quat_w, self.target_quat[env_ids])

    def _reset_target_uniform(
        self,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
    ):
        # Sample base positions uniformly within low/high bounds
        rand_pos = torch.rand((len(env_ids), 3), device=self.device)
        base_pos = rand_pos * (self.target_pos_high - self.target_pos_low) + self.target_pos_low
        
        # Apply additional pose range offsets
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        
        # Apply position offsets
        self.target_pos[env_ids] = base_pos + rand_samples[:, 0:3]
        
        # Apply orientation (roll, pitch, yaw)
        self.target_quat[env_ids] = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3],  # roll
            rand_samples[:, 4],  # pitch
            rand_samples[:, 5]   # yaw
        )
        
        # Convert to world frame
        base_pos_w = self.robot.data.root_pos_w[env_ids]
        base_quat_w = self.robot.data.root_quat_w[env_ids]
        self.target_pos_w[env_ids] = torch_utils.quat_apply(base_quat_w, self.target_pos[env_ids]) + base_pos_w
        self.target_quat_w[env_ids] = torch_utils.quat_mul(base_quat_w, self.target_quat[env_ids])
