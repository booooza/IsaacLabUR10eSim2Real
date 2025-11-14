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

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # initialize metrics
        self._reset_episode_stats()
        if "log" not in self.extras:
            self.extras["log"] = dict()

        # DEBUG: Print actual joint properties
        print("\n=== GRIPPER JOINT DEBUG ===")
        print(f"Gripper joint IDs: {self.gripper_joint_ids}")
        print(f"Stiffness: {self.robot.data.joint_stiffness[0, self.gripper_joint_ids]}")
        print(f"Damping: {self.robot.data.joint_damping[0, self.gripper_joint_ids]}")
        print(f"Effort limits: {self.robot.data.joint_effort_limits[0, self.gripper_joint_ids]}")
        print(f"Velocity limits: {self.robot.data.joint_vel_limits[0, self.gripper_joint_ids]}")
        print("===========================\n")

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
        
        # Process arm joint actions
        joint_scale = torch.tensor(self.cfg.action_scale[:6], device=actions.device)
        
        if self.cfg.use_default_offset:
            if self.cfg.action_type == "position":
                joint_offset = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
            elif self.cfg.action_type == "velocity":
                joint_offset = self.robot.data.default_joint_vel[:, self.joint_ids].clone()
            else:
                joint_offset = torch.zeros_like(joint_scale, device=actions.device)
        else:
            joint_offset = torch.zeros_like(joint_scale, device=actions.device)
        
        # Process arm actions
        processed_arm_actions = joint_offset + arm_actions * (joint_scale / 2)
        
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
        # Combine all joint IDs (6 arm + 2 gripper)
        all_joint_ids = self.joint_ids + self.gripper_joint_ids
        if self.cfg.action_type == "effort":
            self.robot.set_joint_effort_target(self.processed_actions, joint_ids=all_joint_ids)
        if self.cfg.action_type == "position":
            self.robot.set_joint_position_target(self.processed_actions, joint_ids=all_joint_ids)
        if self.cfg.action_type == "velocity":
            self.robot.set_joint_velocity_target(self.processed_actions, joint_ids=all_joint_ids)

    def _get_observations(self) -> VecEnvObs:
        """Compute and return the observations for the environment.
        Returns:
            The observations for the environment.
        """        
        # Get raw joint positions (6 arm joints only)
        joint_positions_raw = self.robot.data.joint_pos[:, self.joint_ids]  # (num_envs, 6)
        joint_scale = torch.tensor(self.cfg.action_scale, device=self.device)
        
        if self.cfg.use_default_offset:
            if self.cfg.action_type == "position":
                joint_offset = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
            elif self.cfg.action_type == "velocity":
                joint_offset = self.robot.data.default_joint_vel[:, self.joint_ids].clone()
            else:
                joint_offset = torch.zeros_like(joint_scale, device=self.device)
        else:
            joint_offset = torch.zeros_like(joint_scale, device=self.device)
        
        joint_positions = (joint_positions_raw - joint_offset) / (joint_scale / 2.0)
        
        # Joint velocities
        joint_velocities = self.robot.data.joint_vel[:, self.joint_ids]  # (num_envs, 6)
        joint_vel_limits = self.robot.data.joint_vel_limits[:, self.joint_ids]
        joint_velocities = torch.clamp(joint_velocities / joint_vel_limits, -1.0, 1.0)
        
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
        current_width_normalized = 2.0 * (current_gripper_width / gripper_max_width) - 1.0
        current_width_tensor = current_width_normalized.unsqueeze(-1)  # (num_envs, 1)
        
        obs = torch.cat(
            (
                joint_positions,          # (num_envs, 6) [-1, 1]
                joint_velocities,         # (num_envs, 6) [-1, 1]
                obj_pos,                   # (num_envs, 3)
                obj_quat,                  # (num_envs, 4)
                target_pos,               # (num_envs, 3)
                target_quat,              # (num_envs, 4)
                current_width_tensor,     # (num_envs, 1) [-1, 1]
                target_width_tensor,      # (num_envs, 1) [-1, 1]
                self.previous_actions,    # (num_envs, 7)
            ),
            dim=-1,
        )
        
        return {"policy": obs}

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
        distance_ee_obj_l2 = torch.norm(object_pos - ee_pos, p=2, dim=-1)
        std = 0.1 # start guiding with tanh from 10 cm
        distance_ee_obj_tanh = 1.0 - torch.tanh(distance_ee_obj_l2 / std)

        gripper_positions = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        current_gripper_width = gripper_positions.sum(dim=-1)
        target_width = 0.03
        gripper_max_width = 0.05
        grip_width_l2 = torch.abs(current_gripper_width - target_width)
        std = 0.010  # start guiding with tanh from 10mm
        grip_width_tanh = 1.0 - torch.tanh(grip_width_l2 / std)
        # Only reward gripper closing when close to object
        close_to_object = distance_ee_obj_l2 < target_width / 2  # Within 1.5cm (assuming cube)
        grip_width_reward = torch.where(
            close_to_object,
            grip_width_tanh,
            torch.zeros_like(grip_width_tanh)
        )

        minimal_height = 0.04 # 4cm above table = lifted
        object_height = object_pos[:, 2]
        table_height = 0.015  # From your init state
        lift_height = object_height - table_height
        is_lifted = torch.where(lift_height > minimal_height, 1.0, 0.0)
        
        envs_lifted = lift_height > minimal_height
        if envs_lifted.any():
            print(f"🛗 Lifting in {envs_lifted.sum().item()} envs.")

        # Linear L2 distance target to obj
        distance_obj_target_l2 = torch.norm(target_pos - object_pos, dim=1)
        # rewarded if the object is lifted above the threshold
        distance_obj_target_tanh = is_lifted * (1 - torch.tanh(distance_obj_target_l2 / 0.3))
        distance_obj_target_tanh_fine = is_lifted * (1 - torch.tanh(distance_obj_target_l2 / 0.05))
        
        # --- Penalties ---
        action_l2 = torch.sum(torch.square(self.actions), dim=1)
        action_diff = self.actions - self.previous_actions
        action_rate_l2 = torch.sum(torch.square(action_diff), dim=1)
        action_rate_limit = self._action_rate_limit(all_joint_ids, action_diff, threshold_ratio=0.1)
        joint_vel_l2 = torch.sum(torch.square(self.robot.data.joint_vel[:, self.joint_ids]), dim=1)
        joint_pos_limit = self._joint_pos_limits(all_joint_ids)
        joint_vel_limit = self._joint_vel_limits(all_joint_ids, soft_ratio=1.0)

        # --- Sparse ---        
        grasp_success = (is_lifted > 0.5) & (distance_obj_target_l2 < 0.1)
        # --- Total weighted reward ---
        reward = (
            # Task rewards
            1.0 * distance_ee_obj_tanh +
            1.0 * grip_width_reward + 
            16.0 * distance_obj_target_tanh + 
            5.0 * distance_obj_target_tanh_fine +
            15 * is_lifted +
            # Regularization penalties
            self.cfg.action_l2_w * action_l2 +
            self.cfg.action_rate_l2_w * action_rate_l2 +
            # Safety limits
            self.cfg.joint_pos_limit_w * joint_pos_limit +
            self.cfg.joint_vel_limit_w * joint_vel_limit
            # Success bonus
        )

        # --- Logging ---
        self.episode_reward += reward
        self.episode_success = self.episode_success | grasp_success
        self.distance_l2 = distance_ee_obj_l2
        self.target_l2 = distance_obj_target_l2
        self.grip_width_l2 = grip_width_l2
        self.lift_height = self.lift_height

        self.extras["log"]["success_rate"] = grasp_success.float().mean().item()

        if grasp_success.any():
            print(f"Grasp success for {grasp_success.sum().item()} envs.")

        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        print("reset")
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self._log_episode_stats(env_ids)
        self._reset_target_pose(env_ids)
        self._reset_articulation(env_ids)
        # self._reset_rigid_objects(env_ids)
        # add "yaw": (-3.14, 3.14),
        self._reset_object_uniform(env_ids, self.scene.rigid_objects['object'], pose_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, velocity_range={})
        self._reset_episode_stats()

    def _log_episode_stats(self, env_ids):
        super()._log_episode_stats(env_ids)
        self.extras["log"]["target_l2"] = self.target_l2.mean().item()
        self.extras["log"]["grip_width_l2"] = self.grip_width_l2.mean().item()
        self.extras["log"]["lift_height"] = self.lift_height.mean().item()

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
