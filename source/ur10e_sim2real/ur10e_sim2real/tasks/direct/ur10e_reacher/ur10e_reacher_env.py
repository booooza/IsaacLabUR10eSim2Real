"""UR10e reacher environment implementation."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import subtract_frame_transforms, quat_mul, quat_conjugate, matrix_from_quat
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from .ur10e_reacher_env_cfg import UR10eReacherEnvCfg


def define_markers() -> VisualizationMarkers:
    """Define goal visualization markers."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/FrameTransformer",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class UR10eReacherEnv(DirectRLEnv):
    """UR10e reacher environment using direct workflow."""
    
    cfg: UR10eReacherEnvCfg

    def __init__(self, cfg: UR10eReacherEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices
        self.joint_ids, joint_names = self.robot.find_joints(self.cfg.joint_names)
        self.num_arm_dofs = len(self.joint_ids)
        
        # End-effector tracking - UR10e uses wrist_3_link as end effector
        self.ee_frame_name = "wrist_3_link"
        self.ee_body_idx, _ = self.robot.find_bodies(self.ee_frame_name)
        
        # Goal and target tracking
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_rot[:, 0] = 1.0  # Initialize with identity quaternion
        
        # Action history for observation
        self.previous_actions = torch.zeros((self.num_envs, 6), device=self.device)
        
        # Reach tracking
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.consecutive_reach_count = torch.zeros(self.num_envs, device=self.device)
        
        # Metrics
        self.success_rate = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        """Set up the scene with robot, goal object, and environment."""
        # Add robot
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Add goal object  
        self.goal_object = RigidObject(self.cfg.goal_object_cfg)
        
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Add articulation and goal to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["goal_object"] = self.goal_object
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Setup visualization markers
        self.goal_markers = define_markers()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        # Store actions for observation
        self.previous_actions = actions.clone()
        
        # Scale actions
        self.processed_actions = actions * self.cfg.action_scale

    def _apply_action(self) -> None:
        """Apply processed actions to the robot."""
        # Apply joint position targets
        joint_pos_target = self.robot.data.joint_pos[:, self.joint_ids] + self.processed_actions
        self.robot.set_joint_position_target(joint_pos_target, joint_ids=self.joint_ids)

    def _get_observations(self) -> dict:
        """Compute environment observations based on the bachelor's thesis state space."""
        # Joint positions (normalized to [-1, 1])
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_pos_norm = 2.0 * (joint_pos - self.robot.data.soft_joint_pos_limits[:, self.joint_ids, 0]) / (
            self.robot.data.soft_joint_pos_limits[:, self.joint_ids, 1] 
            - self.robot.data.soft_joint_pos_limits[:, self.joint_ids, 0]
        ) - 1.0
        
        # Joint velocities (scaled)
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids] * 0.1
        
        # Goal position and rotation
        goal_pos = self.goal_pos
        goal_rot = self.goal_rot
        
        # End-effector pose
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx, :].squeeze(1)
        ee_rot_w = self.robot.data.body_quat_w[:, self.ee_body_idx, :].squeeze(1)
        
        # Relative orientation between end-effector and goal
        relative_rot = quat_mul(ee_rot_w, quat_conjugate(goal_rot))
        
        # Previous actions
        prev_actions = self.previous_actions
        
        # Concatenate all observations (29 dims total as per thesis)
        obs = torch.cat([
            joint_pos_norm,          # 6 dims
            joint_vel,               # 6 dims  
            goal_pos,                # 3 dims
            goal_rot,                # 4 dims
            relative_rot,            # 4 dims
            prev_actions,            # 6 dims
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Bachelor thesis reward implementation from equations 3.16-3.20."""
        
        # Get end-effector position and rotation
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx, :].squeeze(1)
        ee_rot_w = self.robot.data.body_quat_w[:, self.ee_body_idx, :].squeeze(1)
        
        # Equation 3.16: dgoal = ||obj_pos - target_pos||₂
        dgoal = torch.norm(ee_pos_w - self.goal_pos, dim=-1)
        
        # Equation 3.17: rot_dist = 2 * arcsin(min(||quat_diff[:, 1:4]||₂, 1.0))
        quat_diff = quat_mul(ee_rot_w, quat_conjugate(self.goal_rot))
        rot_dist = 2.0 * torch.arcsin(torch.clamp(torch.norm(quat_diff[:, 1:4], dim=-1), max=1.0))
        
        # Equation 3.18: ract = (∑ a²t) × action_penalty_scale
        ract = torch.sum(self.processed_actions**2, dim=-1) * self.cfg.action_penalty_scale
        
        # Equation 3.19: rrot = 1/(|rot_dist| + rot_eps) × rot_reward_scale  
        rrot = 1.0 / (torch.abs(rot_dist) + 0.1) * self.cfg.rotation_reward_scale
        
        # Equation 3.20: rt = (dgoal × dist_reward_scale) + rrot - ract + reach_goal_bonus
        rt = (dgoal * self.cfg.distance_reward_scale) + rrot - ract
        
        # Add reach bonus when close enough (thesis used 0.1 tolerance)
        reach_bonus = torch.where(
            dgoal < 0.1,  # SUCCESS_TOLERANCE from thesis
            self.cfg.reach_bonus,
            0.0
        )

        total_reward = rt + reach_bonus

        print(f"Total reward: {total_reward.mean():.4f}")
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check for episode termination conditions."""
        # Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Early termination for successful reach (optional)
        task_completed = self.consecutive_reach_count >= 10  # Hold for 10 steps
        
        # No catastrophic failures defined for this task
        died = torch.zeros_like(time_out)
        
        return died, time_out | task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        # Reset robot to initial state
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        
        # Reset joint states to default (only for the specific joints we control)
        joint_pos = self.robot.data.default_joint_pos[env_ids][:, self.joint_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids][:, self.joint_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self.joint_ids, env_ids=env_ids)
        
        # Reset goal positions randomly within workspace
        self._reset_goal_positions(env_ids)
        
        # Reset tracking variables
        self.goal_reached[env_ids] = False
        self.consecutive_reach_count[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        
        # Update visualization
        self._update_goal_visualization()

    def _reset_goal_positions(self, env_ids: Sequence[int]):
        """Reset goal positions randomly within the workspace."""
        num_resets = len(env_ids)
        
        # Get robot base positions for these environments
        robot_base_pos = self.robot.data.root_pos_w[env_ids]
        
        # Sample random positions within defined ranges
        new_x = torch.rand(num_resets, device=self.device) * 0.6 + 0.3  # 0.3 to 0.9m
        new_y = torch.rand(num_resets, device=self.device) * 0.6 - 0.3  # -0.3 to 0.3m  
        new_z = torch.rand(num_resets, device=self.device) * 0.4 + 0.2  # 0.2 to 0.6m
        
        # Store goals in WORLD coordinates (relative to robot base)
        self.goal_pos[env_ids, 0] = robot_base_pos[:, 0] + new_x  
        self.goal_pos[env_ids, 1] = robot_base_pos[:, 1] + new_y  
        self.goal_pos[env_ids, 2] = robot_base_pos[:, 2] + new_z
        
        # Set goal orientations
        self.goal_rot[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        
        # Update goal object positions in simulation
        goal_positions = self.goal_pos[env_ids]  # Already in world coordinates
        goal_orientations = self.goal_rot[env_ids]
        
        # Write to goal object
        root_state = torch.zeros((len(env_ids), 13), device=self.device)
        root_state[:, :3] = goal_positions
        root_state[:, 3:7] = goal_orientations
        self.goal_object.write_root_state_to_sim(root_state, env_ids)

    def _update_goal_visualization(self):
        """Update goal visualization markers."""
        # Update marker positions at goal locations
        goal_positions = self.goal_pos + self.scene.env_origins
        goal_orientations = self.goal_rot
        
        # Visualize frames at goal positions
        self.goal_markers.visualize(goal_positions, goal_orientations)

    def get_metrics(self) -> dict:
        """Return environment metrics."""
        return {
            "success_rate": torch.mean(self.goal_reached.float()).item(),
            "avg_consecutive_reach": torch.mean(self.consecutive_reach_count).item(),
            "avg_goal_distance": torch.mean(torch.norm(
                self.robot.data.body_pos_w[:, self.ee_body_idx, :].squeeze(1) - self.goal_pos, dim=-1
            )).item(),
        }