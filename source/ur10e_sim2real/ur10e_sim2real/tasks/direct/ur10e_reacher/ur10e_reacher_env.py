"""UR10e reacher environment implementation"""

from __future__ import annotations

from datetime import datetime
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
import wandb

if TYPE_CHECKING:
    from .ur10e_reacher_env_cfg import UR10eReacherEnvCfg


def define_goal_markers(tolerance: float = 0.05) -> VisualizationMarkers:
    """Define goal visualization markers with radius matching success tolerance."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GoalMarkers",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=tolerance,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), 
                ),
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
        print(f"Found {self.num_arm_dofs} joints: {joint_names}")
        
        # End-effector tracking - UR10e uses wrist_3_link as end effector
        self.ee_frame_name = "wrist_3_link"
        self.ee_body_idx, ee_body_names = self.robot.find_bodies(self.ee_frame_name)
        print(f"End-effector body index: {self.ee_body_idx}, names: {ee_body_names}")
        
        # Debug: Print all available body names to verify correct end-effector
        all_body_names = self.robot.body_names
        print(f"All available robot bodies: {all_body_names}")

        # Goal and target tracking
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_rot[:, 0] = 1.0  # Initialize with identity quaternion
        
        # Action history for observation
        self.previous_actions = torch.zeros((self.num_envs, 6), device=self.device)
        
        # Reach tracking
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.consecutive_reach_count = torch.zeros(self.num_envs, device=self.device)

        # Store cumulative rewards
        self.cumulative_reward = 0.0
        
        # Store joint limits for normalization
        joint_limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids, :]
        self.joint_lower_limits = joint_limits[0, :, 0]  # Lower limits
        self.joint_upper_limits = joint_limits[0, :, 1]   # Upper limits
        print(f"Joint limits: {self.joint_lower_limits} to {self.joint_upper_limits}")

        if cfg.wandb_run:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )
            wandb.config.update({"env_cfg": cfg.to_dict()})

    def _setup_scene(self):
        """Set up the scene with robot and environment."""
        print("Setting up UR10e Reacher environment scene...")

        # Add robot
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # Setup visualization markers
        self.goal_markers = define_goal_markers(self.cfg.success_tolerance,)

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
        
        # 1. Joint positions (6 dims) - normalized to [-1, 1]
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_pos_norm = 2.0 * (joint_pos - self.joint_lower_limits) / (
            self.joint_upper_limits - self.joint_lower_limits
        ) - 1.0
        
        # Joint velocities (scaled)
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids] * 0.1
        
        # 3. Goal position (3 dims) - world coordinates
        goal_pos = self.goal_pos
        
        # 4. Goal orientation (4 dims) - quaternion
        goal_rot = self.goal_rot
        
        # 5. End-effector pose
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx, :].squeeze(1)
        ee_rot_w = self.robot.data.body_quat_w[:, self.ee_body_idx, :].squeeze(1)
        
        # 6. Relative orientation (4 dims) - end-effector to goal
        relative_rot = quat_mul(ee_rot_w, quat_conjugate(goal_rot))
        
        # 7. Previous actions (6 dims)
        prev_actions = self.previous_actions
        
        # Concatenate observations (29 dims total)
        obs = torch.cat([
            joint_pos_norm,     # 6 dims: θ₁...θ₆ normalized
            joint_vel,          # 6 dims: θ'₁...θ'₆ scaled  
            goal_pos,           # 3 dims: x_goal, y_goal, z_goal
            goal_rot,           # 4 dims: goal orientation quaternion
            relative_rot,       # 4 dims: relative orientation end-effector to goal
            prev_actions,       # 6 dims: a₁...a₆ previous actions
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Simple distance-based reward for reaching."""
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx, :].squeeze(1)
        dgoal = torch.norm(ee_pos_w - self.goal_pos, dim=-1)
        
        reward = -dgoal
        
        # Add success bonus
        reward += torch.where(dgoal < 0.05, 10.0, 0.0)  # Bonus for reaching
        
        # Add action penalty for smoothness
        action_penalty = 0.01 * torch.sum(self.processed_actions**2, dim=-1)
        #reward -= action_penalty

        # Compute success mask
        success_mask = dgoal < self.cfg.success_tolerance
        # If successful, print debug info
        if success_mask.any():
            print(f"SUCCESS!!! Env(s) {success_mask.nonzero(as_tuple=True)[0].cpu().numpy()} reached the goal!")

        # Update cumulative rewards
        self.cumulative_reward += reward.mean().item()

        # Log to wandb if run is active
        if wandb.run is not None:
            wandb.log({
                "step": self.common_step_counter,
                "reward/total_mean": reward.mean().item(),
                "reward/avg_dgoal": dgoal.mean().item(),
                "reward/avg_action_penalty": action_penalty,
                "reward/cumulative": self.cumulative_reward,
                "reward/running_avg": self.cumulative_reward / (self.common_step_counter + 1),
                "metrics/success_count": int(success_mask.sum().item()),
                "metrics/success_rate": float(success_mask.float().mean().item()),
            })

        return reward
    
    def _get_rewards_thesis(self) -> torch.Tensor:
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
        rrot = 1.0 / (torch.abs(rot_dist) + self.cfg.rot_eps) * self.cfg.rotation_reward_scale
        
        # Equation 3.20: rt = (dgoal × dist_reward_scale) + rrot - ract + reach_goal_bonus
        distance_reward = dgoal * self.cfg.distance_reward_scale  # Note: scale is negative (-2.0)
        rt = distance_reward + rrot - ract
        
        # Add reach bonus when within success tolerance (thesis used 0.1 tolerance)
        reach_bonus = torch.where(
            dgoal < self.cfg.success_tolerance,                # 250.0 as in thesis
            0.0
        )

        total_reward = rt + reach_bonus

        if self.common_step_counter % 100 == 0:
            print(f"Step {self.common_step_counter}: reward={total_reward[0].cpu().numpy():.2f}, dgoal={dgoal[0].cpu().numpy():.3f}")

        # Compute success mask
        success_mask = dgoal < self.cfg.success_tolerance

        # Update cumulative rewards
        self.cumulative_reward += total_reward.mean().item()

        # Log to wandb if run is active
        if wandb.run is not None:
            wandb.log({
                "reward/total_mean": total_reward.mean().item(),
                "reward/distance_mean": distance_reward.mean().item(),
                "reward/rotation_mean": rrot.mean().item(),
                "reward/action_penalty_mean": ract.mean().item(),
                "reward/reach_bonus_mean": reach_bonus.mean().item(),
                "reward/cumulative": self.cumulative_reward,
                "reward/running_avg": self.cumulative_reward / (self.common_step_counter + 1),
                "metrics/avg_dgoal": dgoal.mean().item(),
                "metrics/success_count": int(success_mask.sum().item()),
                "metrics/success_rate": float(success_mask.float().mean().item()),
            })

        # Update visualization every step
        self._update_goal_visualization()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check for episode termination conditions."""
        # Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Early termination for successful reach
        task_completed = torch.zeros_like(time_out)
        
        # No catastrophic failures defined for this task
        died = torch.zeros_like(time_out)
        
        dones = died, time_out | task_completed
        
        if dones[1].any():
            print(f"Resetting {dones[1].sum().item()} environments (timeout or task completed).")

        return dones

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
        print(f"Robot base position (world): {robot_base_pos[0].cpu().numpy()}")
        
        # Sample random positions within defined ranges:
        # x: 0.3 to 0.9m, y: -0.3 to 0.3m, z: 0.2 to 0.6m
        new_x = torch.rand(num_resets, device=self.device) * 0.6 + 0.3  # 0.3 to 0.9m
        new_y = torch.rand(num_resets, device=self.device) * 0.6 - 0.3  # -0.3 to 0.3m  
        new_z = torch.rand(num_resets, device=self.device) * 0.4 + 0.2  # 0.2 to 0.6m
        
        print(f"Sampled goal offset: x={new_x[0].cpu().numpy():.3f}, y={new_y[0].cpu().numpy():.3f}, z={new_z[0].cpu().numpy():.3f}")
        
        # Store goals in world coordinates (relative to robot base)
        self.goal_pos[env_ids, 0] = robot_base_pos[:, 0] + new_x  
        self.goal_pos[env_ids, 1] = robot_base_pos[:, 1] + new_y  
        self.goal_pos[env_ids, 2] = robot_base_pos[:, 2] + new_z
        
        print(f"New goal position (world): {self.goal_pos[env_ids][0].cpu().numpy()}")
        print(f"Expected distance from robot base: {torch.norm(self.goal_pos[env_ids][0] - robot_base_pos[0]).cpu().numpy():.3f}m")
        
        # Set goal orientations (identity quaternion)
        self.goal_rot[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
    

    def _update_goal_visualization(self):
        """Update goal visualization markers."""
        # Update marker positions at goal locations
        goal_positions = self.goal_pos # + self.scene.env_origins
        goal_orientations = self.goal_rot
        
        # Visualize frames at goal positions
        self.goal_markers.visualize(goal_positions, goal_orientations)

    def get_metrics(self) -> dict:
        """Return environment metrics."""
        ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx, :].squeeze(1)
        distances = torch.norm(ee_pos_w - self.goal_pos, dim=-1)
        
        return {
            "success_rate": torch.mean(self.goal_reached.float()).item(),
            "avg_consecutive_reach": torch.mean(self.consecutive_reach_count).item(),
            "avg_goal_distance": torch.mean(distances).item(),
            "min_goal_distance": torch.min(distances).item(),
            "max_goal_distance": torch.max(distances).item(),
        }