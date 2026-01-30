import torch
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from isaaclab.scene import InteractiveScene
from isaaclab.envs import DirectRLEnv, VecEnvObs

class PlayLogger:
    """Buffer logging data and write once at episode end. """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.trajectory_data = []
        self.header = None
        self.has_object = None
        
    def log_step(self, env: DirectRLEnv, dt: float, episode_ids: torch.Tensor):
        """Accumulate data for one step (all envs)."""

        scene = env.scene
        step_count = env.common_step_counter
        episode_length_buf = env.episode_length_buf
        
        robot = scene.articulations['robot']
        joint_names = robot._data.joint_names
        num_envs = robot._data.joint_pos.shape[0]

        if self.has_object is None:
            # Use hasattr or try-except to check
            try:
                _ = scene['object_frame']
                self.has_object = True
            except (KeyError, AttributeError):
                self.has_object = False
            print(f"[INFO] Object frame {'found' if self.has_object else 'not found'} in scene")

        # Get all data (same as your original)
        joint_pos = robot._data.joint_pos
        joint_pos_target = robot._data.joint_pos_target
        joint_pos_limits = robot._data.joint_pos_limits
        soft_joint_pos_limits = robot._data.soft_joint_pos_limits
        
        joint_vel = robot._data.joint_vel
        joint_vel_target = robot._data.joint_vel_target
        joint_vel_limits = robot._data.joint_vel_limits
        soft_joint_vel_limits = robot._data.soft_joint_vel_limits
        
        ee_frame_pos = scene['ee_frame'].data.target_pos_source
        ee_frame_rot = scene['ee_frame'].data.target_quat_source

        if self.has_object:
            object_pos = scene['object_frame'].data.target_pos_source
            object_rot = scene['object_frame'].data.target_quat_source
            if hasattr(env, 'obj_width_per_env'):
                object_width = env.obj_width_per_env  # (num_envs,)
            else:
                object_width = torch.full((num_envs,), 0.03, device=env.device)

        target_pos = env.target_pos
        target_rot = env.target_quat
        
        sim_time = step_count * dt
        
        # Write data for each environment
        for env_id in range(num_envs):
            row = [step_count, sim_time, env_id, episode_ids[env_id].item(), episode_length_buf[env_id].item()]
            
            # Add joint positions
            row.extend(joint_pos[env_id].cpu().numpy().tolist())
            # Add joint position targets
            row.extend(joint_pos_target[env_id].cpu().numpy().tolist())
            # Add joint velocities
            row.extend(joint_vel[env_id].cpu().numpy().tolist())
            # Add joint velocity targets
            row.extend(joint_vel_target[env_id].cpu().numpy().tolist())
            # Add joint position limits (lower, upper for each joint)
            for joint_idx in range(len(joint_names)):
                row.append(joint_pos_limits[env_id, joint_idx, 0].item())
                row.append(joint_pos_limits[env_id, joint_idx, 1].item())
            # Add soft joint position limits (lower, upper for each joint)
            for joint_idx in range(len(joint_names)):
                row.append(soft_joint_pos_limits[env_id, joint_idx, 0].item())
                row.append(soft_joint_pos_limits[env_id, joint_idx, 1].item())
            # Add joint velocity limits
            row.extend(joint_vel_limits[env_id].cpu().numpy().tolist())
            # Add soft joint velocity limits
            row.extend(soft_joint_vel_limits[env_id].cpu().numpy().tolist())
            # Add end effector frame positions
            row.extend([
                ee_frame_pos[env_id, 0, 0].item(),
                ee_frame_pos[env_id, 0, 1].item(),
                ee_frame_pos[env_id, 0, 2].item()
            ])
            # Add end effector frame rotations
            row.extend([
                ee_frame_rot[env_id, 0, 0].item(),
                ee_frame_rot[env_id, 0, 1].item(),
                ee_frame_rot[env_id, 0, 2].item(),
                ee_frame_rot[env_id, 0, 3].item()
            ])
            # Add object positions (or NA if not present)
            if self.has_object:
                row.extend([
                    object_pos[env_id, 0, 0].item(),
                    object_pos[env_id, 0, 1].item(),
                    object_pos[env_id, 0, 2].item()
                ])
                # Add object rotations
                row.extend([
                    object_rot[env_id, 0, 0].item(),
                    object_rot[env_id, 0, 1].item(),
                    object_rot[env_id, 0, 2].item(),
                    object_rot[env_id, 0, 3].item()
                ])
                row.append(object_width[env_id].item())
            else:
                # Add NA values for object pos and rot
                row.extend(['NA'] * 8)  # 3 for pos + 4 for rot
            
            # Add target positions
            row.extend([
                target_pos[env_id, 0].item(),
                target_pos[env_id, 1].item(),
                target_pos[env_id, 2].item()
            ])
            # Add target rotations
            row.extend([
                target_rot[env_id, 0].item(),
                target_rot[env_id, 1].item(),
                target_rot[env_id, 2].item(),
                target_rot[env_id, 3].item()
            ])

            self.trajectory_data.append(row)
        
        # Build header on first call
        if self.header is None:
            self.header = ['step', 'sim_time', 'env_id', 'episode_id', 'episode_step']
            # Add joint position columns
            for name in joint_names:
                self.header.append(f'{name}_pos')
            # Add joint position target columns
            for name in joint_names:
                self.header.append(f'{name}_pos_target')
            # Add joint velocity columns
            for name in joint_names:
                self.header.append(f'{name}_vel')
            # Add joint velocity target columns
            for name in joint_names:
                self.header.append(f'{name}_vel_target')
            # Add joint position limits (lower and upper)
            for name in joint_names:
                self.header.append(f'{name}_pos_limit_lower')
                self.header.append(f'{name}_pos_limit_upper')
            # Add soft joint position limits (lower and upper)
            for name in joint_names:
                self.header.append(f'{name}_soft_pos_limit_lower')
                self.header.append(f'{name}_soft_pos_limit_upper')
            # Add joint velocity limits
            for name in joint_names:
                self.header.append(f'{name}_vel_limit')
            # Add soft joint velocity limits
            for name in joint_names:
                self.header.append(f'{name}_soft_vel_limit')
            # Add end effector position columns
            self.header.extend(['ee_frame_pos_x', 'ee_frame_pos_y', 'ee_frame_pos_z'])
            # Add end effector rotation columns
            self.header.extend(['ee_frame_rot_w', 'ee_frame_rot_x', 'ee_frame_rot_y', 'ee_frame_rot_z'])
            # Add object position columns
            self.header.extend(['object_pos_x', 'object_pos_y', 'object_pos_z'])
            # Add object rotation columns
            self.header.extend(['object_rot_w', 'object_rot_x', 'object_rot_y', 'object_rot_z'])
            # Add object width colum
            self.header.extend(['object_width'])
            # Add target position columns
            self.header.extend(['target_pos_x', 'target_pos_y', 'target_pos_z'])
            # Add target rotation columns
            self.header.extend(['target_rot_w', 'target_rot_x', 'target_rot_y', 'target_rot_z'])
    
    def save(self, filename: str = "play_log.csv", overwrite: bool = True):
        """Write all accumulated data to CSV."""
        csv_file = self.log_dir / filename
        
        if csv_file.exists() and not overwrite:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self.log_dir / f"play_log_{timestamp}.csv"
            print(f"[INFO] Existing log found, saving to: {csv_file.name}")
        
        print(f"[INFO] Writing {len(self.trajectory_data)} rows to {csv_file}")
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(self.trajectory_data)
        
        print(f"[INFO] Log saved successfully: {csv_file}")
