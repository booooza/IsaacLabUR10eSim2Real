"""Configuration for the UR10e reacher environment."""

from ur10e_sim2real.robots.ur10e import UR10e_CFG
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils


@configclass
class UR10eReacherEnvCfg(DirectRLEnvCfg):
    wandb_run = True
    wandb_project = "ur10e-sim2real"
    wandb_entity = "booooza-"

    decimation = 2  # 60Hz control frequency (120Hz / 2)
    episode_length_s = 10.0  # 600 timesteps at 60Hz (thesis requirement)
    
    # Number of environments (thesis used 512, but start smaller for debugging)
    num_envs = 256
    
    # Spaces definition
    action_space = 6  # 6 joints of UR10e
    observation_space = 29  # Based on bachelor's thesis: joint pos (6) + joint vel (6) + goal pos (3) + goal rot (4) + relative rot (4) + actions (6)
    state_space = 0
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 120Hz physics
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            enable_ccd=True,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
        ),
    )
    
    # Robot configuration
    robot_cfg: ArticulationCfg = UR10e_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UR10e_CFG.spawn.replace(
            # Enhanced collision properties for better debugging
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            # Enhanced rigid body properties
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
        ),
    )
    
    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=2.0,
        replicate_physics=True,
    )
    
    # Joint names for UR10e (exact thesis setup)
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint", 
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    
    # Reward scales (based on bachelor's thesis)
    distance_reward_scale = -2.0
    rotation_reward_scale = 1.0
    action_penalty_scale = -0.0002
    reach_bonus = 250.0
    rot_eps = 0.1
    
    success_tolerance = 0.1  # 0.1m tolerance (10cm)
    
    action_scale = 20 * (1/120)  # dt from sim = 1/120Hz → ≈0.167
    
    # Observation scaling (thesis specific)
    vel_obs_scale = 0.2  # Exact thesis velObsScale value
    
    # Goal position ranges (exact thesis workspace)
    goal_position_range = {
        "x": (0.3, 0.9),   # Forward reach (thesis values)
        "y": (-0.3, 0.3),  # Left-right (thesis values)
        "z": (0.2, 0.6),   # Up-down (thesis values)
    }
    
    # Debug configuration
    debug_print_interval = 60  # Print debug info every 60 steps (1 second at 60Hz)
    enable_debug_visualization = True
    
    def __post_init__(self):
        """Post initialization to validate configuration."""
        super().__post_init__()
        
        # Validate thesis parameters
        assert self.distance_reward_scale < 0, "Distance reward scale must be negative (thesis)"
        assert self.action_penalty_scale < 0, "Action penalty scale must be negative (thesis)"
        assert self.reach_bonus > 0, "Reach bonus must be positive (thesis)"
        assert self.success_tolerance > 0, "Success tolerance must be positive"
        assert self.vel_obs_scale > 0, "Velocity observation scale must be positive"
        
        # Validate workspace
        x_range = self.goal_position_range["x"]
        y_range = self.goal_position_range["y"] 
        z_range = self.goal_position_range["z"]
        
        assert x_range[1] > x_range[0], "Invalid x range"
        assert y_range[1] > y_range[0], "Invalid y range"
        assert z_range[1] > z_range[0], "Invalid z range"
        assert x_range[0] > 0, "X range must be positive (forward reach)"
        assert z_range[0] > 0, "Z range must be positive (above ground)"
        
        # Calculate expected observation space size
        expected_obs_size = (
            len(self.joint_names) +    # Joint positions (6)
            len(self.joint_names) +    # Joint velocities (6)  
            3 +                        # Goal position (3)
            4 +                        # Goal orientation (4)
            4 +                        # Relative orientation (4)
            len(self.joint_names)      # Previous actions (6)
        )
        assert self.observation_space == expected_obs_size, f"Observation space mismatch: expected {expected_obs_size}, got {self.observation_space}"
        
        print(f"UR10eReacherEnvCfg validation passed!")
        print(f"  Environments: {self.num_envs}")
        print(f"  Episode length: {self.episode_length_s}s ({self.episode_length_s * 60:.0f} steps)")
        print(f"  Action space: {self.action_space}")
        print(f"  Observation space: {self.observation_space}")
        print(f"  Workspace: x{x_range}, y{y_range}, z{z_range}")


@configclass
class UR10eReacherEnvCfgPlay(UR10eReacherEnvCfg):
    """Configuration for playing with the UR10e reacher environment."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0  # More space for visualization
        
        # Longer episodes for manual testing
        self.episode_length_s = 30.0
        
        # Slower actions for manual control and observation
        self.action_scale = 0.05
        
        # More frequent debug prints
        self.debug_print_interval = 30  # Every 0.5 seconds
        
        # Enhanced visualization
        self.enable_debug_visualization = True
        
        print(f"UR10eReacherEnvCfgPlay configured for debugging:")
        print(f"  Single environment with enhanced visualization")
        print(f"  30s episodes, slower actions, frequent debug prints")

