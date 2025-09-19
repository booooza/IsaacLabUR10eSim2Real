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
    # env
    decimation = 2  # 60Hz control frequency like thesis
    episode_length_s = 5.0 # Should give ~600 timesteps at 120Hz like thesis
    # Number of environments to run in parallel
    num_envs = 512
    # Spaces definition
    action_space = 6  # 6 joints of UR10e
    observation_space = 29  # Based on bachelor's thesis: joint pos (6) + joint vel (6) + goal pos (3) + goal rot (4) + relative rot (4) + actions (6)
    state_space = 0
    
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    
    # robot configuration
    robot_cfg: ArticulationCfg = UR10e_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs, 
        env_spacing=2.0, 
        replicate_physics=True
    )
    
    # Joint names for UR10e
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint", 
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]
    
    # Goal object configuration
    goal_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.5, 0.5], rot=[1.0, 0.0, 0.0, 0.0]),
    )
    
    # Reward scales (based on bachelor's thesis)
    distance_reward_scale = -2.0 #1.0
    rotation_reward_scale = 1.0 # 0.5
    action_penalty_scale = -0.0002 # 0.01
    reach_bonus = 250.0
    
    # Reach threshold
    reach_distance_threshold = 0.1  # meters (10cm)
    reach_rotation_threshold = 0.1   # radians
    
    # Action smoothing
    action_scale = 1.0
    
    # Goal reset ranges (based on thesis workspace)
    goal_position_range = {
        "x": (0.1, 0.9),  # Forward reach
        "y": (-0.4, 0.4), # Left-right
        "z": (0.1, 0.8),  # Up-down  
    }


@configclass  
class UR10eReacherEnvCfgPlay(UR10eReacherEnvCfg):
    """Configuration for playing with the UR10e reacher environment."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play/visualization
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        
        # longer episodes for manual testing
        self.episode_length_s = 1.0
        
        # modify action scaling for more responsive control
        self.action_scale = 0.2
