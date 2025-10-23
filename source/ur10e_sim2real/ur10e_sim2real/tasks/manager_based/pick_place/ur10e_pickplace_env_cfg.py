"""
Manager-based Pick-and-Place Environment for UR10e with Robotiq Hand-E Gripper.

This environment implements a curriculum learning approach with stages:
1. Reach: Basic end-effector positioning
2. Grasp: Adding gripper control (future)
3. Full Pick-and-Place: Complete manipulation (future)

Currently implements Stage 1 (Reach) with asymmetric actor-critic observations.
"""

from dataclasses import MISSING
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config.actions import ReachStageActionsCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config.events import ReachStageEventCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config.rewards import ReachStageRewardsCfg
import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.devices import DevicesCfg, Se3KeyboardCfg, Se3GamepadCfg, Se3SpaceMouseCfg

# Import UR10e configuration with Robotiq gripper
from isaaclab_assets import UR10e_ROBOTIQ_GRIPPER_CFG

# Import MDP functions
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config import PickPlaceSceneCfg, PickPlaceActionsCfg, PickPlaceObservationsCfg, PickPlaceEventCfg, PickPlaceRewardsCfg, PickPlaceTerminationsCfg, PickPlaceCurriculumCfg, ReachStageObservationsCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

##
# Environment Configuration
##

@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for pick-and-place environment."""
    
    # Scene
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=False) # check if replicate_physics can be True
    
    # MDP components
    actions: ReachStageActionsCfg = ReachStageActionsCfg()
    observations: ReachStageObservationsCfg = ReachStageObservationsCfg()
    events: ReachStageEventCfg = ReachStageEventCfg()
    rewards: ReachStageRewardsCfg = ReachStageRewardsCfg()
    terminations: PickPlaceTerminationsCfg = PickPlaceTerminationsCfg()
    curriculum: PickPlaceCurriculumCfg = PickPlaceCurriculumCfg()
    
    # No commands needed (using events for goal setting)
    commands = None

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.seed = 42
        self.decimation = 2  # 50 Hz control
        self.episode_length_s = 10.0  # 10 second episodes
        
        # Simulation settings
        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation
        
        # PhysX settings - for stable physics
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
        self.viewer.lookat = (0.5, 0.0, 0.5)

        # Teleoperation devices for debugging
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.01,
                    rot_sensitivity=0.01,
                    gripper_term=True, 
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.01,
                    rot_sensitivity=0.01,
                    gripper_term=True,
                    sim_device=self.sim.device,
                ),
            },
        )

@configclass
class PickPlaceEnvPlayCfg(PickPlaceEnvCfg):
    """Configuration for pick-and-place environment during play/testing."""
    
    def __post_init__(self):
        # Run parent post-init
        super().__post_init__()

        # Viewer settings
        self.viewer.eye = (1.5, -1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        
        # Longer episodes for testing
        self.episode_length_s = 1000.0
        
        # Use only 1 environment for teleop
        self.scene.num_envs = 1
