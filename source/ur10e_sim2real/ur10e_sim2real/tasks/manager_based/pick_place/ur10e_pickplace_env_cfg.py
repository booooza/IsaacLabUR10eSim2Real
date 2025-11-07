"""
Manager-based Pick-and-Place Environment for UR10e with Robotiq Hand-E Gripper.

This environment implements a curriculum learning approach with stages:
1. Reach: Basic end-effector positioning
2. Grasp: Adding gripper control (future)
3. Full Pick-and-Place: Complete manipulation (future)

Currently implements Stage 1 (Reach) with asymmetric actor-critic observations.
"""

from dataclasses import MISSING
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config.actions import GraspStageActionsCfg, ReachStageActionsCfg, ReachStageDeltaJointVelocityActionsCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config.curriculum import ReachStageCurriculumCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config.events import GraspStageEventCfg, ReachStageDeterministicEventCfg, ReachStageEventCfg, ReachStageRandomizeObjectOnSuccessEventCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config.observations import GraspStageObservationsCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config.rewards import GraspStageRewardsCfg, IsaacReachStageDenseRewardsCfg, IsaacReachStageHighSparseRewardsCfg, IsaacReachStageSparseRewardsCfg, LiftStageRewardsCfg, ReachStageRewardsCfg, ReachStageRewardsCfgV2, ReachStageRewardsCfgV3
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config.scene import LiftStageSceneCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.config.terminations import LiftStageTerminationsCfg, ReachStageSuccessTerminationsCfg, ReachStageTerminationsCfg, ReachStageTimeoutTerminationsCfg
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
        self.decimation = 2  # 30 Hz control frequency

        # episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))
        self.episode_length_s = 12.0  # ceil(8.0 / (2 * 0.004)) = 1000 steps per episode
        
        # Simulation settings
        # The physics simulation time-step (in seconds).
        # self.sim.dt = 0.004  # 1/0.004 = 250 Hz physics
        self.sim.dt = 1.0 / 60.0  # 60 Hz physics
        self.sim.render_interval = self.decimation

        # Target settings for sim-to-real transfer
        # self.decimation = 2  # 125 Hz control frequency
        # self.episode_length_s = 8.0 # ceil(8.0 / (2 * 0.004)) = 1000 steps per episode
        # self.sim.dt = 0.004  # 1/0.004 = 250 Hz physics

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

# IsaacReachStageDenseRewardsCfg
# IsaacReachStageSparseRewardsCfg
# IsaacReachStageHighSparseRewardsCfg

# ReachStageTimeoutTerminationsCfg
# ReachStageSuccessTerminationsCfg

# ReachStageEventCfg
# ReachStageRandomizeObjectOnSuccessEventCfg

@configclass
class ReachStageEnvCfg(PickPlaceEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=True)
    
    # MDP components
    actions: ReachStageActionsCfg = ReachStageActionsCfg()
    observations: ReachStageObservationsCfg = ReachStageObservationsCfg()
    events: ReachStageEventCfg = ReachStageEventCfg()
    rewards: ReachStageRewardsCfgV3 = ReachStageRewardsCfgV3()
    terminations: ReachStageTerminationsCfg = ReachStageTerminationsCfg()
    curriculum: ReachStageCurriculumCfg = ReachStageCurriculumCfg()
    
    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.seed = 42

        self.decimation = 2  # 62.5 Hz policy (125/2)
        self.sim.dt = 1 / 125  # 125 Hz physics
        self.episode_length_s = 5.0  # 312 steps (5.0s * 62.5 Hz)

        # Target settings for sim-to-real transfer
        # self.decimation = 1  # 125 Hz control frequency
        # self.episode_length_s = 5.0  # ~625 steps
        # self.sim.dt = 1.0 / 125.0  # 125 Hz 8 ms physics step
        # self.sim.render_interval = self.decimation

# Terminate on Timeout
@configclass
class ReachTimeout3EnvCfg(ReachStageEnvCfg):  # Sparse w=10, Timeout
    rewards: IsaacReachStageHighSparseRewardsCfg = IsaacReachStageHighSparseRewardsCfg()
    events: ReachStageEventCfg = ReachStageEventCfg()
    terminations: ReachStageTimeoutTerminationsCfg = ReachStageTimeoutTerminationsCfg()

# Terminate on Timeout
@configclass
class ReachStageDeltaJointVelocityActionsEnvCfg(ReachStageEnvCfg):
    actions: ReachStageDeltaJointVelocityActionsCfg = ReachStageDeltaJointVelocityActionsCfg()
    rewards: ReachStageRewardsCfgV3 = ReachStageRewardsCfgV3()
    events: ReachStageEventCfg = ReachStageEventCfg()
    terminations: ReachStageTimeoutTerminationsCfg = ReachStageTimeoutTerminationsCfg()
    curriculum = None

@configclass
class ReachStagePlayEnvCfg(ReachStageEnvCfg):
    """Configuration for reach environment during play/testing."""
    events: ReachStageDeterministicEventCfg = ReachStageDeterministicEventCfg()
    terminations: ReachStageTimeoutTerminationsCfg = ReachStageTimeoutTerminationsCfg()
    
    def __post_init__(self):
        # Run parent post-init
        super().__post_init__()
        
        self.seed = 999

        # Viewer settings
        self.viewer.eye = (1.5, -2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        
        # Longer episodes for testing
        # episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))
        self.episode_length_s = 5.0
        
        # Use only 1 environment for teleop
        self.scene.num_envs = 1

@configclass
class ReachStagePlayLoopEnvCfg(ReachStageEnvCfg):
    """Configuration for reach environment during play/testing."""
    events: ReachStageRandomizeObjectOnSuccessEventCfg = ReachStageRandomizeObjectOnSuccessEventCfg()
    terminations: ReachStageTimeoutTerminationsCfg = ReachStageTimeoutTerminationsCfg()
    
    def __post_init__(self):
        # Run parent post-init
        super().__post_init__()
        
        self.seed = 999

        # Viewer settings
        self.viewer.eye = (1.5, -2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        
        # Longer episodes for testing
        # episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))
        self.episode_length_s = 15.0
        
        # Use only 1 environment for teleop
        self.scene.num_envs = 1

@configclass
class GraspStageEnvCfg(PickPlaceEnvCfg):
    # MDP components
    actions: GraspStageActionsCfg = GraspStageActionsCfg()
    observations: GraspStageObservationsCfg = GraspStageObservationsCfg()
    events: GraspStageEventCfg = GraspStageEventCfg()
    rewards: GraspStageRewardsCfg = GraspStageRewardsCfg()
    terminations: PickPlaceTerminationsCfg = PickPlaceTerminationsCfg()
    curriculum: PickPlaceCurriculumCfg = PickPlaceCurriculumCfg()

@configclass
class LiftStageEnvCfg(PickPlaceEnvCfg):
    scene: LiftStageSceneCfg = LiftStageSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=True)
    actions: GraspStageActionsCfg = GraspStageActionsCfg()
    observations: GraspStageObservationsCfg = GraspStageObservationsCfg()
    events: GraspStageEventCfg = GraspStageEventCfg()
    rewards: LiftStageRewardsCfg = LiftStageRewardsCfg()
    terminations: LiftStageTerminationsCfg = LiftStageTerminationsCfg()
    curriculum: PickPlaceCurriculumCfg = PickPlaceCurriculumCfg()

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
