"""Event configuration for pick-and-place task."""

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg

# Import reset functions from MDP module
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp


@configclass
class PickPlaceEventCfg:
    """Event configuration for resets and randomization."""
    
    # Reset robot to home position
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (1.0, 1.0),  # Exact home position
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # Reset object pose
    reset_object = EventTermCfg(
        func=mdp.reset_object_pose,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "position_range_x": (0.4, 0.6),
            "position_range_y": (-0.2, 0.2),
            "height": 0.05,
        },
    )
    
    # Reset target pose
    reset_target = EventTermCfg(
        func=mdp.reset_target_pose,
        mode="reset",
        params={
            "target_cfg": SceneEntityCfg("target"),
            "position_range_x": (0.5, 0.7),
            "position_range_y": (-0.2, 0.2),
            "height": 0.02,
            "min_distance_from_object": 0.15,
        },
    )
