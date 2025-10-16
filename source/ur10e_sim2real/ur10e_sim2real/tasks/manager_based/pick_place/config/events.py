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
    reset_articulation_to_default = EventTermCfg(
        func=mdp.reset_articulation_to_default,
        mode="reset",
    )

    # Reset object and target poses
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (-0.25, 0.25),      # 0.0 ± 0.25 = [-0.25, 0.25]
                "y": (0.25, -0.25),      # -0.45 ± 0.25 = [-0.7, -0.2]
                "z": (0.0, 0.0), 
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_target = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "pose_range": {
                "x": (-0.25, 0.25),      # 0.0 ± 0.25 = [-0.25, 0.25]
                "y": (0.25, -0.25),      # -0.45 ± 0.25 = [-0.7, -0.2]
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    randomize_object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_ids=[0]),  # RigidObject has only one body
            "mass_distribution_params": (0.01, 0.5),  # 10g to 500g
            "operation": "abs",  # Set absolute mass values
            "distribution": "uniform",
        },
    )

    # This event term randomizes the scale of the cube.
    # The mode is set to 'prestartup', which means that the scale is randomize on the USD stage before the
    # simulation starts.
    # Note: USD-level randomizations require the flag 'replicate_physics' to be set to False.
    # Base size: 2.5cm  (Range: 1cm to 4.5cm)
    randomize_object_scale = EventTermCfg(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",  # Must be at startup, not reset. Every environment must keep its scale!
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": {
                "x": (0.4, 1.8),   # 2.5cm * 0.4 = 1cm, 2.5cm * 1.8 = 4.5cm
                "y": (0.4, 1.8),
                "z": (0.4, 1.8),
            },
        },
    )

    # This event term randomizes the visual color of the cube.
    # Similar to the scale randomization, this is also a USD-level randomization and requires the flag
    # 'replicate_physics' to be set to False.
    randomize_color = EventTermCfg(
        func=mdp.randomize_visual_color,
        mode="prestartup",
        params={
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "asset_cfg": SceneEntityCfg("object"),
            "mesh_name": "geometry/mesh",
            "event_name": "rep_cube_randomize_color",
        },
    )

    # Custom logging
    log_custom_metrics = EventTermCfg(
        func=mdp.log_custom_metrics,
        mode="reset",
    )
