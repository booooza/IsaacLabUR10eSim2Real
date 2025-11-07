from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import REACH_POSITION_SUCCESS_THRESHOLD, REACH_ROTATION_SUCCESS_THRESHOLD

@configclass
class PickPlaceTerminationsCfg:
    """Termination conditions."""
    
    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Object falling off table
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": -0.05,  # 5cm below table surface
            "asset_cfg": SceneEntityCfg("object")
        },
    )

@configclass
class ReachStageTerminationsCfg:
    """Termination conditions for reach stage."""
    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {
                "x": (-0.5, 0.5),
                "y": (-1.0, -0.2),
                "z": (0.0, 2.0),
            },
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    # abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)
    
@configclass
class LiftStageTerminationsCfg(PickPlaceTerminationsCfg):
    """Termination conditions."""
    
    # Success termination - reaches hover target within position thresholds
    # object_lifted = DoneTerm(
    #     func=mdp.lift_termination,
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "minimal_height": 0.20,
    #         "spawn_offset": 0.01,
    #     }
    # )

@configclass
class ReachStageTimeoutTerminationsCfg:
    """Termination conditions for reach stage."""
    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class ReachStageSuccessTerminationsCfg:
    """Termination conditions for reach stage."""
    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Success termination - reaches hover target within position thresholds
    success = DoneTerm(
        func=mdp.success_termination,
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": REACH_POSITION_SUCCESS_THRESHOLD, # 1 cm
            "rotation_threshold": REACH_ROTATION_SUCCESS_THRESHOLD # ~ 2.8 degrees
        }
    )
