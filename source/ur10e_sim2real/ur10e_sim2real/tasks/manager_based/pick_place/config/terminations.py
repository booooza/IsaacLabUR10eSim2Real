from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg

@configclass
class PickPlaceTerminationsCfg:
    """Termination conditions."""
    
    # Time out
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    
    # Object falling off table
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": -0.05,  # 5cm below table surface
            "asset_cfg": SceneEntityCfg("object")
        },
    )

    # Success termination - reaches hover target within position thresholds
    # reach_success = DoneTerm(
    #     func=mdp.reach_termination,
    #     params={
    #         "source_frame_cfg": SceneEntityCfg("ee_frame"),
    #         "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
    #         "position_threshold": 0.1,
    #         "rotation_threshold": 0.1,
    #     },
    # )
