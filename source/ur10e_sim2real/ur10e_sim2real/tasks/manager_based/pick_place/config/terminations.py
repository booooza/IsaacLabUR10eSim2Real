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

    # Success (for reach stage)
    reach_success = DoneTerm(
        func=mdp.reach_success_termination,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("ee_frame"),
            "threshold": 0.01, # Success threshold in meters
            "hover_height": 0.05, # Hover height above target in meters
        },
    )
