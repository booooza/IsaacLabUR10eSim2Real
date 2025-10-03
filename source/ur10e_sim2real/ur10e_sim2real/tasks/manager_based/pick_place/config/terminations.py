from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm

@configclass
class PickPlaceTerminationsCfg:
    """Termination conditions."""
    
    # Time out
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    
    # Success (for future stages)
    # task_success = DoneTerm(
    #     func=task_success_termination,
    #     params={
    #         "threshold": 0.05,
    #         "object_cfg": SceneEntityCfg("object"),
    #         "target_cfg": SceneEntityCfg("target"),
    #     },
    # )
    