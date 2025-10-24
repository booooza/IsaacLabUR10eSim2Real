from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.managers import CurriculumTermCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import REACH_POSITION_SUCCESS_THRESHOLD, REACH_ROTATION_SUCCESS_THRESHOLD 


@configclass
class PickPlaceCurriculumCfg:
    """Curriculum configuration (currently fixed to reach stage)."""
    # action_rate = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "action_rate_penalty",
    #         "weight": -0.001,
    #         "num_steps": 10000
    #     }
    # )
    
    # joint_vel = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "joint_vel_penalty_near_target",
    #         "weight": -0.5,
    #         "num_steps": 10000      # Ramp to: -0.5 over 5000 steps
    #     }
    # )
    pass