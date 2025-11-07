import torch 

from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from isaaclab.managers import CurriculumTermCfg
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import REACH_POSITION_SUCCESS_THRESHOLD, REACH_ROTATION_SUCCESS_THRESHOLD

@configclass
class ReachStageCurriculumCfg:
    # horizon_length
    # epoch = num_steps / horizon_length
    # action_rate = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 25000} # around epoch 200
    # )
    # horizon_length = CurriculumTermCfg(

    # Action rate limit starting value is -0.01
    # horizon_length 128 epoch 200 = 25000 steps
    # horizon_length 24 epoch 200 = 41666 steps
    
    action_rate_limit = CurriculumTermCfg(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate_limit", "weight": -0.3, "num_steps": 4750} # around epoch 200
    )
    
@configclass
class PickPlaceCurriculumCfg:
    """Curriculum configuration (currently fixed to reach stage)."""
    # action_rate_penalty = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "action_rate_penalty",
    #         "weight": -0.005,
    #         "num_steps": 150
    #     }
    # )
    # joint_vel = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight, 
    #     params={
    #         "term_name": "joint_vel",
    #         "weight": -0.001, 
    #         "num_steps": 150
    #     }
    # )
    # joint_vel_penalty_near_target = CurriculumTermCfg(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "joint_vel_penalty_near_target",
    #         "weight": -0.001,
    #         "num_steps": 150
    #     }
    # )
