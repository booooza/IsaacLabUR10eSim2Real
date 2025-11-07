"""Curriculum functions for pick-and-place task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp

"""Helper functions for curriculum."""

def set_weight_by_stage(
    env: "ManagerBasedRLEnv", 
    env_ids: torch.Tensor,
    stage_weights: dict
):
    # Map per-env stage to weight and take mean (or other aggregation)
    current_stages = env.stage[env_ids]
    weights = torch.tensor([stage_weights[int(s)] for s in current_stages], device=env.device)
    return weights.mean().item()

def modify_on_episode_count(env, env_ids, old_value, new_weight: float, num_episodes: int): 
    """Overrides a term's weight after a specified number of episodes."""

    # Check the total number of episodes completed
    # The counter is usually found on the episode manager.
    if env.episode_manager.current_episode_count > num_episodes:
        # We are modifying a weight (a float), so we return the new float value.
        return new_weight 
    
    return mdp.modify_term_cfg.NO_CHANGE # Keep the old value if threshold not met

def linear_increase(env, env_ids, old_value, start_weight, end_weight, num_steps):
    progress = min(env.common_step_counter / num_steps, 1.0)
    new_weight = start_weight + (end_weight - start_weight) * progress
    return new_weight
