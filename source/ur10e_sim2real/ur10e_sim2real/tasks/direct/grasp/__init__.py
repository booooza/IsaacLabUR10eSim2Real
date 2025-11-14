# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Grasp-Direct-v0",
    entry_point=f"{__name__}.grasp_env:GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg:GraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_grasp_cfg.yaml",
    },
)

gym.register(
    id="Grasp-Direct-Play-v0",
    entry_point=f"{__name__}.grasp_env:GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg:GraspEnvPlayCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_grasp_cfg.yaml",
    },
)

gym.register(
    id="Grasp-Direct-SAC-v0",
    entry_point=f"{__name__}.grasp_env:GraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg:GraspEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_grasp_cfg.yaml",
    },
)
