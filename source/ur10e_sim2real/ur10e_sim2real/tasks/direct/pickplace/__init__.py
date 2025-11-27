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
    id="PickPlace-Direct-v0",
    entry_point=f"{__name__}.pickplace_env:PickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:PickPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pickplace_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_pickplace_cfg.yaml",
    },
)

gym.register(
    id="PickPlace-Direct-Play-v0",
    entry_point=f"{__name__}.pickplace_env:PickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:PickPlaceEnvPlayCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pickplace_cfg.yaml",
    },
)

gym.register(
    id="PickPlace-Direct-SAC-v0",
    entry_point=f"{__name__}.pickplace_env:PickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_env_cfg:PickPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_pickplace_cfg.yaml",
    },
)
