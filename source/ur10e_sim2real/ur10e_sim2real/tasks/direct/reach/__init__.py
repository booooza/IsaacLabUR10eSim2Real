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
    id="Reach-Direct-v0",
    entry_point=f"{__name__}.reach_env:ReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:ReachEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_reach_cfg.yaml",
    },
)

gym.register(
    id="Reach-Direct-SAC-v0",
    entry_point=f"{__name__}.reach_env:ReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:ReachEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_reach_cfg.yaml",
    },
)
