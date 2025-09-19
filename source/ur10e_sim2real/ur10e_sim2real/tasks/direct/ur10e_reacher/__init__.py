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
    id="UR10e-Reacher-Direct-v0",
    entry_point=f"{__name__}.ur10e_reacher_env:UR10eReacherEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_reacher_env_cfg:UR10eReacherEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_cfg_ur10e_reacher.yaml",
    },
)

gym.register(
    id="UR10e-Reacher-Direct-Play-v0",
    entry_point=f"{__name__}.ur10e_reacher_env:UR10eReacherEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_reacher_env_cfg:UR10eReacherEnvCfgPlay",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_cfg_ur10e_reacher.yaml",
    },
)
