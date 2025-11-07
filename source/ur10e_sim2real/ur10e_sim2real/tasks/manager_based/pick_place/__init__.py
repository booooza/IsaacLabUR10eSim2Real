import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="UR10e-Sim2Real-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_pickplace_env_cfg:ReachStageEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_reach_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_reach:UR10eReachPPORunnerCfg",
    },
)

gym.register(
    id="UR10e-Sim2Real-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_pickplace_env_cfg:PickPlaceEnvPlayCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="UR10e-Sim2Real-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_pickplace_env_cfg:GraspStageEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="UR10e-Sim2Real-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_pickplace_env_cfg:LiftStageEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_lift_cfg.yaml",
    },
)

gym.register(
    id="UR10e-Sim2Real-Grasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_pickplace_env_cfg:PickPlaceEnvPlayCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

# Terminate on Timeout Experiments
gym.register(
    id="UR10e-Reach-Timeout-3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_pickplace_env_cfg:ReachTimeout3EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_reach_cfg.yaml",
    },
)

gym.register(
    id="UR10e-Reach-Delta",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_pickplace_env_cfg:ReachStageDeltaJointVelocityActionsEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_reach_cfg.yaml",
    },
)


gym.register(
    id="UR10e-Reach-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_pickplace_env_cfg:ReachStagePlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_reach_cfg.yaml",
    },
)

gym.register(
    id="UR10e-Reach-Play-Loop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10e_pickplace_env_cfg:ReachStagePlayLoopEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_reach_cfg.yaml",
    },
)
