from __future__ import annotations

from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp import delta_joint_actions
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING
from isaaclab.utils import configclass

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.envs.mdp.actions import JointActionCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import actions_cfg

@configclass
class DeltaJointVelocityActionCfg(JointActionCfg):
    """Configuration for delta joint velocity action (OI∆).
    
    Inherits from JointActionCfg to get scale, offset, clip functionality.
    """
    
    class_type: type[ActionTerm] = delta_joint_actions.DeltaJointVelocityAction
    
    c_scale: float = 1.0
    """Scaling factor for delta integration. Max velocity change = c_scale * action."""
