import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reach_success_termination(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    hover_height: float = 0.15,
) -> torch.Tensor:
    """Terminate when EE reaches hover point above object."""
    object_entity = env.scene[object_cfg.name]
    target_frame = env.scene[target_cfg.name]
    
    object_pos = object_entity.data.root_pos_w[:, :3]
    
    # Hover target
    hover_target = object_pos.clone()
    hover_target[:, 2] += hover_height
    
    # EE position
    ee_pos = target_frame.data.target_pos_w[..., 0, :3]
    
    distance = torch.norm(ee_pos - hover_target, p=2, dim=-1)
    success = distance < threshold
    
    return success
