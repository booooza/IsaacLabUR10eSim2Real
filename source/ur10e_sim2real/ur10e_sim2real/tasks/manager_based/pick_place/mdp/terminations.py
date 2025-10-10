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
):
    """Terminate episode and mark success if reach threshold is met."""
    object_entity = env.scene[object_cfg.name]
    target_frame = env.scene[target_cfg.name]

    object_pos = object_entity.data.root_pos_w[:, :3]
    target_pos = target_frame.data.target_pos_w[..., 0, :3]

    distance = torch.norm(object_pos - target_pos, p=2, dim=-1)
    success = distance < threshold

    # Return success boolean tensor; typically combined with other done conditions elsewhere
    return success
