from isaaclab.utils import configclass
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp

@configclass
class PickPlaceCurriculumCfg:
    """Curriculum configuration (currently fixed to reach stage)."""
    
    # Stage tracking (for future implementation)
    # Current stage: 0 = reach, 1 = grasp, 2 = pick-and-place
    # stage = 0  # Fixed to reach for now
    pass