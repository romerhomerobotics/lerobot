import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("franka_panda")
@dataclass
class FrankaPandaRobotConfig(RobotConfig):
    use_joints: bool = False
    use_eef: bool = True
    use_cameras: bool = True
    use_gripper_action: bool = True 
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image": CameraConfig(
                width=640,
                height=480,
                fps=30
            ),
            "wrist_image": CameraConfig(
                width=640, 
                height=480,
                fps=30
            )
        }
    )