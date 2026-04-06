from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("franka_panda")
@dataclass
class FrankaPandaRobotConfig(RobotConfig):
    eef_states: str = "/eef_pose"
    eef_commands: str = "/cartesian_pose_command"
    
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