from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("franka_panda")
@dataclass
class FrankaPandaRobotConfig(RobotConfig):
    topic_joint_states: str = "/eef_pose"
    topic_joint_commands: str = "/cartesian_pose_command"
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "camera_front": CameraConfig(
                width=640,
                height=480,
                fps=30
            ),
            "wrist": CameraConfig(
                width=640, 
                height=480,
                fps=30
            )
        }
    )