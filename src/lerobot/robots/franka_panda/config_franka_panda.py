from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("franka_panda")
@dataclass
class FrankaPandaRobotConfig(RobotConfig):
    topic_joint_states: str = "/franka/joint_states"
    topic_joint_commands: str = "/franka/joint_commands"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
