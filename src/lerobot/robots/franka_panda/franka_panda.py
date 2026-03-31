import logging
import time
from typing import Dict, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.robots.robot import Robot
from lerobot.robots.franka_panda.config_franka_panda import FrankaPandaRobotConfig

logger = logging.getLogger(__name__)

class FrankaPandaROS2Node(Node):
    def __init__(self, config: FrankaPandaRobotConfig):
        super().__init__('lerobot_franka_panda_node')
        self.config = config
        self.bridge = CvBridge()
        
        self.latest_joint_state = None
        self.latest_images = {cam_name: None for cam_name in self.config.cameras.keys()}
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.config.topic_joint_states,
            self.joint_state_callback,
            10
        )
        
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            self.config.topic_joint_commands,
            10
        )
        
        self.image_subs = {}
        for cam_name, cam_config in self.config.cameras.items():
            # Treat index_or_path as the ROS topic. If not provided, fallback to a standard name.
            topic = str(cam_config.index_or_path) if cam_config.index_or_path else f"/{cam_name}/image_raw"
            self.image_subs[cam_name] = self.create_subscription(
                Image,
                topic,
                self.make_image_callback(cam_name),
                10
            )

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def make_image_callback(self, cam_name: str):
        def image_callback(msg: Image):
            try:
                # Convert to BGR array (expected by OpenCV routines often used)
                # or 'rgb8' depending on your model. 'bgr8' is OpenCV default.
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_images[cam_name] = cv_image
            except Exception as e:
                logger.error(f"Error converting image for camera {cam_name}: {e}")
        return image_callback


class FrankaPanda(Robot):
    config_class = FrankaPandaRobotConfig
    name = "franka_panda"

    def __init__(self, config: FrankaPandaRobotConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self.node = None
        self.cameras = self.config.cameras # Make available to standard lerobot logic
        self._motors_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"
        ]

    @property
    def observation_features(self) -> Dict[str, type | Tuple]:
        features = {f"{m}.pos": float for m in self._motors_names}
        for cam_name, cam_config in self.config.cameras.items():
            # Assuming standard RGB/BGR 3-channel images
            features[cam_name] = (cam_config.height, cam_config.width, 3)
        return features

    @property
    def action_features(self) -> Dict[str, type]:
        return {f"{m}.pos": float for m in self._motors_names}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True # Handled in ROS2

    def calibrate(self) -> None:
        pass # Handled in ROS2

    def configure(self) -> None:
        pass # Handled in ROS2

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = FrankaPandaROS2Node(self.config)
        self._is_connected = True
        logger.info(f"{self} connected via ROS2.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Spin once to get the latest messages
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        obs_dict = {}
        
        # 1. Process Joint States
        if self.node.latest_joint_state is not None:
            # Match joint names from the message to extract positions
            names = self.node.latest_joint_state.name
            positions = self.node.latest_joint_state.position
            pos_dict = dict(zip(names, positions))
            
            for m in self._motors_names:
                obs_dict[f"{m}.pos"] = float(pos_dict.get(m, 0.0))
        else:
            # Fallback if no message received yet
            for m in self._motors_names:
                obs_dict[f"{m}.pos"] = 0.0
                
        # 2. Process Cameras
        for cam_name, cam_config in self.config.cameras.items():
            img = self.node.latest_images.get(cam_name)
            if img is not None:
                obs_dict[cam_name] = img
            else:
                # Provide a blank placeholder image if none received yet
                obs_dict[cam_name] = np.zeros(
                    (cam_config.height, cam_config.width, 3), 
                    dtype=np.uint8
                )
                
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        msg = Float32MultiArray()
        # Collect commanded positions
        msg.data = [float(action[f"{m}.pos"]) for m in self._motors_names if f"{m}.pos" in action]
        self.node.action_pub.publish(msg)
        return action

    @check_if_not_connected
    def disconnect(self):
        if self.node:
            self.node.destroy_node()
            self.node = None
        self._is_connected = False
        logger.info(f"{self} disconnected.")
