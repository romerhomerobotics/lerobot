import sys
import os
import threading
import logging
import time
from typing import Dict, Tuple

import numpy as np
import cv2

# For bridge_client.py
bridge_lib_path = os.path.expanduser('~/home_robotics/homerobotics_ws/src/ros_external/ros_external')
if bridge_lib_path not in sys.path:
    sys.path.append(bridge_lib_path)
from bridge_client import BridgeClient
from .run_bridge_client import BridgeROSInterface

from std_msgs.msg import Float32MultiArray

# LeRobot imports
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.robots.robot import Robot
from lerobot.robots.franka_panda.config_franka_panda import FrankaPandaRobotConfig

logger = logging.getLogger(__name__)

class FrankaPanda(Robot):
    config_class = FrankaPandaRobotConfig
    name = "franka_panda"

    def __init__(self, config: FrankaPandaRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = config.cameras
        self._is_connected = False
        
        self.client = None
        self.interface = None
        
    @property
    def observation_features(self) -> Dict[str, type | Tuple]:
        features = {
            "image": (480, 640, 3), # (height, width, channels)
            "wrist_image": (480, 640, 3),
        }
        # 7D state (xyz + quat)
        for i in range(7):
            features[f"pose_{i}"] = float
        return features

    @property
    def action_features(self) -> Dict[str, type]:
        # 7D action (xyz + quat)
        return {f"pose_{i}": float for i in range(7)}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True 

    def calibrate(self) -> None:
        pass 

    def configure(self) -> None:
        pass 

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.client = BridgeClient()
        self.interface = BridgeROSInterface(self.client, mode="sim") 
        
        self._is_connected = True
        logger.info(f"{self} connected via BridgeROSInterface.")

    # ------------------------------ API ------------------------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Fetch data from interface
        full_rgb, wrist_rgb, proprio, gripper, full_stamp, wrist_stamp = self.interface.get_latest()
        
        # proprio = {"eef_pos": pos, "eef_quat": quat, "gr_state": gripper}
        if proprio is not None:
             pose_7d = np.concatenate([
                proprio["eef_pos"], 
                proprio["eef_quat"]
            ]).astype(np.float32)
        else:
            pose_7d = np.zeros(7, dtype=np.float32)
            
        obs_dict = {
            "image": full_rgb if full_rgb is not None else np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist_image": wrist_rgb if wrist_rgb is not None else np.zeros((480, 640, 3), dtype=np.uint8),
        }
        
        # Add individual pose components
        for i in range(7):
            obs_dict[f"pose_{i}"] = float(pose_7d[i])
            
        # logger.debug("OBS_DICT KEYS: ", obs_dict.keys())
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        # 7D action from policy = [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
        action_vec = np.array([action[f"pose_{i}"] for i in range(7)], dtype=np.float32)
        
        delta_action = action_vec[:6]  # delta xyzrpy
        gripper_output = action_vec[6] # -1 (close) to 1 (open)
        
        # Map gripper from [-1, 1] to [0, 100] for the ROS controller
        gripper_mapped = (gripper_output + 1.0) * 50.0
        gripper_mapped = np.clip(gripper_mapped, 0.0, 100.0)

        print(f"[FrankaPanda] Delta: {delta_action}, Gripper: {gripper_mapped:.1f}")        
        # self.interface.publish_delta_action(delta_action, gripper_mapped)
        self.interface.publish_delta_action(delta_action, None)
        return action

    @check_if_not_connected
    def disconnect(self):
        if self.interface:
            self.interface.shutdown()  
            self.interface = None
            self.client = None
        self._is_connected = False
        logger.info(f"{self} disconnected from BridgeROSInterface.")


if __name__ == "__main__":
    print("--- Starting Bridge Robot Client ---")
    
    config = FrankaPandaRobotConfig()
    panda = FrankaPanda(config)
    
    try:
        print("Connecting to bridge server...")
        panda.connect()
        print("Connected! Listening for observations...\n")
        
        while True:
            obs = panda.get_observation()
            
            # Formatted printing so the terminal doesn't get spammed with massive arrays
            print(f"\n--- Latest Observation @ {time.time():.2f} ---")
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    print(f"{k:18}: Array shape {v.shape}")
                elif isinstance(v, dict):
                    print(f"{k:18}: Dict with keys {list(v.keys())}")
                else:
                    print(f"{k:18}: {v}")
                    
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt. Shutting down...")
        
    finally:
        panda.disconnect()
        print("Shutdown complete.")