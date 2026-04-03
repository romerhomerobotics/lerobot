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
from run_bridge_client import BridgeROSInterface

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
        self._is_connected = False
        
        self.client = None
        self.interface = None
        
    @property
    def observation_features(self) -> Dict[str, type | Tuple]:
        # TODO: adjust these
        return {
            "full_rgb": (480, 640, 3),     
            "wrist_rgb": (480, 640, 3),    
            "proprio": dict,              
            "gripper": float,              
            "full_timestamp": object,      
            "wrist_timestamp": object,     
        }

    @property
    def action_features(self) -> Dict[str, type]:
        return {}

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
        
        obs_dict = {
            "full_rgb": full_rgb,
            "wrist_rgb": wrist_rgb,
            "proprio": proprio,
            "gripper": gripper,
            "full_timestamp": full_stamp,
            "wrist_timestamp": wrist_stamp,
        }
        
        print("OBS_DICT KEYS: ", obs_dict.keys())
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        pose_action = action["pose"]      
        gripper_action = action["gripper"] 
        
        self.interface.publish_action_pose(pose_action, gripper_action)
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