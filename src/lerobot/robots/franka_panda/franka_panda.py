import sys
import os
import threading
import logging
import time
from typing import Dict, Tuple

import numpy as np
import cv2

from std_msgs.msg import Float32MultiArray
from multiprocessing import Process, Pipe
from .run_bridge_client import bridge_persistent_worker

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
        
        # Multiprocessing setup for the bridge interface
        self.parent_conn, self.child_conn = Pipe()
        mode = self.config.id.split("_")[-1] if "_" in self.config.id else "sim"
        self.worker_process = Process(
            target=bridge_persistent_worker, 
            args=(self.child_conn, mode),
            daemon=True
        )
        
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
        if not self.worker_process.is_alive():
            self.worker_process.start()
        self._is_connected = True
        logger.info(f"{self} connected via persistent BridgeWorker process.")

    # ------------------------------ API ------------------------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Request data from worker
        self.parent_conn.send("get_latest")
        status, state = self.parent_conn.recv()
        
        if status != "ok":
            print(f"[FrankaPanda] Error getting observation: {state}")
            # Return empty/default if error
            return {
                "image": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
                "pose_0": 0.0, "pose_1": 0.0, "pose_2": 0.0,
                "pose_3": 0.0, "pose_4": 0.0, "pose_5": 0.0, "pose_6": 0.0,
            }

        full_rgb, wrist_rgb, proprio, gripper, full_stamp, wrist_stamp = state
        
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
        self.parent_conn.send(("publish_delta", delta_action, gripper_mapped))
        return action

    @check_if_not_connected
    def disconnect(self):
        if self.worker_process.is_alive():
            try:
                print("[FrankaPanda] Requesting worker shutdown and log save...")
                self.parent_conn.send("shutdown")
                # Wait for confirmation that logs are saved
                if self.parent_conn.poll(timeout=5.0):
                    resp = self.parent_conn.recv()
                    print(f"[FrankaPanda] Worker confirmed: {resp}")
                
                self.worker_process.join(timeout=2.0)
            except Exception as e:
                print(f"[FrankaPanda] Error during disconnect handshake: {e}")
            
            if self.worker_process.is_alive():
                print("[FrankaPanda] Worker still alive, terminating...")
                self.worker_process.terminate()
        
        self._is_connected = False
        logger.info(f"{self} disconnected and worker process stopped.")


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