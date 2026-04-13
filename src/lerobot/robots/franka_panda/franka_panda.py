import logging
import time
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from ..robot import Robot
from .config_franka_panda import FrankaPandaRobotConfig
from .run_bridge_client import bridge_persistent_worker

logger = logging.getLogger(__name__)

class FrankaPanda(Robot):
    config_class = FrankaPandaRobotConfig
    name = "franka_panda"

    def __init__(self, config: FrankaPandaRobotConfig):
        super().__init__(config)
        self.config: FrankaPandaRobotConfig = config
        self._is_connected = False
        self.parent_conn, self.child_conn = Pipe()
        
        # Decide bridge mode based on config id
        mode = "sim"
        if hasattr(self.config, "id") and self.config.id:
             mode = self.config.id.split("_")[-1] if "_" in self.config.id else "sim"
             
        self.worker_process = Process(
            target=bridge_persistent_worker, 
            args=(self.child_conn, self.config, mode),
            daemon=True
        )
        
        # self.cameras is expected by lerobot-record for thread management.
        self.cameras = {name: None for name in self.config.cameras}
        
    @property
    def observation_features(self) -> Dict[str, type | Tuple]:
        features = {}
        
        # Cameras
        if self.config.use_cameras:
            for cam_name in self.config.cameras:
                h = self.config.cameras[cam_name].height
                w = self.config.cameras[cam_name].width
                features[cam_name] = (h, w, 3)

        # Gripper
        features["gripper"] = float

        # EEF Pose (6D: xyz + rpy)
        if self.config.use_eef:
            for i in range(6):
                features[f"pose_{i}"] = float

        # NOTE: We include a 1D 'state' vector for policy compatibility.
        # We use a list [dim] rather than a tuple (dim,) to avoid image misclassification.
        state_dim = 0
        if self.config.use_joints:
            state_dim += 7
        state_dim += 1 # gripper
        if self.config.use_eef:
            state_dim += 6 # xyz + rpy
            
        features["state"] = [state_dim]
        
        return features

    @property
    def action_features(self) -> Dict[str, type]:
        features = {}
        
        if self.config.use_joints:
            # Joint mode: 7 arm joints
            for i in range(7):
                features[f"joint_{i}"] = float
        else:
            # Cartesian mode: xyz + rpy (6D)
            for i in range(6):
                features[f"pose_{i}"] = float
        
        if self.config.use_gripper_action:
            features["gripper"] = float
            
        return features

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
        self.parent_conn.send("get_latest")
        status, state = self.parent_conn.recv()
        
        if status != "ok":
            print(f"[FrankaPanda] Error getting observation: {state}")
            fallback = {}
            if self.config.use_cameras:
                for cam_name in self.config.cameras:
                    h = self.config.cameras[cam_name].height
                    w = self.config.cameras[cam_name].width
                    fallback[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)
            
            if self.config.use_joints:
                for i in range(7): fallback[f"joint_{i}"] = 0.0
            
            fallback["gripper"] = 0.0
            
            if self.config.use_eef:
                for i in range(7): fallback[f"pose_{i}"] = 0.0
                
            return fallback

        full_rgb, wrist_rgb, proprio, gripper_state, joints, full_stamp, wrist_stamp = state
        
        obs_dict = {}

        # 1. Cameras
        if self.config.use_cameras:
            obs_dict["image"] = full_rgb if full_rgb is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            obs_dict["wrist_image"] = wrist_rgb if wrist_rgb is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 2. Individual features
        if self.config.use_joints:
            if joints is not None:
                arm_joints = joints[:7]
                for i in range(7):
                    obs_dict[f"joint_{i}"] = float(arm_joints[i])
            else:
                for i in range(7):
                    obs_dict[f"joint_{i}"] = 0.0

        obs_dict["gripper"] = float(gripper_state if gripper_state is not None else 0.0)

        # 3. State assembly for policy input
        state_parts = []
        
        if self.config.use_joints:
            # We assume 7D joints + 1D gripper? 
            # If the user model is 7D, they might be using only 7 joints 
            # OR 6 joints + 1 gripper. Given Cartesian is 7D, let's stick to Cartesian.
            if joints is not None:
                state_parts.append(joints[:7])
            else:
                state_parts.append(np.zeros(7))

        # Gripper is always part of state
        state_parts.append([obs_dict["gripper"]])

        if self.config.use_eef:
            if proprio is not None:
                # Convert Quat to RPY
                pos = proprio["eef_pos"]
                quat = proprio["eef_quat"] # [qx, qy, qz, qw]
                try:
                    rpy = R.from_quat(quat).as_euler('xyz')
                except:
                    rpy = np.zeros(3)
                
                # XYZ + RPY (6D)
                pose_6d = np.concatenate([pos, rpy])
                state_parts.append(pose_6d)
                for i in range(6):
                    obs_dict[f"pose_{i}"] = float(pose_6d[i])
            else:
                for i in range(6):
                    obs_dict[f"pose_{i}"] = 0.0
                state_parts.append(np.zeros(6))
        
        if state_parts:
            # Re-order to match model expectations: usually [gripper, pos, orientation] or similar
            # If user said "xyzrpy + gripper", they likely mean [x,y,z,r,p,y,gripper]
            # Let's check the size: 3(xyz) + 3(rpy) + 1(gripper) = 7.
            
            # The order in state_parts currently is [joints, gripper, pose_6d]
            # We will re-assemble for Cartesian mode specifically
            if self.config.use_eef and not self.config.use_joints:
                # [x,y,z, r,p,y, gripper]
                try:
                    final_state = np.array([
                        obs_dict.get("pose_0", 0.0),
                        obs_dict.get("pose_1", 0.0),
                        obs_dict.get("pose_2", 0.0),
                        obs_dict.get("pose_3", 0.0),
                        obs_dict.get("pose_4", 0.0),
                        obs_dict.get("pose_5", 0.0),
                        obs_dict.get("gripper", 0.0)
                    ], dtype=np.float32)
                    obs_dict["state"] = final_state
                except Exception as e:
                    print(f"[FrankaPanda] Error assembling final_state: {e}")
                    obs_dict["state"] = np.zeros(7, dtype=np.float32)
            else:
                obs_dict["state"] = np.concatenate(state_parts).astype(np.float32)
        
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.use_gripper_action:
            gripper_output = float(action["gripper"])
            # Map gripper from [-1, 1] to [0, 100] for the ROS controller
            gripper_mapped = (gripper_output + 1.0) * 50.0
            gripper_mapped = np.clip(gripper_mapped, 0.0, 100.0)
        else:
            gripper_mapped = 100.0 # Default

        if self.config.use_joints:
            # JOINT MODE
            joint_deltas = np.array([action[f"joint_{i}"] for i in range(7)], dtype=np.float32)
            self.parent_conn.send(("publish_joint_delta", joint_deltas, gripper_mapped))
        else:
            # CARTESIAN MODE (xyzrpy + gripper)
            # action contains pose_0..5 (xyz + rpy)
            delta_action = np.array([action[f"pose_{i}"] for i in range(6)], dtype=np.float32)
            self.parent_conn.send(("publish_delta", delta_action, gripper_mapped))
        
        return action

    @check_if_not_connected
    def disconnect(self):
        if self.worker_process.is_alive():
            try:
                print("[FrankaPanda] Requesting worker shutdown and log save...")
                self.parent_conn.send("shutdown")
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
        logger.info(f"{self} disconnected.")


if __name__ == "__main__":
    print("--- Starting Bridge Robot Client ---")
    config = FrankaPandaRobotConfig()
    panda = FrankaPanda(config)