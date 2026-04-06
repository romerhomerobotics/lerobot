import sys
import os

bridge_lib_path = os.path.expanduser('~/home_robotics/homerobotics_ws/src/ros_external/ros_external')
if bridge_lib_path not in sys.path:
    sys.path.append(bridge_lib_path)
from bridge_client import BridgeClient
from bridge_msg import BridgePacket, MESSAGE_CLASSES


import threading
import time
import cv2
import numpy as np

from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CompressedImage

IS_DEBUGGING = False

class BridgeROSInterface:
    def __init__(self, bridge_client, mode="sim"):
        self.client = bridge_client
        self.mode = mode
        
        print(f"[BridgeROSInterface] Initializing in mode: {mode}")

        # State storage and locks
        self._latest_full_rgb = None
        self._latest_full_stamp = None
        self._full_rgb_lock = threading.Lock()

        self._latest_wrist_rgb = None
        self._latest_wrist_stamp = None
        self._wrist_rgb_lock = threading.Lock()

        self._latest_proprio = None
        self._proprio_lock = threading.Lock()

        self._latest_gripper = None
        self._gripper_lock = threading.Lock()

        # Set up Subscriptions via Bridge
        if mode == "real":
            print("[BridgeROSInterface] Subscribing to COMPRESSED images (Real)")
            self.client.subscribe("/side_camera/color/image_compressed", self._full_cam_cb_decompress)
            self.client.subscribe("/wrist/color/image_compressed", self._wrist_cam_cb_decompress)
        else: # sim
            print("[BridgeROSInterface] Subscribing to RAW images (Sim)")
            self.client.subscribe("/camera_front/color/image_raw", self._full_cam_cb)
            self.client.subscribe("/wrist/color/image_raw", self._wrist_cam_cb)

        # Proprioception and Gripper subscriptions
        # Note: /eef_pose must be published by your ROS 2 server!
        self.client.subscribe("/eef_pose", self._proprio_cb)
        self.client.subscribe("/gripper_state", self._gripper_cb)
        
        # Start the background listening thread
        self.client.spin(background=True)
        print("[BridgeROSInterface] Background listener thread started.")

    # ------------------------------ CALLBACKS ------------------------------

    def _full_cam_cb(self, msg: Image):
        if IS_DEBUGGING:
            print(f"[INFO] Received RAW Full Cam Image at {time.time():.2f}")
        try:
            # Replaces cv_bridge: manually convert Image msg to numpy array
            np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            # Assuming incoming image is rgb8 or bgr8. If bgr8, convert:
            rgb = cv2.cvtColor(np_arr, cv2.COLOR_BGR2RGB) if msg.encoding == 'bgr8' else np_arr
            
            with self._full_rgb_lock:
                self._latest_full_rgb = rgb
                self._latest_full_stamp = msg.header.stamp
        except Exception as e:
            print(f"[ERROR] Failed to process full camera image: {e}")

    def _full_cam_cb_decompress(self, msg: CompressedImage):
        if IS_DEBUGGING:
            print(f"[INFO] Received COMPRESSED Full Cam Image at {time.time():.2f}")
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            with self._full_rgb_lock:
                self._latest_full_rgb = rgb
                self._latest_full_stamp = msg.header.stamp
        except Exception as e:
            print(f"[ERROR] Exception decoding image: {e}")

    def _wrist_cam_cb(self, msg: Image):
        if IS_DEBUGGING:
            print(f"[INFO] Received RAW Wrist Cam Image at {time.time():.2f}")
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            rgb = cv2.cvtColor(np_arr, cv2.COLOR_BGR2RGB) if msg.encoding == 'bgr8' else np_arr

            with self._wrist_rgb_lock:
                self._latest_wrist_rgb = rgb
                self._latest_wrist_stamp = msg.header.stamp
        except Exception as e:
            print(f"[ERROR] Failed to process wrist camera image: {e}")

    def _wrist_cam_cb_decompress(self, msg: CompressedImage):
        if IS_DEBUGGING:
            print(f"[INFO] Received COMPRESSED Wrist Cam Image at {time.time():.2f}")
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            with self._wrist_rgb_lock:
                self._latest_wrist_rgb = rgb
                self._latest_wrist_stamp = msg.header.stamp
        except Exception as e:
            print(f"[ERROR] Exception decoding image: {e}")

    def _proprio_cb(self, msg: Pose):
        if IS_DEBUGGING:
            print(f"[INFO] Received Proprioception (TF Pose) at {time.time():.2f}")
        pos = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)
        quat = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float32)
        
        with self._gripper_lock:
            gripper_val = self._latest_gripper if self._latest_gripper is not None else 0.0
        
        gripper = np.array([gripper_val], dtype=np.float32)
        proprio = {"eef_pos": pos, "eef_quat": quat, "gr_state": gripper}
        
        with self._proprio_lock:
            self._latest_proprio = proprio

    def _gripper_cb(self, msg: Float64):
        if IS_DEBUGGING:
            print(f"[INFO] Received Gripper State: {msg.data} at {time.time():.2f}")
        with self._gripper_lock:
            self._latest_gripper = float(msg.data)

    # ------------------------------ API / PUBLISHERS ------------------------------

    def get_latest(self):
        with self._full_rgb_lock:
            full_rgb = self._latest_full_rgb
            full_stamp = self._latest_full_stamp
        with self._wrist_rgb_lock:
            wrist_rgb = self._latest_wrist_rgb
            wrist_stamp = self._latest_wrist_stamp
        with self._proprio_lock:
            proprio = self._latest_proprio
        with self._gripper_lock:
            gripper = self._latest_gripper

        return (full_rgb, wrist_rgb, proprio, gripper, full_stamp, wrist_stamp)

    def go_to_default_joint(self):
        print("[ACTION] Sending default joint command")
        msg = Float64MultiArray()
        msg.data = [0.0, -0.7852, 0.0, -2.3563, 0.0, 1.5708, 0.7854, 0.0393, 0.0393]
        self.client.send_message("/joint_command", "std_msgs/Float64MultiArray", msg)
        time.sleep(3)

    def publish_action(self, action7, gripper):
        print("[ACTION] Publishing Cartesian Velocity Command")
        msg = Float64MultiArray()
        msg.data = action7.tolist()
        self.client.send_message("/cartesian_velocity_command", "std_msgs/Float64MultiArray", msg)

        gmsg = Float64()
        gmsg.data = float(gripper)
        self.client.send_message("/gripper_command", "std_msgs/Float64", gmsg)

    def publish_action_pose(self, action, gripper, base_pos=None, base_quat=None, default_pos=None):
        print("[ACTION] Publishing Cartesian Pose Command")
        current_pos = base_pos if base_pos is not None else default_pos

        if current_pos is not None and base_quat is not None:
            delta_pos = action[:3]
            delta_rpy = action[3:6]

            target_pos = current_pos + delta_pos
            base_rpy = R.from_quat(base_quat).as_euler('xyz', degrees=False)
            target_rpy = base_rpy + delta_rpy
            target_quat = R.from_euler('xyz', target_rpy, degrees=False).as_quat()

            action7 = np.concatenate([target_pos, target_quat])
        else:
            if len(action) == 7:
                action7 = action
            else:
                print(f"[ERROR] Received {len(action)}D action without base_pos/quat. Expected 7D.")
                return

        msg = Float64MultiArray()
        msg.data = action7.tolist()
        self.client.send_message("/cartesian_pose_command", "std_msgs/Float64MultiArray", msg)

        gmsg = Float64()
        gmsg.data = float(gripper)
        self.client.send_message("/gripper_command", "std_msgs/Float64", gmsg)
        
    def publish_delta_action(self, delta_6d, gripper = None):
        print(f"[ACTION] Publishing Cartesian Delta Command: {delta_6d}")
        msg = Float64MultiArray()
        delta_6d_scaled = delta_6d * 10
        msg.data = delta_6d_scaled.tolist()
        # msg.data = delta_6d.tolist()
        self.client.send_message("/cartesian_delta_command", "std_msgs/Float64MultiArray", msg)

        if gripper is not None: 
            gmsg = Float64()
            gmsg.data = float(gripper)
            self.client.send_message("/gripper_command", "std_msgs/Float64", gmsg)
        
    def shutdown(self):
        self.client.stop_spinning()

def bridge_persistent_worker(conn, mode="sim"):
    """
    Persistent worker process that manages the ZeroMQ Bridge and ROS Interface.
    Avoids GIL issues by isolating network I/O and image decoding.
    """
    print(f"[BridgeWorker] Starting in mode: {mode}")
    client = BridgeClient()
    interface = BridgeROSInterface(client, mode=mode)
    
    while True:
        try:
            cmd_data = conn.recv()
            if cmd_data == "shutdown":
                interface.shutdown()
                break
            elif cmd_data == "get_latest":
                state = interface.get_latest()
                conn.send(("ok", state))
            elif isinstance(cmd_data, tuple):
                cmd = cmd_data[0]
                if cmd == "publish_delta":
                    _, delta, gripper = cmd_data
                    interface.publish_delta_action(delta, gripper)
                elif cmd == "publish_pose":
                    _, pose, gripper = cmd_data
                    interface.publish_action_pose(pose, gripper)
                else:
                    conn.send(("error", f"Unknown tuple command: {cmd}"))
            else:
                conn.send(("error", f"Unknown command: {cmd_data}"))
        except EOFError:
            break
        except Exception as e:
            print(f"[BridgeWorker] Error: {e}")
            try:
                conn.send(("error", str(e)))
            except:
                pass
    conn.close()
    print("[BridgeWorker] Shutdown.")


if __name__=="__main__":
    from multiprocessing import Pipe, Process
    print("--- Testing ZeroMQ Bridge Persistent Worker ---")
    
    parent_conn, child_conn = Pipe()
    p = Process(target=bridge_persistent_worker, args=(child_conn, "sim"))
    p.start()

    time.sleep(2.0) # Give it time to connect

    # Test Delta Action via worker
    print("[Main] Sending test delta action...")
    delta = np.array([0.05, 0, 0, 0, 0, 0], dtype=np.float32)
    parent_conn.send(("publish_delta", delta, 50.0))
    
    # Test observation retrieval
    print("[Main] Requesting latest observation...")
    parent_conn.send("get_latest")
    status, state = parent_conn.recv()
    if status == "ok":
        full_rgb, wrist_rgb, proprio, gripper, full_stamp, wrist_stamp = state
        print(f"[Main] Received observation. Proprio: {proprio is not None}, Gripper: {gripper}")
    
    time.sleep(1.0)
    print("[Main] Shutting down...")
    parent_conn.send("shutdown")
    p.join()
