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
from sensor_msgs.msg import Image, CompressedImage, JointState

IS_DEBUGGING = False

from .config_franka_panda import FrankaPandaRobotConfig

class BridgeROSInterface:
    def __init__(self, bridge_client, config: FrankaPandaRobotConfig, mode="sim"):
        self.client = bridge_client
        self.config = config
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

        self._latest_joints = None
        self._joints_lock = threading.Lock()

        # Logging setup
        self.log_dir = os.path.expanduser("~/lerobot/outputs/test_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self._action_buffer = []
        self._state_buffer = []
        self._cam_dt_buffer = []
        self._full_imgs = []
        self._wrist_imgs = []
        self._last_full_frame_t = None
        self._last_wrist_frame_t = None
        self._last_save_time = time.time()

        # Subscriptions based on flags
        if self.config.use_cameras:
            if mode == "real":
                print("[BridgeROSInterface] Subscribing to COMPRESSED images (Real)")
                self.client.subscribe("/side_camera/color/image_compressed", self._full_cam_cb_decompress)
                self.client.subscribe("/wrist/color/image_compressed", self._wrist_cam_cb_decompress)
            else: # sim
                print("[BridgeROSInterface] Subscribing to RAW images (Sim)")
                self.client.subscribe("/camera_front/color/image_raw", self._full_cam_cb)
                self.client.subscribe("/wrist/color/image_raw", self._wrist_cam_cb)

        if self.config.use_eef:
            print("[BridgeROSInterface] Subscribing to /eef_pose")
            self.client.subscribe("/eef_pose", self._proprio_cb)
            
        if self.config.use_joints:
            print("[BridgeROSInterface] Subscribing to /joint_states")
            self.client.subscribe("/joint_states", self._joint_states_cb)
        
        self.client.subscribe("/gripper_state", self._gripper_cb)
        
        # Start the background listening thread
        self.client.spin(background=True)
        print("[BridgeROSInterface] Background listener thread started.")

    # ------------------------------ CALLBACKS ------------------------------

    def _full_cam_cb(self, msg: Image):
        if IS_DEBUGGING:
            print(f"[INFO] Received RAW Full Cam Image at {time.time():.2f}")
        start_time = time.perf_counter()
        try:
            # Replaces cv_bridge: manually convert Image msg to numpy array
            np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            # Assuming incoming image is rgb8 or bgr8. If bgr8, convert:
            rgb = cv2.cvtColor(np_arr, cv2.COLOR_BGR2RGB) if msg.encoding == 'bgr8' else np_arr
            
            with self._full_rgb_lock:
                self._latest_full_rgb = rgb
                self._latest_full_stamp = msg.header.stamp
            
            # Log full camera image
            self._full_imgs.append(rgb.copy())
            
            # Log dt for full camera
            now = time.time()
            if self._last_full_frame_t is not None:
                dt = now - self._last_full_frame_t
                self._cam_dt_buffer.append([now, dt, 0]) # 0 for full_cam
            self._last_full_frame_t = now

            elapsed = (time.perf_counter() - start_time) * 1000
            if IS_DEBUGGING:
                print(f"[Bridge] Full cam decode: {elapsed:.2f}ms")
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
        start_time = time.perf_counter()
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            rgb = cv2.cvtColor(np_arr, cv2.COLOR_BGR2RGB) if msg.encoding == 'bgr8' else np_arr

            with self._wrist_rgb_lock:
                self._latest_wrist_rgb = rgb
                self._latest_wrist_stamp = msg.header.stamp
            
            # Log wrist camera image
            self._wrist_imgs.append(rgb.copy())
            
            # Log dt for wrist camera
            now = time.time()
            if self._last_wrist_frame_t is not None:
                dt = now - self._last_wrist_frame_t
                self._cam_dt_buffer.append([now, dt, 1]) # 1 for wrist_cam
            self._last_wrist_frame_t = now

            elapsed = (time.perf_counter() - start_time) * 1000
            if IS_DEBUGGING:
                print(f"[Bridge] Wrist cam decode: {elapsed:.2f}ms")
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
        
        # Logging the state
        self._state_buffer.append([time.time(), *pos, *quat, gripper_val])

    def _gripper_cb(self, msg: Float64):
        if IS_DEBUGGING:
            print(f"[INFO] Received Gripper State: {msg.data} at {time.time():.2f}")
        with self._gripper_lock:
            self._latest_gripper = float(msg.data)

    def _joint_states_cb(self, msg: JointState):
        if IS_DEBUGGING:
            print(f"[INFO] Received Joint States at {time.time():.2f}")
        
        # Capture positions for arm (first 7) and gripper (rest)
        # Assuming the first 7 are panda joints based on mono_controller_sim.py
        pos = np.array(msg.position, dtype=np.float32)
        if len(pos) > 0:
            print(f"[DEBUG] Joint States Received: {len(pos)} joints, first: {pos[0]:.3f}")
        else:
            print("[DEBUG] WARNING: Received EMPTY Joint States message!")
            
        with self._joints_lock:
            self._latest_joints = pos
        
        # Also log to state buffer if needed, but primarily used for observation

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
        with self._joints_lock:
            joints = self._latest_joints

        return (full_rgb, wrist_rgb, proprio, gripper, joints, full_stamp, wrist_stamp)

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
        msg = Float64MultiArray()
        # NOTE: this converts position to velocity which is what is used in /cartesian_delta_command
        # We assume robot control freq is ~30Hz, so multiply by 30 to get velocity
        delta_scaled = delta_6d * 30 
        msg.data = delta_scaled.tolist()
        self.client.send_message("/cartesian_delta_command", "std_msgs/Float64MultiArray", msg)
        
        if gripper is not None:
             gmsg = Float64()
             gmsg.data = float(gripper)
             self.client.send_message("/gripper_command", "std_msgs/Float64", gmsg)
             
        # Log action
        self._action_buffer.append([time.time(), *delta_scaled, gripper if gripper is not None else 0.0])

    def publish_joint_delta(self, joint_delta, gripper=None):
        print(f"[ACTION] Publishing Joint Delta Command: {joint_delta}")
        msg = Float64MultiArray()
        msg.data = joint_delta.tolist()
        self.client.send_message("/joint_delta_command", "std_msgs/Float64MultiArray", msg)
        
        if gripper is not None:
            gmsg = Float64()
            gmsg.data = float(gripper)
            self.client.send_message("/gripper_command", "std_msgs/Float64", gmsg)
        
        # Log action
        self._action_buffer.append([time.time(), *joint_delta, gripper if gripper is not None else 0.0])
        
    def shutdown(self):
        print("[Bridge] Shutting down and saving logs...")
        self._save_logs()
        self.client.stop_spinning()

    def _save_logs(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save actions
        if self._action_buffer:
            try:
                filename = os.path.join(self.log_dir, f"actions_{timestamp}.npy")
                data = np.array(self._action_buffer, dtype=np.float32)
                np.save(filename, data)
                print(f"[Bridge] Action logs saved: {filename} ({len(data)} samples)")
            except Exception as e:
                print(f"[Bridge] Failed to save action logs: {e}")

        # Save states
        if self._state_buffer:
            try:
                filename = os.path.join(self.log_dir, f"states_{timestamp}.npy")
                data = np.array(self._state_buffer, dtype=np.float32)
                np.save(filename, data)
                print(f"[Bridge] State logs saved: {filename} ({len(data)} samples)")
            except Exception as e:
                print(f"[Bridge] Failed to save state logs: {e}")

        # Save camera dt logs
        if self._cam_dt_buffer:
            try:
                filename = os.path.join(self.log_dir, f"cam_dt_{timestamp}.npy")
                data = np.array(self._cam_dt_buffer, dtype=np.float32)
                np.save(filename, data)
                print(f"[Bridge] Camera DT logs saved: {filename} ({len(data)} samples)")
            except Exception as e:
                print(f"[Bridge] Failed to save camera DT logs: {e}")

        # Save images as videos
        fps_out = 30 # Default target FPS for playback
        if self._full_imgs:
            try:
                filename = os.path.join(self.log_dir, f"full_cam_{timestamp}.mp4")
                h, w, _ = self._full_imgs[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, fps_out, (w, h))
                for frame in self._full_imgs:
                    # Convert RGB (buffer) to BGR (OpenCV writer)
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"[Bridge] Full camera video saved: {filename} ({len(self._full_imgs)} frames)")
            except Exception as e:
                print(f"[Bridge] Failed to save full camera video: {e}")

        if self._wrist_imgs:
            try:
                filename = os.path.join(self.log_dir, f"wrist_cam_{timestamp}.mp4")
                h, w, _ = self._wrist_imgs[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, fps_out, (w, h))
                for frame in self._wrist_imgs:
                    # Convert RGB (buffer) to BGR (OpenCV writer)
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"[Bridge] Wrist camera video saved: {filename} ({len(self._wrist_imgs)} frames)")
            except Exception as e:
                print(f"[Bridge] Failed to save wrist camera video: {e}")

def bridge_persistent_worker(conn, config, mode="sim"):
    """
    Persistent worker process that manages the ZeroMQ Bridge and ROS Interface.
    Avoids GIL issues by isolating network I/O and image decoding.
    """
    import signal
    # Ignore SIGINT (Ctrl+C) so the parent can handle the shutdown sequence
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    print(f"[BridgeWorker] Starting in mode: {mode}")
    # Use the config passed from the parent process
    
    client = BridgeClient()
    interface = BridgeROSInterface(client, config=config, mode=mode)
    
    try:
        while True:
            try:
                cmd_data = conn.recv()
                if cmd_data == "shutdown":
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
                    elif cmd == "publish_joint_delta":
                        _, joints, gripper = cmd_data
                        interface.publish_joint_delta(joints, gripper)
                    else:
                        conn.send(("error", f"Unknown tuple command: {cmd}"))
                else:
                    conn.send(("error", f"Unknown command: {cmd_data}"))
            except EOFError:
                break
            except Exception as e:
                print(f"[BridgeWorker] Error during command process: {e}")
                try:
                    conn.send(("error", str(e)))
                except:
                    pass
    finally:
        print("[BridgeWorker] Ensuring logs are saved before exiting...")
        interface.shutdown()
        try:
            conn.send("saved") # Final confirmation
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
