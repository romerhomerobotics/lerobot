import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64, Float64MultiArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.duration import Duration
import tf2_ros
from rclpy.qos import qos_profile_sensor_data


class ROSInterface(Node):
    def __init__(self, mode="sim"):
        super().__init__("openvla_oft_interface")

        self.get_logger().info(f"Initialized ROSInterface in mode: {mode}")

        self.bridge = CvBridge()
        self._latest_full_rgb = None
        self._latest_full_stamp = None
        self._full_rgb_lock = threading.Lock()

        self._latest_wrist_rgb = None
        self._latest_wrist_stamp = None
        self._wrist_rgb_lock = threading.Lock()

        self._latest_proprio = None
        self._proprio_lock = threading.Lock()

        self._side_cam_callback_group = MutuallyExclusiveCallbackGroup()
        self._wrist_cam_callback_group = MutuallyExclusiveCallbackGroup()
        self._proprio_callback_group = MutuallyExclusiveCallbackGroup()
        self._gripper_callback_group = MutuallyExclusiveCallbackGroup()

        self._last_full_cam_time = 0.0
        self._last_wrist_cam_time = 0.0


        # Subscribers
        if mode == "real":
            self.get_logger().info("Subscribing to COMPRESSED images for REAL robot (224x224)")
            self.sub_full = self.create_subscription(
                CompressedImage,
                "/side_camera/color/image_compressed",
                self._full_cam_cb_decompress,
                qos_profile_sensor_data,
                callback_group=self._side_cam_callback_group,
            )
            self.sub_wrist = self.create_subscription(
                CompressedImage,
                "/wrist/color/image_compressed",
                self._wrist_cam_cb_decompress,
                qos_profile_sensor_data,
                callback_group=self._wrist_cam_callback_group,
            )
        else: # sim
            self.get_logger().info("Subscribing to RAW images for SIMULATION")
            self.sub_full = self.create_subscription(
                Image,
                "/camera_front/color/image_raw",
                self._full_cam_cb,
                qos_profile_sensor_data,
                callback_group=self._side_cam_callback_group,
            )
            self.sub_wrist = self.create_subscription(
                Image,
                "/wrist/color/image_raw",
                self._wrist_cam_cb,
                qos_profile_sensor_data,
                callback_group=self._wrist_cam_callback_group,
            )

        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)

        self.proprio_timer = self.create_timer(
            0.01, self._proprio_cb, callback_group=self._proprio_callback_group
        )  

        self.pub_action = self.create_publisher(
            Float64MultiArray,
            "/cartesian_velocity_command",
            5
        )
        self.pub_action_pose = self.create_publisher(
            Float64MultiArray,
            "/cartesian_pose_command",
            7
        )
        # Joint command publisher
        self.pub_joint_action = self.create_publisher(
            Float64MultiArray,
            "/joint_command",
            7
        )

        # Gripper state storage
        self._latest_gripper = None
        self._gripper_lock = threading.Lock()

        # Gripper subscriber
        self.sub_gripper = self.create_subscription(
            Float64,
            "/gripper_state",
            self._gripper_cb,
            10,
            callback_group=self._gripper_callback_group
        )

        # Gripper command publisher
        self.pub_gripper = self.create_publisher(
            Float64,
            "/gripper_command",
            7
        )

    def _gripper_cb(self, msg: Float64):
        # self.get_logger().info("entered _gripper_cb")
        with self._gripper_lock:
            self._latest_gripper = float(msg.data)
        # self.get_logger().info("exiting _gripper_cb")

    def go_to_default_joint(self):
        msg = Float64MultiArray()
        msg.data = [0.0, -0.7852, 0.0, -2.3563, 0.0, 1.5708, 0.7854, 0.0393, 0.0393]
        self.pub_joint_action.publish(msg)
        time.sleep(3)

    # ------------------------------ CALLBACKS ------------------------------

    def _full_cam_cb(self, msg: Image):
        try:
           bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
           rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
           with self._full_rgb_lock:
               self._latest_full_rgb = rgb
               self._latest_full_stamp = msg.header.stamp
        except Exception as e:
           self.get_logger().error(f"Failed to process full camera image: {e}")

    def _full_cam_cb_decompress(self, msg: CompressedImage):
        # self.get_logger().info("entered _full_cam_cb_decompress")
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if rgb is None:
                self.get_logger().warn("Failed to decode compressed image")
                return
            with self._full_rgb_lock:
                self._latest_full_rgb = rgb
                self._latest_full_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"Exception decoding image: {e}")
        # self.get_logger().info("exiting _full_cam_cb_decompress")

    def _wrist_cam_cb(self, msg: Image):
        try:
           bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
           rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
           with self._wrist_rgb_lock:
               self._latest_wrist_rgb = rgb
               self._latest_wrist_stamp = msg.header.stamp
        except Exception as e:
           self.get_logger().error(f"Failed to process wrist camera image: {e}")

    def _wrist_cam_cb_decompress(self, msg: CompressedImage):
        # self.get_logger().info("entered _wrist_cam_cb_decompress")
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if rgb is None:
                self.get_logger().warn("Failed to decode compressed image")
                return
            with self._wrist_rgb_lock:
                self._latest_wrist_rgb = rgb
                self._latest_wrist_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"Exception decoding image: {e}")


    def _proprio_cb(self):
        # self.get_logger().info("entered _proprio_cb")
        try:
            # Lookup latest transform INSTANTLY without blocking the thread
            # In ROS2, Time() with no args means "latest available"
            now = rclpy.time.Time()
            tr = self.tfBuffer.lookup_transform(
                "panda_link0", 
                "eef", 
                now,
                timeout=Duration(seconds=0.0)
            ).transform

            pos = np.array([tr.translation.x, tr.translation.y, tr.translation.z], dtype=np.float32)
            quat = np.array([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w], dtype=np.float32)
            
            with self._gripper_lock:
                gripper_val = self._latest_gripper if self._latest_gripper is not None else 0.0
            
            gripper = np.array([gripper_val], dtype=np.float32)
            proprio = {"eef_pos": pos, "eef_quat": quat, "gr_state": gripper}
            
            with self._proprio_lock:
                self._latest_proprio = proprio

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # Fail silently or with a warning to avoid log spam during startup
            self.get_logger().warn(f"Waiting for TF (panda_link0 -> eef): {e}", throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().error(f"Unexpected error in _proprio_cb: {e}")
        # self.get_logger().info("exiting _proprio_cb")


    # ------------------------------ API ------------------------------

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

        return (
            full_rgb,
            wrist_rgb,
            proprio,
            gripper,
            full_stamp,
            wrist_stamp
        )

    def publish_action(self, action7, gripper):
        # Publish Cartesian delta / joint vel
        msg = Float64MultiArray()
        msg.data = action7.tolist()
        self.pub_action.publish(msg)

        # Publish single-float gripper command
        gmsg = Float64()
        gmsg.data = float(gripper)
        self.pub_gripper.publish(gmsg)

    def publish_action_pose(self, action, gripper, base_pos=None, base_quat=None, default_pos=None):
        # Determine base pose
        current_pos = base_pos if base_pos is not None else default_pos

        if current_pos is not None and base_quat is not None:
            # Action is delta (dx, dy, dz, dr, dp, dy)
            delta_pos = action[:3]
            delta_rpy = action[3:6]

            target_pos = current_pos + delta_pos

            # Rotation
            # Convert base quat to Euler
            base_rpy = R.from_quat(base_quat).as_euler('xyz', degrees=False)
            target_rpy = base_rpy + delta_rpy
            target_quat = R.from_euler('xyz', target_rpy, degrees=False).as_quat()

            # Construct 7D pose
            action7 = np.concatenate([target_pos, target_quat])
        else:
            # Action is absolute pose (7D)
            if len(action) == 7:
                action7 = action
            else:
                self.get_logger().error(
                    f"publish_action_pose received {len(action)}D action without base_pos/quat. Expected 7D."
                )
                return

        # Publish Cartesian pose
        msg = Float64MultiArray()
        msg.data = action7.tolist()
        self.pub_action_pose.publish(msg)

        # Publish single-float gripper command
        gmsg = Float64()
        gmsg.data = float(gripper)
        self.pub_gripper.publish(gmsg)
