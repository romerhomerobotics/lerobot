import sys
import os
import rclpy

# 1. Add the path to the bridge library so Python can find bridge_server.py
bridge_lib_path = os.path.expanduser('~/home_robotics/homerobotics_ws/src/ros_external/ros_external')
if bridge_lib_path not in sys.path:
    sys.path.append(bridge_lib_path)

# 2. Import the BridgeServer class
from bridge_server import BridgeServer

def main(args=None):
    rclpy.init(args=args)
    
    print("--- Starting ZeroMQ Bridge Server ---")
    server = BridgeServer()
    
    # Register Camera Topics 
    # For SIM:
    server.register_topic('/camera_front/color/image_raw', 'sensor_msgs/Image')
    server.register_topic('/wrist/color/image_raw', 'sensor_msgs/Image')
    
    # For REAL 
    # server.register_topic('/side_camera/color/image_compressed', 'sensor_msgs/CompressedImage')
    # server.register_topic('/wrist/color/image_compressed', 'sensor_msgs/CompressedImage')

    # 5. Register State Topics
    # NOTE: You must have another ROS 2 node publishing the TF data to /eef_pose
    server.register_topic('/eef_pose', 'geometry_msgs/Pose') 
    server.register_topic('/gripper_state', 'std_msgs/Float64')
    server.register_topic('/joint_states', 'sensor_msgs/JointState')

    # 6. Register Action Topics (Allows the client to send commands to ROS 2)
    server.register_topic('/cartesian_velocity_command', 'std_msgs/Float64MultiArray')
    server.register_topic('/cartesian_pose_command', 'std_msgs/Float64MultiArray')
    server.register_topic('/cartesian_delta_command', 'std_msgs/Float64MultiArray')
    server.register_topic('/joint_command', 'std_msgs/Float64MultiArray')
    server.register_topic('/joint_delta_command', 'std_msgs/Float64MultiArray')
    server.register_topic('/gripper_command', 'std_msgs/Float64')
    
    print("\n[Server] Bridge Server is running and listening to ROS 2...")
    print("[Server] Press Ctrl+C to exit.\n")
    
    # 7. Keep the node alive to process ZMQ and ROS 2 messages
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        print("\n[Server] Caught KeyboardInterrupt. Shutting down...")
    finally:
        # Clean up the node properly
        server.destroy_node()
        rclpy.shutdown()
        print("[Server] Shutdown complete.")

if __name__ == '__main__':
    main()
