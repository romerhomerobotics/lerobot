import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import tf2_ros

class TfToPosePublisher(Node):
    def __init__(self):
        super().__init__('tf_to_pose_publisher')
        
        # 1. Create a publisher for the pose topic
        self.publisher_ = self.create_publisher(Pose, '/eef_pose', 10)
        
        # 2. Set up the TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 3. Create a timer to query TF and publish at 50Hz (0.02 seconds)
        # You can adjust this frequency based on how fast your client needs updates
        self.timer = self.create_timer(0.01, self.publish_pose)
        
        self.get_logger().info("TF to Pose Publisher started. Waiting for transforms...")

    def publish_pose(self):
        try:
            # Look up the latest available transform from panda_link0 to eef
            # Note: The target frame is usually the first argument, and the source frame is the second.
            trans = self.tf_buffer.lookup_transform(
                'panda_link0', # Target frame (Base)
                'eef',         # Source frame (End Effector)
                rclpy.time.Time()
            )
            
            # Create the Pose message
            pose_msg = Pose()
            
            # Extract Translation -> Position
            pose_msg.position.x = trans.transform.translation.x
            pose_msg.position.y = trans.transform.translation.y
            pose_msg.position.z = trans.transform.translation.z
            
            # Extract Rotation -> Orientation (Quaternion)
            pose_msg.orientation.x = trans.transform.rotation.x
            pose_msg.orientation.y = trans.transform.rotation.y
            pose_msg.orientation.z = trans.transform.rotation.z
            pose_msg.orientation.w = trans.transform.rotation.w
            
            # Publish it
            self.get_logger().info("Publisng EEF Pose")
            self.publisher_.publish(pose_msg)
            
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"TF Lookup Error: {e}", throttle_duration_sec=2.0)
        except tf2_ros.ConnectivityException as e:
            self.get_logger().warn(f"TF Connectivity Error: {e}", throttle_duration_sec=2.0)
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"TF Extrapolation Error: {e}", throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().error(f"Unexpected TF error: {e}", throttle_duration_sec=2.0)

def main(args=None):
    rclpy.init(args=args)
    node = TfToPosePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
