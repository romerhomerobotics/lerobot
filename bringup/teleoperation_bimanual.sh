#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================================
# 1. FOLLOWER (ROBOT) CONFIGURATION
# ==========================================
ROBOT_TYPE="bi_so_follower"
ROBOT_ID="homerobotics_bimanual_follower"

# Follower Left Arm
ROBOT_LEFT_PORT="/dev/ttyACM3"
CAM_LEFT_NAME="gripper_left"
CAM_LEFT_IDX=4
CAM_MID_NAME="middle"
CAM_MID_IDX=2

# Follower Right Arm
ROBOT_RIGHT_PORT="/dev/ttyACM2"
CAM_RIGHT_NAME="gripper_right"
CAM_RIGHT_IDX=0

# Global Camera Settings
CAM_WIDTH=640
CAM_HEIGHT=480
CAM_FPS=30
CAM_FOURCC="MJPG"
CAM_TYPE="opencv"

# ==========================================
# 2. LEADER (TELEOP) CONFIGURATION
# ==========================================
TELEOP_TYPE="bi_so_leader"
TELEOP_ID="homerobotics_bimanual_leader"
TELEOP_LEFT_PORT="/dev/ttyACM1"
TELEOP_RIGHT_PORT="/dev/ttyACM0"

# ==========================================
# 3. BUILD JSON CAMERA PAYLOADS (For Live Viewing)
# ==========================================

LEFT_CAM_JSON=$(cat <<EOF
{
  "${CAM_LEFT_NAME}": {
    "type": "${CAM_TYPE}",
    "index_or_path": ${CAM_LEFT_IDX},
    "width": ${CAM_WIDTH},
    "height": ${CAM_HEIGHT},
    "fps": ${CAM_FPS},
    "fourcc": "${CAM_FOURCC}"
  },
  "${CAM_MID_NAME}": {
    "type": "${CAM_TYPE}",
    "index_or_path": ${CAM_MID_IDX},
    "width": ${CAM_WIDTH},
    "height": ${CAM_HEIGHT},
    "fps": ${CAM_FPS},
    "fourcc": "${CAM_FOURCC}"
  }
}
EOF
)

RIGHT_CAM_JSON=$(cat <<EOF
{
  "${CAM_RIGHT_NAME}": {
    "type": "${CAM_TYPE}",
    "index_or_path": ${CAM_RIGHT_IDX},
    "width": ${CAM_WIDTH},
    "height": ${CAM_HEIGHT},
    "fps": ${CAM_FPS},
    "fourcc": "${CAM_FOURCC}"
  }
}
EOF
)

# ==========================================
# 4. EXECUTE TELEOP COMMAND
# ==========================================
echo "Starting LeRobot Teleoperation (Live View, No Recording)..."

lerobot-teleoperate \
  --robot.type="${ROBOT_TYPE}" \
  --robot.id="${ROBOT_ID}" \
  --robot.left_arm_config.port="${ROBOT_LEFT_PORT}" \
  --robot.left_arm_config.cameras="${LEFT_CAM_JSON}" \
  --robot.right_arm_config.port="${ROBOT_RIGHT_PORT}" \
  --robot.right_arm_config.cameras="${RIGHT_CAM_JSON}" \
  --teleop.type="${TELEOP_TYPE}" \
  --teleop.id="${TELEOP_ID}" \
  --teleop.left_arm_config.port="${TELEOP_LEFT_PORT}" \
  --teleop.right_arm_config.port="${TELEOP_RIGHT_PORT}" \
  --display_data="true"
