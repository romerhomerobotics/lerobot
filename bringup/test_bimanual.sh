#!/bin/bash
rm -rf inference_outputs

# --- Robot Configuration ---
# Hardware Ports
RIGHT_PORT="/dev/ttyACM2"
LEFT_PORT="/dev/ttyACM3"

# Camera Indices (OpenCV /dev/videoX)
CAM_GRIPPER_RIGHT=4
CAM_GRIPPER_LEFT=0
CAM_MIDDLE=2

# --- Project Configuration ---
OUTPUT_ROOT="/home/kovan/lerobot/inference_outputs"
POLICY_PATH="/home/kovan/lerobot/outputs/train/homerobotics_fold_cloth_50hz/checkpoints/100000/pretrained_model"
DATASET_REPO="maksimgorki/eval_homerobotics_2"
TASK_NAME="deneme"

# --- Execute ---
lerobot-record \
  --robot.type=bi_so_follower \
  --robot.id=homerobotics_bimanual_follower \
  --robot.right_arm_config.port="${RIGHT_PORT}" \
  --robot.right_arm_config.cameras="{
    \"gripper_right\": {
      \"type\": \"opencv\",
      \"index_or_path\": \"/dev/video${CAM_GRIPPER_RIGHT}\",
      \"width\": 640, \"height\": 480, \"fps\": 30, \"fourcc\": \"MJPG\", \"backend\": \"V4L2\"
    }
  }" \
  --robot.left_arm_config.port="${LEFT_PORT}" \
  --robot.left_arm_config.cameras="{
    \"gripper_left\": {
      \"type\": \"opencv\",
      \"index_or_path\": \"/dev/video${CAM_GRIPPER_LEFT}\",
      \"width\": 640, \"height\": 480, \"fps\": 30, \"fourcc\": \"MJPG\", \"backend\": \"V4L2\"
    },
    \"middle\": {
      \"type\": \"opencv\",
      \"index_or_path\": \"/dev/video${CAM_MIDDLE}\",
      \"width\": 640, \"height\": 480, \"fps\": 30, \"fourcc\": \"MJPG\", \"backend\": \"V4L2\"
    }
  }" \
  --display_data=false \
  --dataset.repo_id="${DATASET_REPO}" \
  --dataset.single_task="${TASK_NAME}" \
  --dataset.push_to_hub=false \
  --dataset.root="${OUTPUT_ROOT}" \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --policy.n_action_steps=90
