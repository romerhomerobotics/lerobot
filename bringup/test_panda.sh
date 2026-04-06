#!/bin/bash
rm -rf inference_outputs

# --- Project Configuration ---
OUTPUT_ROOT="/home/maksimgorki/lerobot/inference_outputs"
POLICY_PATH="/home/maksimgorki/lerobot/outputs/train/franka_panda/08-30-59_act/checkpoints/100000/pretrained_model"
DATASET_REPO="maksimgorki/eval_panda"
TASK_NAME="test_franka_panda"

FPS=30

# --- Execute ---
# Note: We use lerobot-record since it supports evaluation mode when --policy.path is provided.
# It also allows for visualization and dataset recording of the evaluation run.

lerobot-record \
  --robot.type=franka_panda \
  --robot.id=franka_panda_test \
  --display_data=false \
  --dataset.repo_id="${DATASET_REPO}" \
  --dataset.single_task="${TASK_NAME}" \
  --dataset.push_to_hub=false \
  --dataset.root="${OUTPUT_ROOT}" \
  --dataset.fps=${FPS} \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --dataset.episode_time_s=60
