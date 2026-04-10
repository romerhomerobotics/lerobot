#!/bin/bash
rm -rf inference_outputs

# --- Project Configuration ---
OUTPUT_ROOT="/home/maksimgorki/lerobot/inference_outputs"
POLICY_PATH="/home/maksimgorki/lerobot/outputs/train/franka_panda/07-14-15_act/checkpoints/150000/pretrained_model"
DATASET_REPO="maksimgorki/eval_panda"
TASK_NAME="test_franka_panda"

FPS=50

# --- Patch Policy Config to match live environment keys ---
# We replace the wrist image first to avoid substring conflicts
if grep -q '"observation.image_wrist"' "${POLICY_PATH}/config.json"; then
  sed -i 's/"observation.image_wrist"/"observation.images.wrist_image"/g' "${POLICY_PATH}/config.json"
  echo "Patched wrist_image key in policy config."
fi

if grep -q '"observation.image"' "${POLICY_PATH}/config.json"; then
  sed -i 's/"observation.image"/"observation.images.image"/g' "${POLICY_PATH}/config.json"
  echo "Patched main image key in policy config."
fi


# TODO: adjust display_data to visualize franka observations and actions

# NOTE: The below policy settings are omitted because they make the control 
# loop a lot slower. Image update rates drop from 10 hz to 2-3 hz.
# --policy.temporal_ensemble_coeff=0.01 
# --policy.n_action_steps=1 

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
