#!/bin/bash

lerobot-train \
  --dataset.repo_id="maksimgorki/homerobotics-lerobot-50hz" \
  --dataset.video_backend="pyav" \
  --policy.type=act \
  --output_dir="outputs/train/homerobotics_fold_cloth_50hz" \
  --job_name="act_so101_test" \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id="maksimgorki/fold_cloth_50hz"
