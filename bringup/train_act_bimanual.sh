#!/bin/bash

lerobot-train \
  --dataset.repo_id="maksimgorki/model_fold_cloth_twice_50hz_with_variance" \
  --dataset.video_backend="pyav" \
  --policy.type=act \
  --output_dir="outputs/train/model_fold_cloth_twice_50hz_with_variance" \
  --job_name="act_so101_test" \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id="maksimgorki/m_fold_cloth_twice_50hz_with_variance" \
  --steps=200000 \
  --dataset.image_transforms.enable=true
