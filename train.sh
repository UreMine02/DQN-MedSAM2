#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_3d.py \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/ppo_po_agent.yaml \
    -checkpoint_path ./output/msd_task02+ppo+gae1.0 \
    -dataset msd \
    -task Task02 \
    -exp_name msd_task02+ppo+gae1.0 \
    -data_path /data/datasets/Combined/MSD \
    -lr 5e-4 \
    -val_freq 1 \
    -ep 50 \
    -q_updates_per_step 5 \
    -lazy_penalty 0 \
    -wandb_enabled