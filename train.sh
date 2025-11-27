#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_3d.py \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/c51_q_agent.yaml \
    -checkpoint_path ./output/sarcoma+mvalues+no_invalid+freeze+obj_ptr+c51 \
    -dataset msd \
    -task Task02 \
    -exp_name msd_task02+mvalues+no_invalid+freeze+obj_ptr+c51 \
    -data_path /data/datasets/Combined_Dataset/MSD \
    -lr 5e-4 \
    -val_freq 5 \
    -ep 50 \
    -q_updates_per_step 2 \
    -lazy_penalty -0.1 \
    -wandb_enabled