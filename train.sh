#!/bin/bash

EXP=test

CUDA_VISIBLE_DEVICES=1 python train_3d.py \
    -exp_name $EXP \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/ppo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset msd \
    -task Task02 \
    -data_path /data/datasets/MSD \
    -lr 2e-4 \
    -val_freq 1 \
    -ep 1 \
    -q_updates_per_step 2 \
    -lazy_penalty 0 \
    -invalid_penalty 0 \
    -num_support 3 \
    # -wandb_enabled

# CUDA_VISIBLE_DEVICES=1 python train_3d.py \
#     -exp_name $EXP \
#     -pretrain output/test/2025-12-10-11-43-43/epoch_0_dice0.6265.pth \
#     -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
#     -rl_config rl_modules/config/ppo_po_agent.yaml \
#     -checkpoint_path ./output/$EXP \
#     -dataset msd \
#     -task Task02 \
#     -data_path /data/datasets/MSD \
#     -lr 2e-4 \
#     -val_freq 1 \
#     -ep 1 \
#     -q_updates_per_step 0 \
#     -lazy_penalty 0 \
#     -invalid_penalty 0 \
#     -num_support 3 \

# CUDA_VISIBLE_DEVICES=0 python train_3d.py \
#     -exp_name sarcoma+ppo+normalized_gae0.99+entropy1e-1+num_support3 \
#     -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
#     -rl_config rl_modules/config/ppo_po_agent.yaml \
#     -checkpoint_path ./output/sarcoma+ppo+normalized_gae0.99+entropy1e-1+num_support3 \
#     -dataset sarcoma \
#     -task Sarcoma \
#     -data_path /data/datasets/Sarcoma \
#     -lr 1e-4 \
#     -val_freq 1 \
#     -ep 300 \
#     -q_updates_per_step 2 \
#     -lazy_penalty 0 \
#     -invalid_penalty 0.0 \
#     -num_support 3 \
#     -wandb_enabled

