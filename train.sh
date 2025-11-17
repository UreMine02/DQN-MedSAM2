#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_3d.py \
    -sam_ckpt /data/code/DQN-MedSAM2/checkpoints/sam2_hiera_tiny.pt \
    -checkpoint_path /data/code/DQN-MedSAM2/output/sarcoma+mvalues+no_invalid+freeze+double_dqn \
    -dataset sarcoma \
    -exp_name sarcoma+mvalues+no_invalid+freeze+double_dqn \
    -data_path /data/datasets \
    -lr 4e-4 \
    -val_freq 5 \
    -ep 30 \
    -lazy_penalty -0.1 \
    -wandb_enabled

# CUDA_VISIBLE_DEVICES=1 python train_3d.py \
#     -sam_ckpt /data/code/DQN-MedSAM2/checkpoints/sam2_hiera_tiny.pt \
#     -checkpoint_path /data/code/DQN-MedSAM2/output/msd-task02-mvalues-more_pen \
#     -dataset msd \
#     -exp_name msd-task02-mvalues-more_pen \
#     -task Task02 \
#     -data_path /data/datasets/Combined_Dataset/MSD \
#     -lr 4e-4 \
#     -val_freq 5 \
#     -ep 50 \
#     -lazy_penalty -2 \
#     -invalid_penalty -10 \
#     -wandb_enabled