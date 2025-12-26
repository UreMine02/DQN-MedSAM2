#!/bin/bash

EXP=msd_task03+grpo+entropy1e-1+num_support10+clip_grad0.1

python train_3d.py \
    -exp_name $EXP \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset msd \
    -task Task03 \
    -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/MSD \
    -lr 1e-4 \
    -val_freq 1 \
    -ep 200 \
    -q_updates_per_step 2 \
    -lazy_penalty -0.01 \
    -invalid_penalty -0.01 \
    -num_support 10 \
    -distributed

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

