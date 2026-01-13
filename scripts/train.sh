#!/bin/bash

EXP=msd_task03+ppo+entropy1e-3+num_support10+clip_grad0.5+norm0.5

CUDA_VISIBLE_DEVICES=0 python train_3d.py \
    -exp_name $EXP \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/ppo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset msd \
    -task Task03 \
    -data_path /data/datasets/nii/MSD \
    -lr 1e-4 \
    -val_freq 1 \
    -ep 50 \
    -q_updates_per_step 2 \
    -lazy_penalty -0.01 \
    -invalid_penalty -0.01 \
    -val_bg_point 5 \
    -val_fg_point 5 \
    -train_bg_point 5 \
    -train_fg_point 10 \
    -val_prompt_every -1 \
    -train_num_prompted_frame 2 \
    -train_only_point \
    -distributed