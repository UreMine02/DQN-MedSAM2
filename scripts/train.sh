#!/bin/bash

EXP=msd_task03+grpo+entropy1e-3+num_support10+clip_grad0.5+norm0.5

CUDA_VISIBLE_DEVICES=0 python train_3d.py \
    -exp_name $EXP \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset msd \
    -task Task03 \
    -data_path /data/datasets/MSD \
    -lr 1e-4 \
    -val_freq 1 \
    -ep 100 \
    -q_updates_per_step 2 \
    -lazy_penalty -0.01 \
    -invalid_penalty -0.01 \
    -num_support 10 \
    -wandb_enabled