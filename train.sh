#!/bin/bash

EXP=msd_task02+grpo+no_icl

CUDA_VISIBLE_DEVICES=1 python train_3d.py \
    -exp_name $EXP \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset msd \
    -task Task02 \
    -data_path /data/datasets/MSD \
    -lr 1e-4 \
    -val_freq 1 \
    -ep 300 \
    -q_updates_per_step 2 \
    -lazy_penalty 0 \
    -invalid_penalty 0.0 \
    -num_support 0 \