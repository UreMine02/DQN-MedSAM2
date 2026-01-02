#!/bin/bash

EXP=msd+grpo+entropy1e-1+num_support10+clip_grad0.1

python train_3d.py \
    -exp_name $EXP \
    -pretrain output/msd+grpo+entropy1e-1+num_support10+clip_grad0.1/2025-12-27-20-17-46/best.pth \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset msd \
    -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/MSD \
    -lr 1e-4 \
    -val_freq 1 \
    -ep 100 \
    -q_updates_per_step 2 \
    -lazy_penalty -0.01 \
    -invalid_penalty -0.01 \
    -num_support 10 \
    -distributed