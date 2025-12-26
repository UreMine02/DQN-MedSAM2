#!/bin/bash

EXP=sarcoma+grpo+entropy1e-1+num_support10+clip_grad0.1

# python train_3d.py \
#     -exp_name $EXP \
#     -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
#     -rl_config rl_modules/config/grpo_po_agent.yaml \
#     -checkpoint_path ./output/$EXP \
#     -dataset sarcoma \
#     -task Task02 \
#     -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/MSD \
#     -lr 1e-4 \
#     -val_freq 1 \
#     -ep 100 \
#     -q_updates_per_step 2 \
#     -lazy_penalty -0.01 \
#     -invalid_penalty -0.01 \
#     -num_support 10 \
#     -distributed

# CUDA_VISIBLE_DEVICES=0 python train_3d.py \
#     -exp_name sarcoma+grpo+entropy1e-1+num_support10+clip_grad0.1 \
#     -sam_ckpt /mnt/12T/cuong/medsam2-icl-ql/checkpoint/sam2_hiera_tiny.pt \
#     -rl_config rl_modules/config/grpo_po_agent.yaml \
#     -checkpoint_path ./output/sarcoma+grpo+entropy1e-1+num_support10+clip_grad0.1 \
#     -dataset sarcoma \
#     -task Sarcoma \
#     -data_path /mnt/12T/fred/medical_image \
#     -lr 1e-4 \
#     -val_freq 1 \
#     -ep 300 \
#     -q_updates_per_step 2 \
#     -lazy_penalty 0 \
#     -invalid_penalty 0.0 \
#     -num_support 3 \

CUDA_VISIBLE_DEVICES=1 python train_3d.py \
    -exp_name msdTask2+grpo+entropy1e-1+num_support10+clip_grad0.1 \
    -sam_ckpt /mnt/12T/cuong/medsam2-icl-ql/checkpoint/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -checkpoint_path ./output/msdTask2+grpo+entropy1e-1+num_support10+clip_grad0.1 \
    -dataset msd \
    -task Task02_Heart \
    -data_path /mnt/12T/cuong/AAAI/Combined_Dataset/MSD \
    -lr 1e-4 \
    -val_freq 1 \
    -ep 300 \
    -q_updates_per_step 2 \
    -lazy_penalty 0 \
    -invalid_penalty 0.0 \
    -num_support 0 \

