#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_3d.py \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/a2c_po_agent.yaml \
    -checkpoint_path ./output/sarcoma+a2c+gae0 \
    -dataset sarcoma \
    -exp_name sarcoma+a2c+gae0 \
    -data_path /data/datasets/Sarcoma \
    -lr 5e-4 \
    -val_freq 5 \
    -ep 30 \
    -q_updates_per_step 5 \
    -lazy_penalty -0.1 \
    -wandb_enabled