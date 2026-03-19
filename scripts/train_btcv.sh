#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 32 # num cpus
#SBATCH --gres=gpu:4 # num gpus
#SBATCH --mem=200GB # ram
#SBATCH --time=2-00:00:00 # time
#SBATCH -J btcv # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/btcv-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2
conda init
conda activate rlsam2

EXP=btcv+grpo+icl+tw_gating+semantic_filtering

python train_3d.py \
    -exp_name $EXP \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset btcv \
    -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/BTCV \
    -lr 2e-4 \
    -val_freq 1 \
    -ep 100 \
    -q_updates_per_step 1 \
    -lazy_penalty 0.0 \
    -invalid_penalty -0.01 \
    -num_support 5 \
    -distributed
