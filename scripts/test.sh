#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=128GB # ram
#SBATCH --time=30:00 # time
#SBATCH -J btcv_eval # job name
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/btcv_eval-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

python eval_3d.py \
    -pretrain output/btcv+grpo+entropy1e-1+num_support10+clip_grad0.1/2025-12-31-09-20-12/best.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -dataset btcv \
    -task Task03 \
    -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/BTCV \
    -num_support 16
