#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=128GB # ram
#SBATCH --time=30:00 # time
#SBATCH -J msd03_eval # job name
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd03_eval-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

export CUDA_VISIBLE_DEVICES=0

python eval_3d.py \
    -pretrain output/sarcoma+grpo+no_icl/2025-12-26-19-24-41/best.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -dataset sarcoma \
    -task Task02 \
    -data_path /data/datasets/Sarcoma \
    -val_fg_point 1 \
    -val_bg_point 0 \
    -val_prompt_every 10
