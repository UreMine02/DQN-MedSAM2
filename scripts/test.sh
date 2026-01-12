#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=128GB # ram
#SBATCH --time=30:00 # time
#SBATCH -J btcv_eval # job name
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/btcv_eval-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

export CUDA_VISIBLE_DEVICES=0

python eval_3d.py \
    -pretrain output/sarcoma+grpo+prompt/2026-01-11-17-49-42/best.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -dataset sarcoma \
    -task '' \
    -data_path /data/datasets/nii/Sarcoma \
    -val_bg_point 5 \
    -val_fg_point 5 \
    -val_prompt_every -1 \
    -vis
