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

export CUDA_VISIBLE_DEVICES=1

python eval_3d.py \
    -pretrain output/msd_task02+ppo+entropy1e-3/2026-01-05-19-21-35/best.pth \
    -rl_config rl_modules/config/ppo_po_agent.yaml \
    -dataset msd \
    -task Task02 \
    -data_path /data/datasets/MSD \
    -num_support 1
