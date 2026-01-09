#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=128GB # ram
#SBATCH --time=30:00 # time
#SBATCH -J msd03_eval # job name
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd03_eval-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

# export CUDA_VISIBLE_DEVICES=1

for shot in 1 5;
do
    python eval_3d.py \
        -pretrain output/msd_task03+grpo+icl/2026-01-08-05-50-58/best.pth \
        -rl_config rl_modules/config/grpo_po_agent.yaml \
        -dataset msd \
        -task Task03 \
        -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
        -num_support $shot
done
