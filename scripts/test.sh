#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=64GB # ram
#SBATCH --time=30:00 # time
#SBATCH -J msd02_eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd02_eval-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

declare -a ckpt=(
    output/msd_task02+ppo+icl/2026-02-16-20-19-11/best.pth
)

# export CUDA_VISIBLE_DEVICES=1

for pretrain in ${ckpt[@]}
do
    for shot in 2;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/ppo_po_agent.yaml \
            -dataset msd \
            -task Task02 \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot
    done
done