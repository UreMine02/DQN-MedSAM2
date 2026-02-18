#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 32 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=200GB # ram
#SBATCH --time=24:00:00 # time
#SBATCH -J msd02 # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd02+adapter-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2
conda init
conda activate rlsam2

EXP=msd_task02+no_agent+icl

python train_3d.py \
    -exp_name $EXP \
    -sam_config sam2_hiera_l.yaml \
    -sam_ckpt /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/checkpoints/sam2_hiera_large.pt \
    -rl_config rl_modules/config/ppo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset msd \
    -task Task02 \
    -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
    -lr 1e-4 \
    -val_freq 1 \
    -ep 200 \
    -q_updates_per_step 2 \
    -lazy_penalty -0.01 \
    -invalid_penalty -0.01 \
    -num_support 5 \
    -no_agent