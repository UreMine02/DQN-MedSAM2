#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 32 # num cpus
#SBATCH --gres=gpu:4 # num gpus
#SBATCH --mem=128GB # ram
#SBATCH --time=10:00 # time
#SBATCH -J msd02 # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd02-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2
conda init
conda activate rlsam2

EXP=msd_task02+ppo+perceiver
export CUDA_VISIBLE_DEVICES=0

python train_3d.py \
    -exp_name $EXP \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/ppo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset msd \
    -task Task02 \
    -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/MSD \
    -lr 1e-4 \
    -val_freq 1 \
    -ep 50 \
    -q_updates_per_step 5 \
    -lazy_penalty -0.1 \
    -invalid_penalty -0.1 \
    -num_support 3 \
    -wandb_enabled