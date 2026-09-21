#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 32 # num cpus
#SBATCH --gres=gpu:4 # num gpus
#SBATCH --mem=200GB # ram
#SBATCH --time=12:00:00 # time
#SBATCH -J msd03 # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd03-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2
# conda init
# conda activate rlsam2

EXP=msd_task03+icl+grpo+global_pool8+group24

python train_3d.py \
    -exp_name $EXP \
    -sam_config sam2_hiera_t \
    -sam_ckpt /data/rlsam2/checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset msd \
    -task "Task03" \
    -data_path /data/rlsam2/datasets/nii/ \
    -lr 2e-4 \
    -val_freq 1 \
    -ep 50 \
    -warmup_ep 0 \
    -stop_sam2_ep -1 \
    -q_updates_per_step 8 \
    -num_support 5 \
    -memory_bank_size 6 \
    -pool_size 8 \
    -pool_stride 4 \
    -recall_every 1 \
    -agent_act_every 1 \
    -rl_group_size 24 \
    -agent_lr_T_max 50 \
    -fold -1 -n_folds 5 \
    -eval_test \
    -seed 0 \
    -wandb_enabled \
    -distributed
    