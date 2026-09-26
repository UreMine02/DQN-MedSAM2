#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 32 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=200GB # ram
#SBATCH --time=2:00:00 # time
#SBATCH -J msd02 # job name
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd02-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2
# conda init
# conda activate rlsam2

export CUDA_VISIBLE_DEVICES=1

# for SEED in 0;
# do
#     for FOLD in 0 1 2 3 4;
#     do
EXP=msd_task02+icl+global_pool16_stride #+fold${FOLD}+seed${SEED}
python train_3d.py \
    -exp_name $EXP \
    -sam_config sam2_hiera_t \
    -sam_ckpt ./checkpoints/sam2_hiera_tiny.pt \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -checkpoint_path ./output/$EXP \
    -dataset msd \
    -task Task02 \
    -data_path /data/datasets/nii/ \
    -lr 2e-4 -val_freq 30 -ep 30 -warmup_ep 0 -stop_sam2_ep -1 \
    -num_support 5 -memory_bank_size 6 \
    -pool_size 16 -pool_stride 4 -pool_policy stride -pool_novelty_iou 0.05 -recall_every 1 \
    -agent_act_every 1 -rl_group_size 12 -agent_lr_T_max 100 -q_updates_per_step 4 \
    -fold -1 -n_folds 5 \
    -eval_test \
    -seed 0 \
    -wandb_enabled
#     done
# done