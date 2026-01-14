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

declare -a ckpt=(
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_12_dice0.6191.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_19_dice0.5596.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_30_dice0.6005.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_37_dice0.5702.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_39_dice0.6487.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_42_dice0.5336.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_43_dice0.5174.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_44_dice0.5759.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_46_dice0.6254.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset btcv \
            -data_path /data/rlsam2/datasets/nii/BTCV \
            -num_support $shot
    done
done