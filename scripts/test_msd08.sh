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

ckpt=(
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_20_dice0.3753.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task08 \
            -data_path /data/rlsam2/datasets/nii/MSD \
            -num_support $shot
    done
done