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

declare -a ckpt=(
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_20_dice0.4905.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_29_dice0.5645.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_20_dice0.3753.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_39_dice0.9158.pth
)

declare -a tasks=(
    Task01
    Task07
    Task08
    Task09
)
# export CUDA_VISIBLE_DEVICES=0

for index in ${!ckpt[@]};
do
    pretrain=${ckpt[$index]}
    task=${tasks[$index]}
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task $task \
            -data_path /data/rlsam2/datasets/nii/MSD \
            -num_support $shot
            # -no_agent
            # -vis
    done
done