#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=24:00:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-btcv-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

declare -a ckpt=(
    output/msd_task02+icl+no_agent+topk_mem/2026-09-17-12-41-48/epoch_29_dice0.8437.pth
    output/msd_task03+icl+no_agent+long_horizon+no_augment/2026-04-30-05-44-46/epoch_30_dice0.6514.pth
    output/msd_task04+no_agent+icl+long_horizon+no_augment/2026-04-28-20-05-44/epoch_30_dice0.7700.pth
    output/msd_task05+no_agent+long_horizon+no_augment/2026-04-13-18-04-27/epoch_30_dice0.6146.pth
    output/msd_task06+icl+no_agent+long_horizon+no_augment/2026-04-28-17-12-14/epoch_30_dice0.5449.pth
    output/msd_task07+no_agent+icl+long_horizon+no_augment/2026-04-28-20-07-36/epoch_30_dice0.5643.pth
    output/msd_task08+no_agent+icl+long_horizon+no_augment/2026-04-28-20-20-31/epoch_30_dice0.5185.pth
    output/msd_task09+icl+no_agent+long_horizon+no_augment/2026-04-11-12-19-00/epoch_30_dice0.9007.pth
    output/msd_task10+icl+no_agent+long_horizon+no_augment/2026-04-12-15-52-38/epoch_30_dice0.5119.pth
    output/sarcoma+icl+no_agent+long_horizon+no_augment/2026-04-10-19-47-31/epoch_30_dice0.7361.pth
    output/btcv+icl+no_agent+long_horizon+no_augment/2026-04-30-00-40-30/epoch_30_dice0.6800.pth
)

export CUDA_VISIBLE_DEVICES=0

for idx in ${!ckpt[@]}
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain ${ckpt[idx]} \
            -rl_config rl_modules/config/ppo_po_agent.yaml \
            -dataset msd \
            -task "Task02" \
            -data_path /data/datasets/nii/ \
            -num_support $shot \
            -memory_bank_size 6 \
            -memory_select topk \
            # -no_agent \
            # -random_drop
    done
done
