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
    # MSD Heart
    # output/msd_task02+icl+ppo+skip_penalty+long_horizon+no_augment/2026-04-08-08-07-00/best.pth
    # output/msd_task02+icl+ppo+no_agent+long_horizon+no_augment/2026-04-10-14-46-14/best.pth
    # output/msd_task02+no_agent+icl+no_augment/2026-04-03-19-32-08/best.pth
    # output/msd_task02+icl+ppo+long_horizon+no_augment/2026-04-17-16-43-56/best.pth
    # output/msd_task02+icl+ppo+long_horizon+no_augment/2026-04-16-10-46-24/best.pth
    # output/msd_task02+icl+grpo+lazy_pen1.0+long_horizon+flip_augment/2026-04-23-16-01-08/best.pth

    # Sarcoma
    # output/sarcoma+icl+ppo+long_horizon+no_augment/2026-04-10-19-46-46/best.pth
    # output/sarcoma+icl+no_agent+long_horizon+no_augment/2026-04-10-19-47-31/best.pth

    # MSD Colon
    output/msd_task10+icl+grpo+long_horizon+no_augment/2026-04-25-13-09-26/best.pth
    # output/msd_task10+icl+no_agent+long_horizon+no_augment/2026-04-12-15-52-38/best.pth
    # output/msd_task10+icl+ppo+long_horizon+no_augment/2026-04-17-18-39-30/best.pth

    # MSD Spleen
    # output/msd_task09+icl+ppo+long_horizon+no_augment/2026-04-11-12-20-13/best.pth
    # output/msd_task09+icl+no_agent+long_horizon+no_augment/2026-04-11-12-19-00/best.pth
    
    # MSD Prostate
    # output/msd_task05+no_agent+long_horizon+no_augment/2026-04-13-18-04-27/best.pth
    # output/msd_task05+ppo+long_horizon+no_augment/2026-04-13-14-13-19/best.pth
)

export CUDA_VISIBLE_DEVICES=0

for idx in ${!ckpt[@]}
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain ${ckpt[idx]} \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "Task10" \
            -data_path /data/datasets/nii/ \
            -num_support $shot \
            -memory_bank_size 6 \
            -ablation \
            # -vis
            # -no_agent
    done
done
