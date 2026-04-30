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

declare -a ckpts=(
    # Sarcoma
    # output/sarcoma+icl+grpo+long_horizon+no_augment/2026-04-29-14-22-57/best.pth
    # output/sarcoma+icl+no_agent+long_horizon+no_augment/2026-04-10-19-47-31/best.pth

    # MSD Heart
    # output/msd_task02+icl+no_agent+long_horizon+no_augment/2026-04-15-19-42-29/best.pth
    # output/msd_task02+icl+grpo+lazy_pen1.0+long_horizon+flip_augment/2026-04-23-16-01-08/best.pth
    output/msd_task02+icl+grpo+long_horizon+no_augment/2026-04-30-18-07-03/best.pth
    
    # MSD Hippocampus
    # output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_24_dice0.7822.pth
    # output/msd_task04+no_agent+icl+long_horizon+no_augment/2026-04-28-20-05-44/best.pth

    # MSD Prostate
    # output/msd_task05+grpo+long_horizon+no_augment/2026-04-26-11-35-38/best.pth
    # output/msd_task05+no_agent+long_horizon+no_augment/2026-04-13-18-04-27/best.pth

    # MSD Lung
    # output/msd_task06+grpo+long_horizon+no_augment/2026-04-26-16-12-37/best.pth
    # output/msd_task06+icl+no_agent+long_horizon+no_augment/2026-04-28-17-12-14/best.pth

    # MSD Pancreas
    # output/msd_task07+no_agent+icl+long_horizon+no_augment/2026-04-28-20-07-36/best.pth
    # output/msd_task07+grpo+icl+long_horizon+no_augment/2026-04-26-17-18-17/epoch_21_dice0.5748.pth

    # MSD Hepatic
    # output/msd_task08+no_agent+icl+long_horizon+no_augment/2026-04-28-20-20-31/best.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/best.pth

    # MSD Spleen
    # output/msd_task09+icl+grpo+long_horizon+no_augment/2026-04-25-19-48-27/best.pth
    # output/msd_task09+icl+no_agent+long_horizon+no_augment/2026-04-11-12-19-00/best.pth

    # MSD Colon
    # output/msd_task10+icl+grpo+long_horizon+no_augment/2026-04-25-13-09-26/best.pth
    # output/msd_task10+icl+no_agent+long_horizon+no_augment/2026-04-12-15-52-38/best.pth
)

declare -a datasets=(
    msd
)

declare -a tasks=(
    Task02
)

declare -a no_agents=(
    agent
)

export CUDA_VISIBLE_DEVICES=0

for idx in ${!ckpts[@]}
do
    # echo ${ckpts[idx]}
    # echo ${tasks[idx]}
    # echo ${no_agents[idx]}
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain "${ckpts[idx]}" \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset "${datasets[idx]}" \
            -task "${tasks[idx]}" \
            -data_path /data/datasets/nii/ \
            -num_support $shot \
            -memory_bank_size 6 \
            -ablation \
            -vis \
            $( [ ${no_agents[idx]} = "no_agent" ] && echo "-no_agent" )
    done
done
