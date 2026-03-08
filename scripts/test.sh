#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=30:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

declare -a ckpt=(
    # output/msd_task03+grpo+icl/2026-02-28-19-16-27/best.pth
    # output/msd_task04+grpo+icl/2026-02-28-21-24-19/best.pth
    # output/msd_task05+grpo+icl/2026-02-28-17-32-43/best.pth
    # output/msd_task06+grpo+icl/2026-02-28-19-16-27/best.pth
    # output/msd_task07+grpo+icl/2026-02-28-17-32-48/best.pth
    # output/msd_task08+grpo+icl/2026-02-28-19-16-27/best.pth

    # output/msd_task07+no_agent+icl/2026-03-03-16-50-01/best.pth
    # output/msd_task08+no_agent+icl/2026-03-03-17-16-27/best.pth

    output/msd_task05+no_agent+icl/2026-03-04-21-27-17/best.pth
)

declare -a dataset=(
    # Task03
    # Task04
    Task05
    # Task06
    # Task07
    # Task08
)

# export CUDA_VISIBLE_DEVICES=1

for idx in ${!ckpt[@]}
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain ${ckpt[idx]} \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task ${dataset[idx]} \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot \
            -no_agent
    done
done
