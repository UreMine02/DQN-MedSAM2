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
    output/msd_task02+grpo+icl/2026-02-22-10-36-54/best.pth
    # output/msd_task02+no_agent+icl+freeze/2026-02-19-20-33-41/best.pth
    # output/msd_task02+grpo+icl+agent_only/2026-02-22-15-36-52/best.pth

    # output/msd_task02+grpo+icl/2026-02-26-12-37-46/best.pth
    # output/msd_task02+grpo+icl/2026-02-26-19-17-03/best.pth
    # output/msd_task02+grpo+icl/2026-02-26-17-39-36/best.pth

    # output/sarcoma+grpo+icl/2026-03-03-22-24-08/best.pth
    # output/sarcoma+no_agent+icl/2026-03-03-20-51-26/best.pth

    # output/msd_task09+grpo+icl/2026-03-02-21-58-22/best.pth
    # output/msd_task09+no_agent+icl+augmentation/2026-03-02-17-35-55/best.pth

    # output/msd_task06+grpo+icl/2026-03-03-22-25-28/best.pth
    # output/msd_task06+no_agent+icl/2026-03-03-20-51-01/best.pth
)

export CUDA_VISIBLE_DEVICES=0

for pretrain in ${ckpt[@]};
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "Task02" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            -vis \
            # -no_agent
            # -ablation \
    done
done