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
    # output/msd_task10+grpo+icl/2026-02-24-15-34-45/best.pth
    output/msd_task10+grpo+icl/2026-02-24-15-34-45/epoch_48_dice0.5752.pth
)

export CUDA_VISIBLE_DEVICES=0

for pretrain in ${ckpt[@]}
do
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task10\
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            -no_agent \
            # -vis
    done
done