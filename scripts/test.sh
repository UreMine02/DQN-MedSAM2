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
    # output/msd_task02+grpo+icl+train_max_policy/2026-01-19-12-34-55/best.pth
    # output/msd_task02+no_agent+icl/2026-01-19-14-26-46/epoch_49_dice0.8807.pth
    # output/msd_task02+no_agent+icl+segmentation_loss_only/2026-01-19-20-13-15/best.pth
    # output/msd_task02+grpo+icl+segmentation_loss_only/2026-01-19-21-14-00/best.pth
    # output/sarcoma+grpo+icl/2026-01-21-13-06-51/best.pth
    output/sarcoma+grpo+icl+entrop1e-3/2026-01-21-18-03-09/best.pth
)

export CUDA_VISIBLE_DEVICES=1

for pretrain in ${ckpt[@]};
do
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset sarcoma \
            -task Task02 \
            -data_path /data/datasets/nii/Sarcoma \
            -num_support $shot \
            # -no_agent
            # -vis
    done
done