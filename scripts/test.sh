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
    # output/dpc/msd_task03+no_agent+icl/epoch_5_dice0.1750.pth
    # output/msd_task03+no_agent+icl/2026-01-27-22-07-05/epoch_4_dice0.6530.pth

    # output/msd_task03+no_agent+icl/2026-01-23-10-03-30/epoch_5_dice0.1750.pth
    # ./checkpoints/sam2_hiera_tiny.pt
    
    # /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_20_dice0.4905.pth
    # /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_29_dice0.5645.pth
    # /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_20_dice0.3753.pth
    # /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_39_dice0.9158.pth

    # output/dpc/msd_task07+grpo+icl/2026-01-28-00-18-39/best.pth
    # output/epoch_5_dice0.5446.pth
    # output/msd_task02/2026-01-28-04-47-14/best.pth
    # output/dpc/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_7_dice0.5675.pth

    # output/sarcoma+grpo+icl+entrop1e-3/2026-01-21-18-03-09/best.pth
    # output/sarcoma+no_agent+icl+fullfinetuning/2026-01-22-09-29-29/epoch_2_dice0.7036.pth

    # output/msd_task02+no_agent+icl+correct_iou/2026-01-19-19-23-17/best.pth
    # output/msd_task02+grpo+icl+agent_only/2026-01-28-14-50-10/epoch_9_dice0.9099.pth

    # output/msd_task07+phase_1/2026-01-29-00-44-52/epoch_7_dice0.7497.pth
    output/dpc/msd_task03+grpo+icl/2026-01-23-15-39-00/best.pth
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
            -task Task03 \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot 
    done
done