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
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_0_dice0.2672.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_1_dice0.4155.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_2_dice0.5166.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_3_dice0.5055.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_4_dice0.5374.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_5_dice0.4379.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_6_dice0.4816.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_7_dice0.4566.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_8_dice0.5189.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_9_dice0.5850.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_10_dice0.5693.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_11_dice0.5605.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_12_dice0.5999.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_13_dice0.5738.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_14_dice0.5901.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_15_dice0.6269.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_16_dice0.6204.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_17_dice0.6879.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_18_dice0.5837.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_19_dice0.6231.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_20_dice0.5965.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_21_dice0.5409.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_22_dice0.5839.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_23_dice0.6041.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_24_dice0.6417.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_25_dice0.5105.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_26_dice0.6117.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_27_dice0.6589.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_28_dice0.6063.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_29_dice0.6078.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_30_dice0.6786.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_31_dice0.6570.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_32_dice0.5632.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_33_dice0.6141.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_34_dice0.6814.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_35_dice0.5719.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_36_dice0.6353.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_37_dice0.6086.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_38_dice0.5964.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_39_dice0.6743.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_40_dice0.6127.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_41_dice0.6549.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_42_dice0.6517.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_43_dice0.6642.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_44_dice0.6088.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_45_dice0.6788.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_46_dice0.6955.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_47_dice0.6753.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_48_dice0.6397.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-08-44-18/epoch_49_dice0.6280.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task05 \
            -data_path /data/rlsam2/datasets/nii/MSD \
            -num_support $shot
    done
done