#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=128GB # ram
#SBATCH --time=30:00 # time
#SBATCH -J msd03_eval # job name
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd03_eval-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

# export CUDA_VISIBLE_DEVICES=1

declare -a ckpt=(
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_0_dice0.4846.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_1_dice0.5395.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_2_dice0.5766.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_3_dice0.5603.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_4_dice0.6122.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_5_dice0.6034.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_6_dice0.5952.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_7_dice0.6489.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_8_dice0.5931.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_9_dice0.5889.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_10_dice0.5930.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_11_dice0.6132.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_12_dice0.6399.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_13_dice0.6300.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_14_dice0.6355.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_15_dice0.6401.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_16_dice0.6267.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_17_dice0.6556.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_18_dice0.6192.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_19_dice0.6652.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_20_dice0.6261.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_21_dice0.6679.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_22_dice0.5915.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_23_dice0.6362.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_24_dice0.6482.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_25_dice0.6463.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_26_dice0.5865.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_27_dice0.5869.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_28_dice0.5523.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_29_dice0.5753.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_30_dice0.6000.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_31_dice0.6120.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_32_dice0.6347.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_33_dice0.5704.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_34_dice0.6730.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_35_dice0.7041.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_36_dice0.6402.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_37_dice0.6281.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_38_dice0.6522.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_39_dice0.6442.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_40_dice0.6305.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_41_dice0.6491.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_42_dice0.6648.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_43_dice0.6746.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_44_dice0.6433.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_45_dice0.6159.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_46_dice0.6098.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_47_dice0.6520.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_48_dice0.6708.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-12-21-05-49/epoch_49_dice0.6502.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset btcv \
            -data_path /data/rlsam2/datasets/nii/BTCV \
            -num_support $shot
    done
done