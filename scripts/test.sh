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
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_0_dice0.4966.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_1_dice0.5437.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_2_dice0.5349.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_3_dice0.5101.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_4_dice0.5101.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_5_dice0.5352.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_6_dice0.5678.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_7_dice0.5562.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_8_dice0.5187.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_9_dice0.4931.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_10_dice0.6043.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_11_dice0.5031.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_12_dice0.6191.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_13_dice0.5960.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_14_dice0.5499.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_15_dice0.5804.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_16_dice0.6129.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_17_dice0.5901.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_18_dice0.5557.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_19_dice0.5596.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_20_dice0.5933.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_21_dice0.6118.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_22_dice0.5609.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_23_dice0.5846.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_24_dice0.6205.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_25_dice0.6011.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_26_dice0.5762.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_27_dice0.5652.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_28_dice0.5584.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_29_dice0.6145.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_30_dice0.6005.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_31_dice0.6469.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_32_dice0.5656.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_33_dice0.6428.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_34_dice0.6128.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_35_dice0.6090.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_36_dice0.6283.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_37_dice0.5702.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_38_dice0.6189.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_39_dice0.6487.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_40_dice0.5928.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_41_dice0.6006.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_42_dice0.5336.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_43_dice0.5174.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_44_dice0.5759.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_45_dice0.5726.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_46_dice0.6254.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_47_dice0.6005.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_48_dice0.6316.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-13-12-53-09/epoch_49_dice0.6083.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset btcv \
            -data_path /data/rlsam2/datasets/nii/BTCV \
            -num_support $shot
    done
done