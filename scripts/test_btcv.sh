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
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_0_dice0.3277.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_1_dice0.3293.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_2_dice0.2938.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_3_dice0.3488.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_4_dice0.3853.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_5_dice0.3041.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_6_dice0.3617.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_7_dice0.4214.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_8_dice0.3618.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_9_dice0.3195.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_10_dice0.3651.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_11_dice0.3877.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_12_dice0.3421.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_13_dice0.4027.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_14_dice0.4046.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_15_dice0.3893.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_16_dice0.4401.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_17_dice0.4204.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_18_dice0.4293.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_19_dice0.3945.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_20_dice0.3700.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_21_dice0.4270.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_22_dice0.4471.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_23_dice0.4139.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_24_dice0.4200.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_25_dice0.4550.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_26_dice0.4529.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_27_dice0.3817.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_28_dice0.4584.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_29_dice0.4384.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_30_dice0.4264.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_31_dice0.4233.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_32_dice0.4825.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_33_dice0.4270.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_34_dice0.4384.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_35_dice0.4762.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_36_dice0.4109.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_37_dice0.4264.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_38_dice0.4584.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_39_dice0.4630.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_40_dice0.4572.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_41_dice0.4352.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_42_dice0.4411.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_43_dice0.4810.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_44_dice0.4700.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_45_dice0.4582.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_46_dice0.4539.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_47_dice0.4628.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_48_dice0.4528.pth
    output/btcv+grpo+icl/2026-01-26-21-18-07/epoch_49_dice0.4972.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset btcv \
            -data_path /data/rlsam2/datasets/nii/BTCV \
            -num_support $shot
    done
done