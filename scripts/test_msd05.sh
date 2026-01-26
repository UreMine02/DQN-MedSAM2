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
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_0_dice0.3982.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_1_dice0.4674.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_2_dice0.5436.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_3_dice0.5741.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_4_dice0.5366.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_5_dice0.4406.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_6_dice0.5081.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_7_dice0.5754.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_8_dice0.5067.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_9_dice0.5753.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_10_dice0.5922.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_11_dice0.5285.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_12_dice0.5140.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_13_dice0.5583.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_14_dice0.5917.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_15_dice0.5465.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_16_dice0.6102.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_17_dice0.5695.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_18_dice0.4755.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_19_dice0.5631.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_20_dice0.5840.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_21_dice0.6573.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_22_dice0.6376.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_23_dice0.5969.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_24_dice0.6150.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_25_dice0.6720.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_26_dice0.6915.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_27_dice0.6358.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_28_dice0.5483.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_29_dice0.5937.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_30_dice0.5799.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_31_dice0.6589.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_32_dice0.5299.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_33_dice0.5682.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_34_dice0.6295.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_35_dice0.5695.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_36_dice0.6117.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_37_dice0.5610.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_38_dice0.7043.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_39_dice0.6423.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_40_dice0.6773.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_41_dice0.6757.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_42_dice0.6509.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_43_dice0.5614.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_44_dice0.6592.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_45_dice0.5867.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_46_dice0.6866.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_47_dice0.5762.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_48_dice0.5954.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task05+grpo+icl/2026-01-26-12-31-21/epoch_49_dice0.6215.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 5;
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