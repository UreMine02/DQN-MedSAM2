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
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_0_dice0.3206.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_1_dice0.2138.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_2_dice0.3662.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_3_dice0.5409.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_4_dice0.5459.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_5_dice0.5406.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_6_dice0.5710.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_7_dice0.6581.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_8_dice0.6713.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_9_dice0.6628.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_10_dice0.6821.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_11_dice0.6346.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_12_dice0.6815.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_13_dice0.6615.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_14_dice0.6593.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_15_dice0.7060.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_16_dice0.7154.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_17_dice0.6806.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_18_dice0.7096.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_19_dice0.5853.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_20_dice0.6885.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_21_dice0.6771.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_22_dice0.7070.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_23_dice0.7292.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_24_dice0.7028.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_25_dice0.7519.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_26_dice0.6726.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_27_dice0.6533.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_28_dice0.6737.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_29_dice0.6720.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_30_dice0.6719.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_31_dice0.6277.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_32_dice0.6804.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_33_dice0.6946.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_34_dice0.6494.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_35_dice0.6296.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_36_dice0.7029.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_37_dice0.7633.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_38_dice0.7355.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_39_dice0.7515.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_40_dice0.7124.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_41_dice0.7303.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_42_dice0.6579.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_43_dice0.7068.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_44_dice0.6480.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_45_dice0.6038.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_46_dice0.6520.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_47_dice0.7228.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_48_dice0.7018.pth
    /data/rlsam2/msd05/DQN-MedSAM2/output/msd_task06+grpo+icl/2026-01-26-08-44-27/epoch_49_dice0.6844.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task06 \
            -data_path /data/rlsam2/datasets/nii/MSD \
            -num_support $shot
    done
done