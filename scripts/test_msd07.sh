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
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_0_dice0.3951.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_1_dice0.4439.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_2_dice0.4982.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_3_dice0.4173.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_4_dice0.5324.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_5_dice0.4857.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_6_dice0.5308.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_7_dice0.5675.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_8_dice0.5214.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_9_dice0.5030.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_10_dice0.4581.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_11_dice0.5219.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_12_dice0.5187.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_13_dice0.5480.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_14_dice0.5541.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_15_dice0.5190.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_16_dice0.5903.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_17_dice0.5261.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_18_dice0.5942.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_19_dice0.5650.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_20_dice0.5721.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_21_dice0.5079.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_22_dice0.5666.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_23_dice0.5664.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_24_dice0.5739.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_25_dice0.5819.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_26_dice0.5858.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_27_dice0.5346.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_28_dice0.5921.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_29_dice0.5645.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_30_dice0.6263.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_31_dice0.5330.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_32_dice0.5577.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_33_dice0.5940.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_34_dice0.5652.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_35_dice0.5664.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_36_dice0.6169.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_37_dice0.5735.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_38_dice0.5852.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_39_dice0.5839.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_40_dice0.5815.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_41_dice0.5846.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_42_dice0.5806.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_43_dice0.6429.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_44_dice0.5814.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_45_dice0.5632.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_46_dice0.6004.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_47_dice0.5737.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_48_dice0.5839.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task07+grpo+icl/2026-01-23-10-45-01/epoch_49_dice0.6103.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task07 \
            -data_path /data/rlsam2/datasets/nii/MSD \
            -num_support $shot
    done
done