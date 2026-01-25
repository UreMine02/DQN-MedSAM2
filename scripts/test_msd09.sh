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
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_0_dice0.7352.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_1_dice0.6791.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_2_dice0.7684.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_3_dice0.7166.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_4_dice0.8653.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_5_dice0.8454.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_6_dice0.8594.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_7_dice0.8196.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_8_dice0.8791.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_9_dice0.8844.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_10_dice0.8602.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_11_dice0.8894.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_12_dice0.9024.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_13_dice0.9114.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_14_dice0.9139.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_15_dice0.9157.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_16_dice0.8996.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_17_dice0.8957.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_18_dice0.9002.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_19_dice0.9143.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_20_dice0.8982.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_21_dice0.9036.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_22_dice0.9015.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_23_dice0.9048.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_24_dice0.9109.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_25_dice0.9026.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_26_dice0.9106.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_27_dice0.9093.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_28_dice0.9117.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_29_dice0.9175.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_30_dice0.8982.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_31_dice0.9173.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_32_dice0.9088.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_33_dice0.9069.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_34_dice0.9160.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_35_dice0.9081.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_36_dice0.9102.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_37_dice0.9100.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_38_dice0.9119.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_39_dice0.9158.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_40_dice0.9169.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_41_dice0.9236.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_42_dice0.9193.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_43_dice0.9149.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_44_dice0.9033.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_45_dice0.9218.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_46_dice0.9163.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_47_dice0.9219.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_48_dice0.9251.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task09+grpo+icl/2026-01-23-10-47-46/epoch_49_dice0.9211.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task09 \
            -data_path /data/rlsam2/datasets/nii/MSD \
            -num_support $shot
    done
done