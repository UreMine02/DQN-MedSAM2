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
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_0_dice0.2137.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_1_dice0.1877.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_2_dice0.2523.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_3_dice0.2420.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_4_dice0.2872.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_5_dice0.3411.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_6_dice0.3355.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_7_dice0.3451.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_8_dice0.3724.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_9_dice0.3874.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_10_dice0.3697.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_11_dice0.3793.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_12_dice0.3618.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_13_dice0.3646.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_14_dice0.3272.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_15_dice0.3759.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_16_dice0.3867.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_17_dice0.3603.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_18_dice0.3163.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_19_dice0.3868.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_20_dice0.3753.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_21_dice0.3706.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_22_dice0.4507.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_23_dice0.3863.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_24_dice0.4114.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_25_dice0.4245.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_26_dice0.3508.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_27_dice0.3977.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_28_dice0.4260.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_29_dice0.4098.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_30_dice0.4513.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_31_dice0.4345.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_32_dice0.4189.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_33_dice0.4308.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_34_dice0.4246.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_35_dice0.3971.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_36_dice0.4079.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_37_dice0.4000.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_38_dice0.4086.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_39_dice0.4072.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_40_dice0.4239.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_41_dice0.4510.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_42_dice0.4306.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_43_dice0.4097.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_44_dice0.4308.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_45_dice0.4784.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_46_dice0.4777.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_47_dice0.3866.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_48_dice0.4614.pth
    /data/rlsam2/msd01/DQN-MedSAM2/output/msd_task08+grpo+icl/2026-01-23-10-45-02/epoch_49_dice0.4807.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task08 \
            -data_path /data/rlsam2/datasets/nii/MSD \
            -num_support $shot
    done
done