#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=128GB # ram
#SBATCH --time=30:00 # time
#SBATCH -J msd03_eval # job name
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd03_eval-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

# export CUDA_VISIBLE_DEVICES=1

declare -a ckpt=(
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_0_dice0.3180.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_1_dice0.2247.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_2_dice0.3728.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_3_dice0.3836.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_4_dice0.3207.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_5_dice0.2385.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_6_dice0.4044.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_7_dice0.3056.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_8_dice0.2544.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_9_dice0.2235.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_10_dice0.3212.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_11_dice0.2315.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_12_dice0.2367.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_13_dice0.2762.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_14_dice0.2915.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_15_dice0.2756.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_16_dice0.2481.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_17_dice0.2889.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_18_dice0.2859.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_19_dice0.2581.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_20_dice0.3057.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_21_dice0.3032.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_22_dice0.3163.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_23_dice0.3255.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_24_dice0.2939.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_25_dice0.3109.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_26_dice0.3174.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_27_dice0.2950.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_28_dice0.2675.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_29_dice0.2806.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_30_dice0.3362.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_31_dice0.3070.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_32_dice0.2906.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_33_dice0.2968.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_34_dice0.3036.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_35_dice0.3049.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_36_dice0.2562.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_37_dice0.3330.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_38_dice0.3135.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_39_dice0.2898.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_40_dice0.2815.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_41_dice0.2624.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_42_dice0.3068.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_43_dice0.3101.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_44_dice0.2857.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_45_dice0.3332.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_46_dice0.3227.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_47_dice0.2673.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_48_dice0.3424.pth
    /data/rlsam2/DQN-MedSAM2/output/msd_task01+grpo+icl/2026-01-14-12-32-00/epoch_49_dice0.3400.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task01 \
            -data_path /data/rlsam2/datasets/nii/MSD \
            -num_support $shot
    done
done