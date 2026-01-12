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

declare -a ckpt=(
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_0_dice0.2659.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_1_dice0.2678.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_2_dice0.1916.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_3_dice0.2618.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_4_dice0.4841.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_5_dice0.3475.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_6_dice0.4538.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_7_dice0.4570.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_8_dice0.4220.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_9_dice0.3946.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_10_dice0.4547.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_11_dice0.4140.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_12_dice0.5181.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_13_dice0.5305.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_14_dice0.4777.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_15_dice0.4556.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_16_dice0.4535.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_17_dice0.4612.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_18_dice0.4349.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_19_dice0.5403.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_20_dice0.5462.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_21_dice0.5029.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_22_dice0.5563.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_23_dice0.5930.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_24_dice0.5301.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_25_dice0.5270.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_26_dice0.5197.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_27_dice0.6023.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_28_dice0.5814.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_29_dice0.6270.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_30_dice0.6036.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_31_dice0.5186.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_32_dice0.5586.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_33_dice0.5764.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_34_dice0.4737.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_35_dice0.5734.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_36_dice0.6140.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_37_dice0.5908.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_38_dice0.6240.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_39_dice0.6214.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_40_dice0.6062.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_41_dice0.5913.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_42_dice0.5975.pth
    output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_43_dice0.6416.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_44_dice0.6069.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_45_dice0.5844.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_46_dice0.6125.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_47_dice0.5998.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_48_dice0.5906.pth
    # output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_49_dice0.5987.pth
)

export CUDA_VISIBLE_DEVICES=0

for pretrain in ${ckpt[@]};
do
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task03 \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot
    done
done