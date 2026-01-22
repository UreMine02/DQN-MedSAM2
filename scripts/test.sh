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
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_0_dice0.3722.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_1_dice0.3936.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_2_dice0.4665.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_3_dice0.4505.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_4_dice0.4693.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_5_dice0.4629.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_6_dice0.4830.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_7_dice0.4636.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_8_dice0.4932.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_9_dice0.4660.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_10_dice0.5118.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_11_dice0.4940.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_12_dice0.4337.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_13_dice0.5418.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_14_dice0.4955.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_15_dice0.4757.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_16_dice0.4677.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_17_dice0.5989.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_18_dice0.4867.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_19_dice0.4599.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_20_dice0.5416.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_21_dice0.5191.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_22_dice0.5165.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_23_dice0.5507.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_24_dice0.5597.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_25_dice0.5447.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_26_dice0.5175.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_27_dice0.5626.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_28_dice0.5850.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_29_dice0.5476.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_30_dice0.5424.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_31_dice0.5560.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_32_dice0.5745.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_33_dice0.5410.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_34_dice0.5095.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_35_dice0.5819.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_36_dice0.5784.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_37_dice0.5184.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_38_dice0.5682.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_39_dice0.5744.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_40_dice0.4709.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_41_dice0.5370.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_42_dice0.5813.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_43_dice0.5734.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_44_dice0.4951.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_45_dice0.5870.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_46_dice0.6098.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_47_dice0.5604.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_48_dice0.5889.pth
    output/dpc/msd_task03+grpo+icl/2026-01-22-08-26-55/epoch_49_dice0.5787.pth
)

export CUDA_VISIBLE_DEVICES=1

for pretrain in ${ckpt[@]};
do
    for shot in 1;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task03 \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            # -no_agent
            # -vis
    done
done