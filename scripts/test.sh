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

# declare -a ckpt=(
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_0_dice0.3515.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_1_dice0.5593.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_2_dice0.5290.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_3_dice0.5137.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_4_dice0.5181.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_5_dice0.5189.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_6_dice0.5518.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_7_dice0.5410.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_8_dice0.5459.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_9_dice0.5542.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_10_dice0.6187.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_11_dice0.5775.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_12_dice0.5916.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_13_dice0.5710.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_14_dice0.5439.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_15_dice0.6137.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_16_dice0.5431.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_17_dice0.6022.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_18_dice0.6074.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_19_dice0.6108.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_20_dice0.5816.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_21_dice0.5904.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_22_dice0.5662.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_23_dice0.5844.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_24_dice0.5746.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_25_dice0.5853.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_26_dice0.5266.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_27_dice0.5946.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_28_dice0.5767.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_29_dice0.5940.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_30_dice0.5927.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_31_dice0.5763.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_32_dice0.5800.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_33_dice0.5749.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_34_dice0.5946.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_35_dice0.5746.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_36_dice0.5826.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_37_dice0.5912.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_38_dice0.5896.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_39_dice0.6060.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_40_dice0.5968.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_41_dice0.5647.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_42_dice0.5730.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_43_dice0.5810.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_44_dice0.5853.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_45_dice0.5804.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_46_dice0.6003.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_47_dice0.6035.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_48_dice0.5868.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-12-20-59-31/epoch_49_dice0.5996.pth
# )

ckpt=(
    output/dpc/btcv+grpo+icl/2026-01-12-21-05-49/epoch_9_dice0.5889.pth
)

export CUDA_VISIBLE_DEVICES=0

for pretrain in ${ckpt[@]};
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset btcv \
            -data_path /data/datasets/nii/BTCV \
            -num_support $shot \
            -vis
    done
done