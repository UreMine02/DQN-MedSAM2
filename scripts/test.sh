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
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_0_dice0.2659.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_1_dice0.2678.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_2_dice0.1916.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_3_dice0.2618.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_4_dice0.4841.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_5_dice0.3475.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_6_dice0.4538.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_7_dice0.4570.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_8_dice0.4220.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_9_dice0.3946.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_10_dice0.4547.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_11_dice0.4140.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_12_dice0.5181.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_13_dice0.5305.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_14_dice0.4777.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_15_dice0.4556.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_16_dice0.4535.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_17_dice0.4612.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_18_dice0.4349.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_19_dice0.5403.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_20_dice0.5462.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_21_dice0.5029.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_22_dice0.5563.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_23_dice0.5930.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_24_dice0.5301.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_25_dice0.5270.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_26_dice0.5197.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_27_dice0.6023.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_28_dice0.5814.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_29_dice0.6270.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_30_dice0.6036.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_31_dice0.5186.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_32_dice0.5586.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_33_dice0.5764.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_34_dice0.4737.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_35_dice0.5734.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_36_dice0.6140.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_37_dice0.5908.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_38_dice0.6240.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_39_dice0.6214.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_40_dice0.6062.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_41_dice0.5913.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_42_dice0.5975.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_43_dice0.6416.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_44_dice0.6069.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_45_dice0.5844.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_46_dice0.6125.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_47_dice0.5998.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_48_dice0.5906.pth
#     output/dpc/msd_task03+grpo+icl/2026-01-11-19-44-42/epoch_49_dice0.5987.pth
# )

# declare -a ckpt=(
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_0_dice0.6032.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_1_dice0.5583.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_2_dice0.6668.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_3_dice0.6827.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_4_dice0.6666.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_5_dice0.6176.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_6_dice0.6625.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_7_dice0.6458.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_8_dice0.6552.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_9_dice0.6559.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_10_dice0.6533.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_11_dice0.7299.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_12_dice0.6522.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_13_dice0.7206.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_14_dice0.6528.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_15_dice0.7205.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_16_dice0.7350.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_17_dice0.7383.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_18_dice0.6395.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_19_dice0.7071.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_20_dice0.7201.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_21_dice0.7255.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_22_dice0.7361.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_23_dice0.7512.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_24_dice0.7324.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_25_dice0.6372.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_26_dice0.7358.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_27_dice0.7283.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_28_dice0.7440.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_29_dice0.7442.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_30_dice0.7441.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_31_dice0.6696.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_32_dice0.6661.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_33_dice0.6604.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_34_dice0.6515.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_35_dice0.7312.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_36_dice0.6432.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_37_dice0.6745.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_38_dice0.6377.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_39_dice0.7389.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_40_dice0.7404.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_41_dice0.6485.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_42_dice0.6578.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_43_dice0.7303.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_44_dice0.6370.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_45_dice0.7245.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_46_dice0.7377.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_47_dice0.6421.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_48_dice0.6548.pth
    # output/dpc/msd_task04+grpo+icl/2026-01-11-16-34-59/epoch_49_dice0.7328.pth
# )

declare -a ckpt=(
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_0_dice0.3882.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_1_dice0.4146.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_2_dice0.4531.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_3_dice0.4403.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_4_dice0.5310.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_5_dice0.4483.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_6_dice0.4575.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_7_dice0.4539.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_8_dice0.5229.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_9_dice0.5089.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_10_dice0.4659.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_11_dice0.5244.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_12_dice0.4504.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_13_dice0.4906.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_14_dice0.4626.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_15_dice0.5127.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_16_dice0.5459.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_17_dice0.5604.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_18_dice0.5524.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_19_dice0.5527.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_20_dice0.5111.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_21_dice0.5742.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_22_dice0.5581.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_23_dice0.5675.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_24_dice0.4932.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_25_dice0.5077.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_26_dice0.5800.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_27_dice0.5795.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_28_dice0.5855.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_29_dice0.5404.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_30_dice0.5595.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_31_dice0.5288.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_32_dice0.4979.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_33_dice0.5446.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_34_dice0.5564.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_35_dice0.5930.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_36_dice0.5408.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_37_dice0.5634.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_38_dice0.5941.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_39_dice0.5973.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_40_dice0.5784.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_41_dice0.5080.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_42_dice0.5538.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_43_dice0.5773.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_44_dice0.5212.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_45_dice0.5282.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_46_dice0.5627.pth
    output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_47_dice0.5150.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_48_dice0.5000.pth
    # output/dpc/btcv+grpo+icl/2026-01-11-16-36-16/epoch_49_dice0.4945.pth
)

export CUDA_VISIBLE_DEVICES=1

for pretrain in ${ckpt[@]};
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset btcv \
            -task Task03 \
            -data_path /data/datasets/nii/BTCV \
            -num_support $shot \
            -vis
    done
done