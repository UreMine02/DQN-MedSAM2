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
    output/dpc/msd_task03+no_agent+icl/epoch_0_dice0.1272.pth
    output/dpc/msd_task03+no_agent+icl/epoch_1_dice0.2299.pth
    output/dpc/msd_task03+no_agent+icl/epoch_2_dice0.1770.pth
    output/dpc/msd_task03+no_agent+icl/epoch_3_dice0.2363.pth
    output/dpc/msd_task03+no_agent+icl/epoch_4_dice0.2022.pth
    output/dpc/msd_task03+no_agent+icl/epoch_5_dice0.1750.pth
    output/dpc/msd_task03+no_agent+icl/epoch_6_dice0.1878.pth
    output/dpc/msd_task03+no_agent+icl/epoch_7_dice0.2580.pth
    output/dpc/msd_task03+no_agent+icl/epoch_8_dice0.2030.pth
    output/dpc/msd_task03+no_agent+icl/epoch_9_dice0.1656.pth
    output/dpc/msd_task03+no_agent+icl/epoch_10_dice0.0003.pth
    output/dpc/msd_task03+no_agent+icl/epoch_11_dice0.1723.pth
    output/dpc/msd_task03+no_agent+icl/epoch_12_dice0.1096.pth
    output/dpc/msd_task03+no_agent+icl/epoch_13_dice0.2313.pth
    output/dpc/msd_task03+no_agent+icl/epoch_14_dice0.1475.pth
    output/dpc/msd_task03+no_agent+icl/epoch_15_dice0.0991.pth
    output/dpc/msd_task03+no_agent+icl/epoch_16_dice0.1697.pth
    output/dpc/msd_task03+no_agent+icl/epoch_17_dice0.0999.pth
    output/dpc/msd_task03+no_agent+icl/epoch_18_dice0.0892.pth
    output/dpc/msd_task03+no_agent+icl/epoch_19_dice0.2790.pth
    output/dpc/msd_task03+no_agent+icl/epoch_20_dice0.2590.pth
    output/dpc/msd_task03+no_agent+icl/epoch_21_dice0.0838.pth
    output/dpc/msd_task03+no_agent+icl/epoch_22_dice0.0594.pth
    output/dpc/msd_task03+no_agent+icl/epoch_23_dice0.1521.pth
    output/dpc/msd_task03+no_agent+icl/epoch_24_dice0.1112.pth
    output/dpc/msd_task03+no_agent+icl/epoch_25_dice0.1908.pth
    output/dpc/msd_task03+no_agent+icl/epoch_26_dice0.1768.pth
    output/dpc/msd_task03+no_agent+icl/epoch_27_dice0.1187.pth
    output/dpc/msd_task03+no_agent+icl/epoch_28_dice0.1376.pth
    output/dpc/msd_task03+no_agent+icl/epoch_29_dice0.1202.pth
    output/dpc/msd_task03+no_agent+icl/epoch_30_dice0.1190.pth
    output/dpc/msd_task03+no_agent+icl/epoch_31_dice0.0216.pth
    output/dpc/msd_task03+no_agent+icl/epoch_32_dice0.0088.pth
    output/dpc/msd_task03+no_agent+icl/epoch_33_dice0.1266.pth
    output/dpc/msd_task03+no_agent+icl/epoch_34_dice0.1458.pth
    output/dpc/msd_task03+no_agent+icl/epoch_35_dice0.0903.pth
    output/dpc/msd_task03+no_agent+icl/epoch_36_dice0.2044.pth
    output/dpc/msd_task03+no_agent+icl/epoch_37_dice0.0739.pth
    output/dpc/msd_task03+no_agent+icl/epoch_38_dice0.1144.pth
    output/dpc/msd_task03+no_agent+icl/epoch_39_dice0.1502.pth
    output/dpc/msd_task03+no_agent+icl/epoch_40_dice0.1279.pth
    output/dpc/msd_task03+no_agent+icl/epoch_41_dice0.0864.pth
    output/dpc/msd_task03+no_agent+icl/epoch_42_dice0.1880.pth
    output/dpc/msd_task03+no_agent+icl/epoch_43_dice0.1300.pth
    output/dpc/msd_task03+no_agent+icl/epoch_44_dice0.2398.pth
    output/dpc/msd_task03+no_agent+icl/epoch_45_dice0.1638.pth
    output/dpc/msd_task03+no_agent+icl/epoch_46_dice0.1051.pth
    output/dpc/msd_task03+no_agent+icl/epoch_47_dice0.2180.pth
    output/dpc/msd_task03+no_agent+icl/epoch_48_dice0.2412.pth
    output/dpc/msd_task03+no_agent+icl/epoch_49_dice0.1510.pth
)

export CUDA_VISIBLE_DEVICES=0

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
            -no_agent
            # -vis
    done
done