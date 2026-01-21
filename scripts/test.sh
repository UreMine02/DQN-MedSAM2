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
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_0_dice0.4064.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_1_dice0.4165.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_2_dice0.4506.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_3_dice0.4444.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_4_dice0.4746.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_5_dice0.4455.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_6_dice0.4722.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_7_dice0.4376.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_8_dice0.4760.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_9_dice0.4392.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_10_dice0.4776.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_11_dice0.4760.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_12_dice0.4699.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_13_dice0.4943.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_14_dice0.4403.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_15_dice0.4805.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_16_dice0.4743.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_17_dice0.4752.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_18_dice0.4696.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_19_dice0.4761.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_20_dice0.4905.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_21_dice0.4791.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_22_dice0.4618.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_23_dice0.4668.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_24_dice0.4747.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_25_dice0.4963.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_26_dice0.4950.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_27_dice0.4986.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_28_dice0.4535.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_29_dice0.4829.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_30_dice0.4627.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_31_dice0.5077.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_32_dice0.4775.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_33_dice0.4850.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_34_dice0.5043.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_35_dice0.5009.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_36_dice0.5165.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_37_dice0.5106.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_38_dice0.4854.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_39_dice0.4710.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_40_dice0.4649.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_41_dice0.4703.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_42_dice0.4894.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_43_dice0.4782.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_44_dice0.4761.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_45_dice0.4992.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_46_dice0.4948.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_47_dice0.4938.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_48_dice0.5203.pth
    output/dpc/msd_task01+grpo+icl/2026-01-19-16-37-50/epoch_49_dice0.4743.pth
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
            -task Task01 \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            # -no_agent
            # -vis
    done
done