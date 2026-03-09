#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=1:00:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-task07-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

declare -a ckpt=(
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_0_dice0.4087.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_1_dice0.4827.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_2_dice0.4879.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_3_dice0.5046.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_4_dice0.4988.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_5_dice0.4983.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_6_dice0.5405.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_7_dice0.5431.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_8_dice0.5441.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_9_dice0.5253.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_10_dice0.4989.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_11_dice0.4807.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_12_dice0.5277.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_13_dice0.5470.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_15_dice0.5499.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_16_dice0.5369.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_17_dice0.5123.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_18_dice0.5641.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_19_dice0.5502.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_20_dice0.5659.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_21_dice0.5718.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_22_dice0.5671.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_23_dice0.5417.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_24_dice0.5717.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_25_dice0.5779.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_26_dice0.5628.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_27_dice0.5435.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_28_dice0.5764.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_29_dice0.5392.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_30_dice0.5656.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_31_dice0.5865.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_32_dice0.5733.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_33_dice0.5645.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_34_dice0.5636.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_35_dice0.5451.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_36_dice0.5706.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_37_dice0.5646.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_38_dice0.5887.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_39_dice0.5654.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_40_dice0.5774.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_41_dice0.5738.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_42_dice0.5798.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_43_dice0.5770.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_44_dice0.5754.pth
    output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_45_dice0.5663.pth
)

declare -a dataset=(
    # Task03
    # Task04
    Task05
    # Task06
    # Task07
    # Task08
)

# export CUDA_VISIBLE_DEVICES=1

for idx in ${!ckpt[@]}
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain ${ckpt[idx]} \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "Task07" \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot
            # -no_agent
    done
done
