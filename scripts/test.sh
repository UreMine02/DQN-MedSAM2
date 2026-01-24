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
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_0_dice0.3498.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_1_dice0.4387.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_2_dice0.4585.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_3_dice0.4316.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_4_dice0.3112.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_5_dice0.2528.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_6_dice0.4338.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_7_dice0.5266.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_8_dice0.4497.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_9_dice0.3389.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_10_dice0.4886.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_11_dice0.5672.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_12_dice0.4913.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_13_dice0.5037.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_14_dice0.4846.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_15_dice0.4690.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_16_dice0.5466.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_17_dice0.5207.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_18_dice0.3142.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_19_dice0.5732.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_20_dice0.4972.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_21_dice0.6166.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_22_dice0.4777.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_23_dice0.5392.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_24_dice0.4190.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_25_dice0.5249.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_26_dice0.4008.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_27_dice0.4962.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_28_dice0.4499.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_29_dice0.5484.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_30_dice0.4637.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_31_dice0.5437.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_32_dice0.5268.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_33_dice0.5021.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_34_dice0.5368.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_35_dice0.5356.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_36_dice0.5641.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_37_dice0.4449.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_38_dice0.5673.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_39_dice0.5617.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_40_dice0.5286.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_41_dice0.5292.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_42_dice0.5381.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_43_dice0.5630.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_44_dice0.5842.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_45_dice0.5515.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_46_dice0.4962.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_47_dice0.5230.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_48_dice0.5354.pth
    output/msd_task03+grpo+icl/2026-01-23-15-39-00/epoch_49_dice0.5462.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task03 \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot
            # -no_agent
            # -vis
    done
done