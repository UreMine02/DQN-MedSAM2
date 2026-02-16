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
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_0_dice0.3296.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_1_dice0.4526.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_2_dice0.3251.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_3_dice0.4802.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_4_dice0.4723.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_5_dice0.5446.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_6_dice0.4494.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_7_dice0.3888.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_8_dice0.5100.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_9_dice0.5387.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_10_dice0.5532.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_11_dice0.5618.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_12_dice0.5705.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_13_dice0.4654.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_14_dice0.5564.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_15_dice0.6003.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_16_dice0.4867.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_17_dice0.5946.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_18_dice0.4822.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_19_dice0.5174.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_20_dice0.4936.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_21_dice0.5957.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_22_dice0.5206.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_23_dice0.5393.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_24_dice0.6072.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_25_dice0.5193.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_26_dice0.5649.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_27_dice0.6877.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_28_dice0.6639.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_29_dice0.6886.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_30_dice0.5565.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_31_dice0.6126.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_32_dice0.6331.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_33_dice0.5508.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_34_dice0.5907.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_35_dice0.5909.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_36_dice0.6998.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_37_dice0.5534.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_38_dice0.7173.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_39_dice0.6351.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_40_dice0.5923.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_41_dice0.6663.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_42_dice0.6590.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_43_dice0.5465.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_44_dice0.6083.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_45_dice0.5585.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_46_dice0.6368.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_47_dice0.6218.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_48_dice0.5491.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-19-08-48/epoch_49_dice0.6910.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task05 \
            -data_path /data/rlsam2/datasets/nii/MSD \
            -num_support $shot
    done
done