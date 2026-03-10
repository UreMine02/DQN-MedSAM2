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

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_0_dice0.2136.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_1_dice0.3737.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_2_dice0.3564.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_3_dice0.4488.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_4_dice0.4787.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_5_dice0.4504.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_6_dice0.4811.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_7_dice0.4904.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_8_dice0.4758.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_9_dice0.4974.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_10_dice0.5128.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_11_dice0.5089.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_12_dice0.5058.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_13_dice0.5043.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_14_dice0.4775.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_15_dice0.5179.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_16_dice0.5064.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_17_dice0.4875.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_18_dice0.5224.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_19_dice0.5198.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_20_dice0.4943.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_21_dice0.5278.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_22_dice0.4935.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_23_dice0.5343.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_24_dice0.5407.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_25_dice0.5222.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_26_dice0.5357.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_27_dice0.5264.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_28_dice0.5320.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_29_dice0.5426.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_30_dice0.5217.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_31_dice0.5115.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_32_dice0.5016.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_33_dice0.5232.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_34_dice0.5383.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_35_dice0.5323.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_36_dice0.5470.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_37_dice0.5357.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_38_dice0.5288.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_39_dice0.5331.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_40_dice0.5435.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_41_dice0.5453.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_42_dice0.5176.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_43_dice0.5100.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_44_dice0.5494.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_45_dice0.5242.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_46_dice0.5493.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_47_dice0.5140.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_48_dice0.5480.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_49_dice0.5113.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_50_dice0.5299.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_51_dice0.5247.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_52_dice0.5350.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_53_dice0.5285.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_54_dice0.5066.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_55_dice0.5218.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_56_dice0.5131.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_57_dice0.5338.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_58_dice0.5375.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_59_dice0.5166.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_60_dice0.5436.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_61_dice0.5322.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_62_dice0.5237.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_63_dice0.5423.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_64_dice0.5390.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_65_dice0.5191.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_66_dice0.5285.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_67_dice0.5350.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_68_dice0.5367.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_69_dice0.5167.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_70_dice0.5174.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_71_dice0.5182.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_72_dice0.5251.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_73_dice0.5179.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_74_dice0.5160.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_75_dice0.5296.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_76_dice0.5343.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_77_dice0.5155.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_78_dice0.5287.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_79_dice0.5478.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_80_dice0.5451.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_81_dice0.5078.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_82_dice0.5316.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_83_dice0.5429.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_84_dice0.5216.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_85_dice0.5394.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_86_dice0.5481.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_87_dice0.5399.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_88_dice0.5335.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_89_dice0.5410.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_90_dice0.5391.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_91_dice0.5247.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_92_dice0.5319.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_93_dice0.5404.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_94_dice0.5359.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_95_dice0.5294.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_96_dice0.5517.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_97_dice0.5258.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_98_dice0.5465.pth
    output/msd_task08+grpo+icl/2026-03-10-20-48-58/epoch_99_dice0.5050.pth
)

export CUDA_VISIBLE_DEVICES=1

for idx in ${!ckpt[@]}
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain ${ckpt[idx]} \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "Task08" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot
            # -no_agent
    done
done
