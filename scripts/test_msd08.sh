#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=12:00:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-task08-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    # output/msd_task10+grpo+icl/2026-03-10-19-22-49/best.pth

    # NOTE: TESTING
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_0_dice0.2656.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_1_dice0.3369.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_2_dice0.3669.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_3_dice0.4103.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_4_dice0.4343.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_5_dice0.4114.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_6_dice0.4710.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_7_dice0.4635.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_8_dice0.4597.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_9_dice0.4903.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_10_dice0.5034.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_11_dice0.5053.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_12_dice0.5166.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_13_dice0.4737.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_14_dice0.5101.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_15_dice0.5161.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_16_dice0.5078.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_17_dice0.5123.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_18_dice0.4934.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_19_dice0.5235.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_20_dice0.5249.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_21_dice0.5374.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_22_dice0.5198.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_23_dice0.5297.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_24_dice0.5254.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_25_dice0.5195.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_26_dice0.5326.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_27_dice0.5331.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_28_dice0.5176.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_29_dice0.5369.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_30_dice0.5297.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_31_dice0.5221.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_32_dice0.5448.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_33_dice0.5366.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_34_dice0.5388.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_35_dice0.5329.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_36_dice0.5262.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_37_dice0.5173.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_38_dice0.5400.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_39_dice0.5413.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_40_dice0.5371.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_41_dice0.5411.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_42_dice0.5291.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_43_dice0.5380.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_44_dice0.5327.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_45_dice0.5395.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_46_dice0.5414.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_47_dice0.5415.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_48_dice0.5424.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_49_dice0.5475.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_50_dice0.5460.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_51_dice0.5360.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_52_dice0.5449.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_53_dice0.5456.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_54_dice0.5424.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_55_dice0.5499.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_56_dice0.5362.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_57_dice0.5475.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_58_dice0.5354.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_59_dice0.5472.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_60_dice0.5370.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_61_dice0.5430.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_62_dice0.5426.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_63_dice0.5384.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_64_dice0.5301.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_65_dice0.5453.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_66_dice0.5366.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_67_dice0.5358.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_68_dice0.5382.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_69_dice0.5495.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_70_dice0.5453.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_71_dice0.5388.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_72_dice0.5435.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_73_dice0.5424.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_74_dice0.5463.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_75_dice0.5422.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_76_dice0.5386.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_77_dice0.5467.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_78_dice0.5467.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_79_dice0.5365.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_80_dice0.5457.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_81_dice0.5422.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_82_dice0.5382.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_83_dice0.5409.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_84_dice0.5413.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_85_dice0.5435.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_86_dice0.5470.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_87_dice0.5425.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_88_dice0.5455.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_89_dice0.5454.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_90_dice0.5411.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_91_dice0.5413.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_92_dice0.5417.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_93_dice0.5409.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_94_dice0.5422.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_95_dice0.5454.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_96_dice0.5454.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_97_dice0.5464.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_98_dice0.5383.pth
    output/msd_task08+grpo+icl/2026-03-11-17-05-35/epoch_99_dice0.5440.pth
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
            -task "Task08" \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot
            # -no_agent
    done
done
