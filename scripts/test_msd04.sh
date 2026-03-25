#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=12:00:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-task04-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    # output/msd_task10+grpo+icl/2026-03-10-19-22-49/best.pth

    # NOTE: TESTING
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_0_dice0.6471.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_1_dice0.6690.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_2_dice0.6773.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_3_dice0.6943.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_4_dice0.6864.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_5_dice0.6840.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_6_dice0.7100.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_7_dice0.7071.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_8_dice0.7011.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_9_dice0.7043.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_10_dice0.7232.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_11_dice0.7039.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_12_dice0.6925.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_13_dice0.7059.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_14_dice0.7250.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_15_dice0.7108.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_16_dice0.7125.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_17_dice0.7272.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_18_dice0.7140.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_19_dice0.7232.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_20_dice0.7277.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_21_dice0.7296.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_22_dice0.7090.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_23_dice0.7262.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_24_dice0.7165.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_25_dice0.7295.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_26_dice0.7401.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_27_dice0.7282.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_28_dice0.7288.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_29_dice0.7310.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_30_dice0.7320.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_31_dice0.7277.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_32_dice0.7220.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_33_dice0.7444.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_34_dice0.7117.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_35_dice0.7339.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_36_dice0.7288.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_37_dice0.7319.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_38_dice0.7370.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_39_dice0.7393.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_40_dice0.7240.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_41_dice0.7282.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_42_dice0.7301.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_43_dice0.7419.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_44_dice0.7521.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_45_dice0.7299.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_46_dice0.7368.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_47_dice0.7278.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_48_dice0.7398.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_49_dice0.7283.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_50_dice0.7320.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_51_dice0.7281.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_52_dice0.7345.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_53_dice0.7292.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_54_dice0.7318.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_55_dice0.7324.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_56_dice0.7288.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_57_dice0.7233.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_58_dice0.7288.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_59_dice0.7392.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_60_dice0.7375.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_61_dice0.7357.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_62_dice0.7290.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_63_dice0.7378.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_64_dice0.7291.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_65_dice0.7287.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_66_dice0.7280.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_67_dice0.7179.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_68_dice0.7351.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_69_dice0.7197.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_70_dice0.7260.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_71_dice0.7238.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_72_dice0.7294.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_73_dice0.7310.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_74_dice0.7281.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_75_dice0.7329.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_76_dice0.7352.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_77_dice0.7267.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_78_dice0.7242.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_79_dice0.7307.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_80_dice0.7215.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_81_dice0.7321.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_82_dice0.7290.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_83_dice0.7332.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_84_dice0.7299.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_85_dice0.7335.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_86_dice0.7244.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_87_dice0.7361.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_88_dice0.7353.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_89_dice0.7345.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_90_dice0.7345.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_91_dice0.7324.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_92_dice0.7214.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_93_dice0.7326.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_94_dice0.7367.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_95_dice0.7355.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_96_dice0.7263.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_97_dice0.7341.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_98_dice0.7309.pth
    output/msd_task04+grpo+icl/2026-03-11-18-45-04/epoch_99_dice0.7283.pth
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
            -task "Task04" \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot
            # -no_agent
    done
done
