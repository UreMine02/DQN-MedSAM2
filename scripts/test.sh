#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=24:00:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-btcv-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    # output/msd_task10+grpo+icl/2026-03-10-19-22-49/best.pth

    # NOTE: TESTING
    # output/btcv+grpo+icl+cw_soft_gating+obj_ptr_gating/2026-03-27-20-13-36/best.pth
    # output/msd_task02+grpo+icl+discri_gating+obj_ptr_gating+highres_gating_by_lowres+rigor_augment/2026-03-29-11-38-40/best.pth
    # output/msd_task02+grpo+icl+cw_gating+semantic_filtering+force_add+highres_gating/2026-03-25-20-05-33/best.pth
    # output/msd_task02+grpo+icl+no_agent+rigor_augment/2026-03-30-09-38-30/best.pth

    # output/msd_task02+no_agent+icl+no_augment/2026-04-03-19-32-08/best.pth
    # output/msd_task02+grpo+icl+no_augment/2026-04-04-10-53-56/best.pth
    # output/msd_task02+grpo+icl+cw_soft_gating+obj_ptr_gating+no_augment/2026-04-04-20-00-33/best.pth

    # output/sarcoma+icl+ppo+lazy_penalty0.0+long_horizon+no_augment/2026-04-20-18-59-33/best.pth
    # output/sarcoma+icl+ppo+lazy_penalty0.0+increasing_update+long_horizon+augment/2026-04-21-10-11-01/best.pth
    # output/msd_task02+icl+ppo+long_horizon+no_augment/2026-04-16-10-46-24/best.pth

    # output/msd_task02+icl+grpo+long_horizon+flip_augment/2026-04-22-20-03-14/best.pth
    # output/msd_task02+icl+grpo+lazy_pen0.1+long_horizon+flip_augment/2026-04-23-09-52-39/best.pth
    # output/msd_task02+icl+no_agent+long_horizon+no_augment/2026-04-15-19-42-29/best.pth
    # output/msd_task02+icl+grpo+lazy_pen1.0+long_horizon+flip_augment/2026-04-23-16-01-08/best.pth

    # MSD Pancreas
    # output/msd_task07+grpo+icl+long_horizon+no_augment/2026-04-26-17-18-17/epoch_21_dice0.5748.pth

    # MSD Hippo
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_0_dice0.7393.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_1_dice0.7651.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_2_dice0.7603.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_3_dice0.7728.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_4_dice0.7768.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_5_dice0.7568.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_6_dice0.7708.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_7_dice0.7666.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_8_dice0.7689.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_9_dice0.7760.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_10_dice0.7685.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_11_dice0.7821.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_12_dice0.7656.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_13_dice0.7806.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_14_dice0.7732.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_15_dice0.7715.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_16_dice0.7727.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_17_dice0.7744.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_18_dice0.7704.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_19_dice0.7465.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_20_dice0.7729.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_21_dice0.7780.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_22_dice0.7683.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_23_dice0.7765.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_24_dice0.7822.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_25_dice0.7692.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_26_dice0.7775.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_27_dice0.7758.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_28_dice0.7732.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_29_dice0.7871.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_30_dice0.7716.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_31_dice0.7734.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_32_dice0.7738.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_33_dice0.7744.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_34_dice0.7728.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_35_dice0.7763.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_36_dice0.7732.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_37_dice0.7718.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_38_dice0.7745.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_39_dice0.7729.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_40_dice0.7759.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_41_dice0.7758.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_42_dice0.7739.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_43_dice0.7724.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_44_dice0.7716.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_45_dice0.7722.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_46_dice0.7747.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_47_dice0.7731.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_48_dice0.7726.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_49_dice0.7762.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_50_dice0.7723.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_51_dice0.7738.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_52_dice0.7766.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_53_dice0.7761.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_54_dice0.7727.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_55_dice0.7726.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_56_dice0.7724.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_57_dice0.7722.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_58_dice0.7710.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_59_dice0.7747.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_60_dice0.7737.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_61_dice0.7721.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_62_dice0.7715.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_63_dice0.7795.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_64_dice0.7743.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_65_dice0.7721.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_66_dice0.7739.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_67_dice0.7734.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_68_dice0.7801.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_69_dice0.7733.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_70_dice0.7714.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_71_dice0.7761.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_72_dice0.7741.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_73_dice0.7761.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_74_dice0.7756.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_75_dice0.7738.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_76_dice0.7753.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_77_dice0.7720.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_78_dice0.7730.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_79_dice0.7736.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_80_dice0.7711.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_81_dice0.7717.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_82_dice0.7751.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_83_dice0.7711.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_84_dice0.7737.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_85_dice0.7790.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_86_dice0.7719.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_87_dice0.7744.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_88_dice0.7744.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_89_dice0.7730.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_90_dice0.7727.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_91_dice0.7741.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_92_dice0.7739.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_93_dice0.7741.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_94_dice0.7738.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_95_dice0.7745.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_96_dice0.7739.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_97_dice0.7719.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_98_dice0.7733.pth
    output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_99_dice0.7744.pth
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
            -task "Task04" \
            -data_path /data/datasets/nii/ \
            -num_support $shot \
            -memory_bank_size 6 \
            # -no_agent
    done
done
