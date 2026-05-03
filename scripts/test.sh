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

declare -a ckpts=(
    # BTCV
    # output/btcv+icl+no_agent+long_horizon+no_augment/2026-04-30-00-40-30/best.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_103_dice0.7433.pth

    # Sarcoma
    # output/sarcoma+icl+grpo+long_horizon+no_augment/2026-04-29-14-22-57/best.pth
    # output/sarcoma+icl+no_agent+long_horizon+no_augment/2026-04-10-19-47-31/best.pth

    # MSD Heart
    # output/msd_task02+icl+no_agent+long_horizon+no_augment/2026-04-15-19-42-29/best.pth
    # output/msd_task02+icl+grpo+lazy_pen1.0+long_horizon+flip_augment/2026-04-23-16-01-08/best.pth
    # output/msd_task02+icl+grpo+long_horizon+no_augment/2026-04-30-18-07-03/best.pth
    # output/msd_task02+icl+grpo+long_horizon+no_augment+lr2e-4/2026-05-01-07-12-31/best.pth

    # MSD Liver
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_0_dice0.4721.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_1_dice0.4169.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_2_dice0.4636.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_3_dice0.4636.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_4_dice0.5464.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_5_dice0.5453.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_6_dice0.5444.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_7_dice0.5367.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_8_dice0.5894.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_9_dice0.6102.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_10_dice0.6017.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_11_dice0.6131.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_12_dice0.5870.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_13_dice0.6307.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_14_dice0.6481.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_15_dice0.6030.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_16_dice0.6168.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_17_dice0.6338.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_18_dice0.6384.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_19_dice0.6450.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_20_dice0.6296.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_21_dice0.6452.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_22_dice0.6341.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_23_dice0.6471.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_24_dice0.6430.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_25_dice0.6206.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_26_dice0.6215.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_27_dice0.6097.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_28_dice0.6384.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_29_dice0.6307.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_30_dice0.6348.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_31_dice0.6574.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_32_dice0.6373.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_33_dice0.6488.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_34_dice0.6165.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_35_dice0.6409.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_36_dice0.6605.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_37_dice0.6639.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_38_dice0.6528.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_39_dice0.6524.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_40_dice0.6655.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_41_dice0.6422.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_42_dice0.6470.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_43_dice0.6394.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_44_dice0.6575.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_45_dice0.6446.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_46_dice0.6505.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_47_dice0.6525.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_48_dice0.6315.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_49_dice0.6257.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_50_dice0.6456.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_51_dice0.6377.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_52_dice0.6487.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_53_dice0.6368.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_54_dice0.6053.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_55_dice0.6462.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_56_dice0.6512.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_57_dice0.6543.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_58_dice0.6580.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_59_dice0.6576.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_60_dice0.6394.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_61_dice0.6398.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_62_dice0.6494.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_63_dice0.6508.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_64_dice0.6508.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_65_dice0.6661.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_66_dice0.6322.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_67_dice0.6231.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_68_dice0.6436.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_69_dice0.6657.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_70_dice0.6542.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_71_dice0.6454.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_72_dice0.6436.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_73_dice0.6546.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_74_dice0.6541.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_75_dice0.6516.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_76_dice0.6526.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_77_dice0.6475.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_78_dice0.6191.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_79_dice0.6521.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_80_dice0.6458.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_81_dice0.6497.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_82_dice0.6249.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_83_dice0.6511.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_84_dice0.6495.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_85_dice0.6529.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_86_dice0.6135.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_87_dice0.6437.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_88_dice0.6479.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_89_dice0.6310.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_90_dice0.6420.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_91_dice0.6190.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_92_dice0.6244.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_93_dice0.6524.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_94_dice0.6478.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_95_dice0.6391.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_96_dice0.6329.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_97_dice0.6353.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_98_dice0.6450.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_99_dice0.6473.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_100_dice0.6507.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_101_dice0.6473.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_102_dice0.6489.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_103_dice0.6465.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_104_dice0.6505.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_105_dice0.6492.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_106_dice0.6438.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_107_dice0.6301.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_108_dice0.6491.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_109_dice0.6381.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_110_dice0.6465.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_111_dice0.6462.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_112_dice0.6455.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_113_dice0.6497.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_114_dice0.6415.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_115_dice0.6454.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_116_dice0.6405.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_117_dice0.6399.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_118_dice0.6456.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_119_dice0.6198.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_120_dice0.6413.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_121_dice0.6210.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_122_dice0.6191.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_123_dice0.6451.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_124_dice0.6432.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_125_dice0.6263.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_126_dice0.6353.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_127_dice0.6471.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_128_dice0.6481.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_129_dice0.6476.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_130_dice0.6388.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_131_dice0.6257.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_132_dice0.6154.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_133_dice0.6301.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_134_dice0.6302.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_135_dice0.6431.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_136_dice0.6031.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_137_dice0.6204.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_138_dice0.6084.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_139_dice0.6175.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_140_dice0.6484.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_141_dice0.6510.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_142_dice0.6480.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_143_dice0.6450.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_144_dice0.6203.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-01-20-32-39/epoch_145_dice0.6445.pth
    
    # MSD Hippocampus
    # output/msd_task04+grpo+icl+long_horizon+no_augment/2026-04-27-11-50-03/epoch_24_dice0.7822.pth
    # output/msd_task04+no_agent+icl+long_horizon+no_augment/2026-04-28-20-05-44/best.pth

    # MSD Prostate
    # output/msd_task05+grpo+long_horizon+no_augment/2026-04-26-11-35-38/best.pth
    # output/msd_task05+no_agent+long_horizon+no_augment/2026-04-13-18-04-27/best.pth

    # MSD Lung
    # output/msd_task06+grpo+long_horizon+no_augment/2026-04-26-16-12-37/best.pth
    # output/msd_task06+icl+no_agent+long_horizon+no_augment/2026-04-28-17-12-14/best.pth

    # MSD Pancreas
    # output/msd_task07+no_agent+icl+long_horizon+no_augment/2026-04-28-20-07-36/best.pth
    # output/msd_task07+grpo+icl+long_horizon+no_augment/2026-04-26-17-18-17/epoch_21_dice0.5748.pth

    # MSD HepaticVessel
    # output/msd_task08+no_agent+icl+long_horizon+no_augment/2026-04-28-20-20-31/best.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/best.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_24_dice0.5453.pth

    # MSD Spleen
    # output/msd_task09+icl+grpo+long_horizon+no_augment/2026-04-25-19-48-27/best.pth
    # output/msd_task09+icl+no_agent+long_horizon+no_augment/2026-04-11-12-19-00/best.pth
    # output/msd_task09+icl+grpo+long_horizon+no_augment/2026-05-01-22-59-45/best.pth

    # MSD Colon
    # output/msd_task10+icl+grpo+long_horizon+no_augment/2026-04-25-13-09-26/best.pth
    # output/msd_task10+icl+no_agent+long_horizon+no_augment/2026-04-12-15-52-38/best.pth
)

declare -a datasets=()

declare -a tasks=()

declare -a no_agents=(
    no_agent
    agent
)

for i in ${!ckpts[@]};
do
    # echo $i
    datasets+=("btcv")
    tasks+=("")
    # no_agents+=("no_agent")
done

# echo $datasets

export CUDA_VISIBLE_DEVICES=1

for idx in ${!ckpts[@]}
do
    # echo ${ckpts[idx]}
    # echo ${tasks[idx]}
    # echo ${no_agents[idx]}
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain "${ckpts[idx]}" \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset "${datasets[idx]}" \
            -task "${tasks[idx]}" \
            -data_path /data/datasets/nii/ \
            -num_support $shot \
            -memory_bank_size 6 \
            -ablation \
            -vis \
            $( [ ${no_agents[idx]} = "no_agent" ] && echo "-no_agent" )
    done
done
