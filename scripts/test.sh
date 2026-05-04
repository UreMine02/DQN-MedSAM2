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
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_0_dice0.4325.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_1_dice0.3734.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_2_dice0.4329.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_3_dice0.3293.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_4_dice0.4968.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_5_dice0.5317.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_6_dice0.5288.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_7_dice0.5610.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_8_dice0.5847.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_9_dice0.5733.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_10_dice0.5735.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_11_dice0.5766.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_12_dice0.6086.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_13_dice0.6052.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_14_dice0.6113.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_15_dice0.5906.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_16_dice0.6163.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_17_dice0.5940.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_18_dice0.6210.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_19_dice0.6177.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_20_dice0.5677.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_21_dice0.6101.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_22_dice0.6194.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_23_dice0.6151.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_24_dice0.6383.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_25_dice0.6087.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_26_dice0.6249.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_27_dice0.6207.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_28_dice0.6336.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_29_dice0.6131.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_30_dice0.6280.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_31_dice0.6252.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_32_dice0.6329.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_33_dice0.6389.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_34_dice0.6334.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_35_dice0.6356.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_36_dice0.6369.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_37_dice0.6484.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_38_dice0.6377.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_39_dice0.6359.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_40_dice0.6520.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_41_dice0.6578.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_42_dice0.6512.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_43_dice0.6471.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_44_dice0.6435.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_45_dice0.6415.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_46_dice0.6454.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_47_dice0.6456.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_48_dice0.6270.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_49_dice0.6408.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_50_dice0.6510.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_51_dice0.6244.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_52_dice0.6340.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_53_dice0.6037.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_54_dice0.6045.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_55_dice0.6173.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_56_dice0.6043.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_57_dice0.6374.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_58_dice0.6222.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_59_dice0.6231.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_60_dice0.6111.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_61_dice0.5944.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_62_dice0.6531.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_63_dice0.5989.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_64_dice0.6276.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_65_dice0.6339.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_66_dice0.6326.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_67_dice0.6383.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_68_dice0.6231.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_69_dice0.6393.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_70_dice0.6379.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_71_dice0.6433.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_72_dice0.6237.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_73_dice0.6163.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_74_dice0.6219.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_75_dice0.6272.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_76_dice0.6358.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_77_dice0.6326.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_78_dice0.6201.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_79_dice0.6200.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_80_dice0.6331.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_81_dice0.6265.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_82_dice0.6201.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_83_dice0.6257.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_84_dice0.6316.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_85_dice0.6312.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_86_dice0.5950.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_87_dice0.6266.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_88_dice0.6043.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_89_dice0.6305.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_90_dice0.6311.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_91_dice0.6310.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_92_dice0.6303.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_93_dice0.6242.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_94_dice0.6133.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_95_dice0.6323.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_96_dice0.6235.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_97_dice0.6245.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_98_dice0.6277.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_99_dice0.6309.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_100_dice0.6287.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_101_dice0.6333.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_102_dice0.6270.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_103_dice0.6273.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_104_dice0.6037.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_105_dice0.6274.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_106_dice0.6245.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_107_dice0.6193.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_108_dice0.5918.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_109_dice0.6236.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_110_dice0.6231.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_111_dice0.5908.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_112_dice0.6216.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_113_dice0.6327.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_114_dice0.6326.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_115_dice0.6147.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_116_dice0.6009.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_117_dice0.6271.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_118_dice0.6258.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_119_dice0.6292.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_120_dice0.6225.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_121_dice0.6254.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_122_dice0.6258.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_123_dice0.6295.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_124_dice0.6236.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_125_dice0.6167.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_126_dice0.6182.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_127_dice0.6248.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_128_dice0.6282.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_129_dice0.6253.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_130_dice0.6007.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_131_dice0.5995.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_132_dice0.6172.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_133_dice0.6276.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_134_dice0.6263.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_135_dice0.5997.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_136_dice0.6246.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_137_dice0.6240.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_138_dice0.6075.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_139_dice0.5834.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_140_dice0.6145.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_141_dice0.6355.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_142_dice0.6297.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_143_dice0.6305.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_144_dice0.6309.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_145_dice0.6287.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_146_dice0.6277.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_147_dice0.6166.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_148_dice0.6298.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_149_dice0.6014.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_150_dice0.6184.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_151_dice0.5746.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_152_dice0.6002.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_153_dice0.6278.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_154_dice0.5999.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_155_dice0.6237.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_156_dice0.6290.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_157_dice0.6258.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_158_dice0.6264.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_159_dice0.6307.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_160_dice0.6269.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_161_dice0.5989.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_162_dice0.5989.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_163_dice0.6245.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_164_dice0.6187.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_165_dice0.6251.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_166_dice0.6118.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_167_dice0.6169.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_168_dice0.6107.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_169_dice0.6151.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_170_dice0.5976.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_171_dice0.6081.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_172_dice0.6010.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_173_dice0.6293.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_174_dice0.6216.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_175_dice0.6289.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_176_dice0.6288.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_177_dice0.6286.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_178_dice0.6270.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_179_dice0.6256.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_180_dice0.6269.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_181_dice0.6279.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_182_dice0.5993.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_183_dice0.6270.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_184_dice0.6114.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_185_dice0.5964.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_186_dice0.6259.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-19-46-44/epoch_187_dice0.6270.pth

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
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_24_dice0.5453.pth

    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_0_dice0.3045.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_1_dice0.3119.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_2_dice0.4610.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_3_dice0.4827.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_4_dice0.5065.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_5_dice0.4415.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_6_dice0.5204.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_7_dice0.5149.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_8_dice0.5126.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_9_dice0.5337.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_10_dice0.5355.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_11_dice0.5077.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_12_dice0.5079.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_13_dice0.5397.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_14_dice0.5066.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_15_dice0.5216.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_16_dice0.5329.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_17_dice0.5298.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_18_dice0.5225.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_19_dice0.5456.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_20_dice0.5427.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_21_dice0.5435.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_22_dice0.5417.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_23_dice0.5351.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_24_dice0.5407.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_25_dice0.5312.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_26_dice0.5442.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_27_dice0.5358.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_28_dice0.5409.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_29_dice0.5468.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_30_dice0.5462.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_31_dice0.5421.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_32_dice0.5378.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_33_dice0.5504.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_34_dice0.5427.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_35_dice0.5436.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_36_dice0.5432.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_37_dice0.5505.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_38_dice0.5401.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_39_dice0.5413.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_40_dice0.5304.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_41_dice0.5470.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_42_dice0.5454.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_43_dice0.5392.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_44_dice0.5370.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_45_dice0.5358.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_46_dice0.5251.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_47_dice0.5388.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_48_dice0.5449.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_49_dice0.5437.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_50_dice0.5373.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_51_dice0.5403.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_52_dice0.5479.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_53_dice0.5498.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_54_dice0.5314.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_55_dice0.5385.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_56_dice0.5444.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_57_dice0.5370.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_58_dice0.5337.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_59_dice0.5236.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_60_dice0.5431.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_61_dice0.5453.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_62_dice0.5430.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_63_dice0.5408.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_64_dice0.5398.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_65_dice0.5324.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_66_dice0.5214.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_67_dice0.5462.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_68_dice0.5436.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_69_dice0.5442.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_70_dice0.5447.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_71_dice0.5340.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_72_dice0.5418.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_73_dice0.5399.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_74_dice0.5397.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_75_dice0.5377.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_76_dice0.5291.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_77_dice0.5345.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_78_dice0.5361.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_79_dice0.5383.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_80_dice0.5359.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_81_dice0.5409.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_82_dice0.5397.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_83_dice0.5323.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_84_dice0.5410.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_85_dice0.5377.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_86_dice0.5299.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_87_dice0.5337.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_88_dice0.5323.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_89_dice0.5321.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_90_dice0.5312.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_91_dice0.5363.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_92_dice0.5287.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_93_dice0.5314.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_94_dice0.5354.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_95_dice0.5337.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_96_dice0.5369.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_97_dice0.5319.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_98_dice0.5414.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_99_dice0.5344.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_100_dice0.5336.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_101_dice0.5346.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_102_dice0.5298.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_103_dice0.5310.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_104_dice0.5315.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_105_dice0.5372.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_106_dice0.5320.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_107_dice0.5379.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_108_dice0.5325.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_109_dice0.5305.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_110_dice0.5336.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_111_dice0.5292.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_112_dice0.5319.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_113_dice0.5351.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_114_dice0.5325.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_115_dice0.5323.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_116_dice0.5367.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_117_dice0.5346.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_118_dice0.5364.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_119_dice0.5307.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_120_dice0.5350.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_121_dice0.5367.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_122_dice0.5313.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_123_dice0.5383.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_124_dice0.5357.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_125_dice0.5321.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_126_dice0.5318.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_127_dice0.5378.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_128_dice0.5347.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_129_dice0.5343.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_130_dice0.5340.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_131_dice0.5354.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_132_dice0.5341.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_133_dice0.5331.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_134_dice0.5336.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_135_dice0.5367.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_136_dice0.5344.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_137_dice0.5337.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_138_dice0.5333.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_139_dice0.5307.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_140_dice0.5344.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_141_dice0.5337.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_142_dice0.5270.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_143_dice0.5312.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_144_dice0.5336.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_145_dice0.5360.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_146_dice0.5333.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_147_dice0.5376.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_148_dice0.5344.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_149_dice0.5347.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_150_dice0.5373.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_151_dice0.5360.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_152_dice0.5322.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_153_dice0.5334.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_154_dice0.5362.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_155_dice0.5362.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_156_dice0.5297.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_157_dice0.5343.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_158_dice0.5334.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_159_dice0.5302.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_160_dice0.5332.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_161_dice0.5354.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_162_dice0.5334.pth
    output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_163_dice0.5347.pth

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

declare -a no_agents=()

for i in ${!ckpts[@]};
do
    # echo $i
    datasets+=("msd")
    tasks+=("Task08")
    no_agents+=("agent")
done

# echo $datasets

export CUDA_VISIBLE_DEVICES=0

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
            $( [ ${no_agents[idx]} = "no_agent" ] && echo "-no_agent" )
            # -ablation \
            # -vis \
    done
done
