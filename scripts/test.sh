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
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-04-30-07-54-14/best.pth
    output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/best.pth

    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_0_dice0.5310.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_1_dice0.5239.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_2_dice0.5886.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_3_dice0.6238.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_4_dice0.5672.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_5_dice0.5735.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_6_dice0.6040.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_7_dice0.6165.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_8_dice0.5422.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_9_dice0.6655.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_10_dice0.6060.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_11_dice0.6227.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_12_dice0.6102.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_13_dice0.6404.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_14_dice0.6254.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_15_dice0.6067.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_16_dice0.6228.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_17_dice0.6439.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_18_dice0.6939.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_19_dice0.6633.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_20_dice0.6770.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_21_dice0.6673.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_22_dice0.7023.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_23_dice0.6842.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_24_dice0.7030.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_25_dice0.6952.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_26_dice0.6641.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_27_dice0.6834.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_28_dice0.6945.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_29_dice0.6945.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_30_dice0.6462.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_31_dice0.6755.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_32_dice0.7037.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_33_dice0.6853.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_34_dice0.6965.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_35_dice0.6814.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_36_dice0.6921.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_37_dice0.7198.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_38_dice0.6989.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_39_dice0.7395.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_40_dice0.7083.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_41_dice0.6498.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_42_dice0.6944.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_43_dice0.7328.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_44_dice0.7029.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_45_dice0.7398.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_46_dice0.7157.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_47_dice0.6938.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_48_dice0.7185.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_49_dice0.7113.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_50_dice0.7459.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_51_dice0.7021.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_52_dice0.7406.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_53_dice0.7246.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_54_dice0.7406.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_55_dice0.7105.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_56_dice0.7334.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_57_dice0.7413.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_58_dice0.7452.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_59_dice0.7044.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_60_dice0.7420.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_61_dice0.7405.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_62_dice0.7361.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_63_dice0.7412.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_64_dice0.7530.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_65_dice0.7417.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_66_dice0.7517.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_67_dice0.7126.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_68_dice0.7015.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_69_dice0.7144.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_70_dice0.7302.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_71_dice0.7305.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_72_dice0.6968.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_73_dice0.6880.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_74_dice0.7284.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_75_dice0.7575.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_76_dice0.7148.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_77_dice0.6761.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_78_dice0.7118.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_79_dice0.7536.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_80_dice0.7158.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_81_dice0.7281.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_82_dice0.7414.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_83_dice0.7175.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_84_dice0.7507.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_85_dice0.7212.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_86_dice0.7304.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_87_dice0.7128.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_88_dice0.7167.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_89_dice0.7053.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_90_dice0.7188.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_91_dice0.7089.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_92_dice0.7112.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_93_dice0.7502.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_94_dice0.7248.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_95_dice0.7347.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_96_dice0.7305.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_97_dice0.7467.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_98_dice0.7347.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_99_dice0.7258.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_100_dice0.7304.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_101_dice0.7325.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_102_dice0.7447.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_103_dice0.7433.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_104_dice0.7445.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_105_dice0.7565.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_106_dice0.7393.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_107_dice0.7115.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_108_dice0.7436.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_109_dice0.7500.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_110_dice0.7622.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_111_dice0.7384.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_112_dice0.7512.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_113_dice0.7313.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_114_dice0.7282.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_115_dice0.7383.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_116_dice0.7231.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_117_dice0.7569.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_118_dice0.7410.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_119_dice0.7753.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_120_dice0.7437.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_121_dice0.7321.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_122_dice0.7543.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_123_dice0.7381.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_124_dice0.7174.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_125_dice0.7388.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_126_dice0.7360.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_127_dice0.7515.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_128_dice0.7469.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_129_dice0.7239.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_130_dice0.7409.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_131_dice0.7318.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_132_dice0.7250.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_133_dice0.7306.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_134_dice0.7287.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_135_dice0.7348.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_136_dice0.7197.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_137_dice0.7625.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_138_dice0.7450.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_139_dice0.7428.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_140_dice0.7245.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_141_dice0.7294.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_142_dice0.7407.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_143_dice0.7403.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_144_dice0.7279.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_145_dice0.7252.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_146_dice0.7340.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_147_dice0.7250.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_148_dice0.7253.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_149_dice0.7470.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_150_dice0.7448.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_151_dice0.7268.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_152_dice0.7347.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_153_dice0.7399.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_154_dice0.7356.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_155_dice0.7376.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_156_dice0.7465.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_157_dice0.7364.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_158_dice0.7289.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_159_dice0.7690.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_160_dice0.7257.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_161_dice0.7367.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_162_dice0.7336.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_163_dice0.7416.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_164_dice0.7590.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_165_dice0.7540.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_166_dice0.7554.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_167_dice0.7502.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_168_dice0.7285.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_169_dice0.7446.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_170_dice0.7331.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_171_dice0.7443.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_172_dice0.7313.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_173_dice0.7503.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_174_dice0.7443.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_175_dice0.7541.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_176_dice0.7287.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_177_dice0.7409.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_178_dice0.7542.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_179_dice0.7395.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_180_dice0.7514.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_181_dice0.7309.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_182_dice0.7331.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_183_dice0.7525.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_184_dice0.7473.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_185_dice0.7423.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_186_dice0.7177.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_187_dice0.7440.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_188_dice0.7590.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_189_dice0.7321.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_190_dice0.7374.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_191_dice0.7484.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_192_dice0.7558.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_193_dice0.7672.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_194_dice0.7360.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_195_dice0.7475.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_196_dice0.7383.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_197_dice0.7277.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_198_dice0.7258.pth
    # output/btcv+icl+grpo+long_horizon+no_augment/2026-05-01-15-28-59/epoch_199_dice0.7296.pth

    # Sarcoma
    output/sarcoma+icl+grpo+long_horizon+no_augment/2026-04-29-14-22-57/best.pth
    # output/sarcoma+icl+no_agent+long_horizon+no_augment/2026-04-10-19-47-31/best.pth

    # MSD Heart
    # output/msd_task02+icl+no_agent+long_horizon+no_augment/2026-04-15-19-42-29/best.pth
    # output/msd_task02+icl+grpo+lazy_pen1.0+long_horizon+flip_augment/2026-04-23-16-01-08/best.pth
    # output/msd_task02+icl+grpo+long_horizon+no_augment/2026-04-30-18-07-03/best.pth
    output/msd_task02+icl+grpo+long_horizon+no_augment+lr2e-4/2026-05-01-07-12-31/best.pth

    # MSD Liver
    
    
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

declare -a datasets=(
    btcv
)

declare -a tasks=(
    ""
)

declare -a no_agents=()

for i in ${!ckpts[@]};
do
    # echo $i
    # datasets+=("msd")
    # tasks+=("Task09")
    no_agents+=("agent")
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
            $( [ ${no_agents[idx]} = "no_agent" ] && echo "-no_agent" )
            # -ablation \
            # -vis \
    done
done
