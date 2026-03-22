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

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    # output/msd_task10+grpo+icl/2026-03-10-19-22-49/best.pth

    # NOTE: TESTING
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_0_dice0.4846.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_1_dice0.4808.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_2_dice0.5445.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_3_dice0.5497.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_4_dice0.6039.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_5_dice0.6060.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_6_dice0.6160.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_7_dice0.5497.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_8_dice0.5602.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_9_dice0.6490.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_10_dice0.6108.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_11_dice0.6292.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_12_dice0.6123.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_13_dice0.6184.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_14_dice0.6179.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_15_dice0.6450.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_16_dice0.6467.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_17_dice0.6768.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_18_dice0.6658.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_19_dice0.6739.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_20_dice0.6875.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_21_dice0.6396.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_22_dice0.6455.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_23_dice0.6621.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_24_dice0.6449.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_25_dice0.6777.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_26_dice0.6804.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_27_dice0.7118.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_28_dice0.6754.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_29_dice0.6871.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_30_dice0.6700.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_31_dice0.6786.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_32_dice0.6735.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_33_dice0.6939.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_34_dice0.6694.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_35_dice0.7147.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_36_dice0.6786.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_37_dice0.6969.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_38_dice0.6793.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_39_dice0.6685.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_40_dice0.7186.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_41_dice0.7041.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_42_dice0.7106.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_43_dice0.7023.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_44_dice0.7049.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_45_dice0.7012.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_46_dice0.7192.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_47_dice0.6962.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_48_dice0.7030.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_49_dice0.6745.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_50_dice0.6911.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_51_dice0.7265.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_52_dice0.7156.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_53_dice0.6912.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_54_dice0.7042.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_55_dice0.7231.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_56_dice0.6973.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_57_dice0.6798.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_58_dice0.7135.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_59_dice0.7092.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_60_dice0.6851.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_61_dice0.6717.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_62_dice0.6959.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_63_dice0.7036.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_64_dice0.6832.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_65_dice0.6712.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_66_dice0.6611.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_67_dice0.7274.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_68_dice0.7168.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_69_dice0.7044.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_70_dice0.6968.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_71_dice0.6864.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_72_dice0.7338.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_73_dice0.7417.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_74_dice0.6787.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_75_dice0.7119.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_76_dice0.7050.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_77_dice0.7326.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_78_dice0.7366.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_79_dice0.7213.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_80_dice0.7368.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_81_dice0.7168.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_82_dice0.7017.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_83_dice0.7227.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_84_dice0.7171.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_85_dice0.7241.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_86_dice0.6796.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_87_dice0.7077.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_88_dice0.7217.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_89_dice0.6915.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_90_dice0.7002.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_91_dice0.7428.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_92_dice0.7352.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_93_dice0.7202.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_94_dice0.7472.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_95_dice0.7185.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_96_dice0.7121.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_97_dice0.7170.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_98_dice0.7239.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_99_dice0.7215.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_100_dice0.6924.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_101_dice0.7091.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_102_dice0.6854.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_103_dice0.7163.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_104_dice0.6808.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_105_dice0.7157.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_106_dice0.7258.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_107_dice0.6978.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_108_dice0.7178.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_109_dice0.6767.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_110_dice0.7382.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_111_dice0.7147.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_112_dice0.7136.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_113_dice0.7085.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_114_dice0.7185.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_115_dice0.7156.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_116_dice0.7341.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_117_dice0.7148.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_118_dice0.7204.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_119_dice0.7471.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_120_dice0.7188.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_121_dice0.6868.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_122_dice0.7312.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_123_dice0.7568.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_124_dice0.7082.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_125_dice0.7126.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_126_dice0.7085.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_127_dice0.7397.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_128_dice0.7208.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_129_dice0.7182.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_130_dice0.7148.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_131_dice0.7231.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_132_dice0.7479.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_133_dice0.7283.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_134_dice0.7163.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_135_dice0.7187.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_136_dice0.7128.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_137_dice0.7097.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_138_dice0.6954.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_139_dice0.7438.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_140_dice0.7252.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_141_dice0.7297.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_142_dice0.7438.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_143_dice0.7632.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_144_dice0.7370.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_145_dice0.7249.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_146_dice0.7156.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_147_dice0.7108.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_148_dice0.6989.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_149_dice0.7076.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_150_dice0.7322.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_151_dice0.7215.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_152_dice0.7423.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_153_dice0.6861.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_154_dice0.7422.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_155_dice0.7252.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_156_dice0.6796.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_157_dice0.7382.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_158_dice0.7059.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_159_dice0.7109.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_160_dice0.7410.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_161_dice0.7288.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_162_dice0.7137.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_163_dice0.7266.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_164_dice0.7544.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_165_dice0.7082.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_166_dice0.7270.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_167_dice0.6737.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_168_dice0.7165.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_169_dice0.7355.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_170_dice0.7399.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_171_dice0.7087.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_172_dice0.6915.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_173_dice0.7358.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_174_dice0.7114.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_175_dice0.7035.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_176_dice0.6809.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_177_dice0.7201.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_178_dice0.7251.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_179_dice0.7327.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_180_dice0.7329.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_181_dice0.7123.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_182_dice0.7372.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_183_dice0.6934.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_184_dice0.7329.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_185_dice0.7505.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_186_dice0.6719.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_187_dice0.7054.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_188_dice0.7474.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_189_dice0.7281.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_190_dice0.7351.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_191_dice0.6883.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_192_dice0.7446.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_193_dice0.7032.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_194_dice0.7145.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_195_dice0.7232.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_196_dice0.7098.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_197_dice0.7231.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_198_dice0.7361.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_199_dice0.7212.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_200_dice0.7140.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_201_dice0.7425.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_202_dice0.7168.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_203_dice0.7425.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_204_dice0.7417.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_205_dice0.7494.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_206_dice0.6790.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_207_dice0.6807.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_208_dice0.7218.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_209_dice0.7216.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_210_dice0.7226.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_211_dice0.7024.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_212_dice0.7100.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_213_dice0.6952.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_214_dice0.7222.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_215_dice0.7374.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_216_dice0.7321.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_217_dice0.7241.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_218_dice0.7480.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_219_dice0.7365.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_220_dice0.7165.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_221_dice0.7026.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_222_dice0.7436.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_223_dice0.7476.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_224_dice0.7135.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_225_dice0.7166.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_226_dice0.7483.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_227_dice0.6957.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_228_dice0.7232.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_229_dice0.6973.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_230_dice0.7221.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_231_dice0.7282.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_232_dice0.7132.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_233_dice0.7179.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_234_dice0.7367.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_235_dice0.7208.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_236_dice0.7188.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_237_dice0.7449.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_238_dice0.7104.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_239_dice0.7275.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_240_dice0.7136.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_241_dice0.6976.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_242_dice0.6892.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_243_dice0.7328.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_244_dice0.6720.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_245_dice0.7313.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_246_dice0.7204.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_247_dice0.7371.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_248_dice0.7193.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_249_dice0.7576.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_250_dice0.7367.pth
    output/btcv+grpo+icl+cw_gating_before_pos+semantic_filtering/2026-03-21-13-27-40/epoch_251_dice0.7444.pth
)

# export CUDA_VISIBLE_DEVICES=0

for idx in ${!ckpt[@]}
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain ${ckpt[idx]} \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset btcv \
            -task "" \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/BTCV \
            -num_support $shot
            # -no_agent
    done
done
