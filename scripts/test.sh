#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=24:00:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-task03-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    # output/msd_task10+grpo+icl/2026-03-10-19-22-49/best.pth

    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/best.pth
    # NOTE: TESTING
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_0_dice0.3980.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_1_dice0.4014.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_2_dice0.3287.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_3_dice0.4745.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_4_dice0.5082.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_5_dice0.4836.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_6_dice0.5208.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_7_dice0.4804.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_8_dice0.5734.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_9_dice0.5655.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_10_dice0.5468.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_11_dice0.5714.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_12_dice0.5924.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_13_dice0.5833.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_14_dice0.5869.pth
    # output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_15_dice0.5929.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_16_dice0.6071.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_17_dice0.6124.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_18_dice0.6044.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_19_dice0.6048.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_20_dice0.6250.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_21_dice0.5992.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_22_dice0.5993.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_23_dice0.5896.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_24_dice0.6089.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_25_dice0.5962.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_26_dice0.6135.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_27_dice0.6121.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_28_dice0.6321.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_29_dice0.6029.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_30_dice0.6343.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_31_dice0.6310.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_32_dice0.6315.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_33_dice0.6305.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_34_dice0.6040.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_35_dice0.6127.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_36_dice0.6090.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_37_dice0.6227.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_38_dice0.6382.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_39_dice0.6371.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_40_dice0.6312.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_41_dice0.6347.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_42_dice0.6258.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_43_dice0.6367.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_44_dice0.6419.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_45_dice0.6240.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_46_dice0.6356.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_47_dice0.6621.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_48_dice0.6500.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_49_dice0.6273.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_50_dice0.6390.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_51_dice0.6462.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_52_dice0.6263.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_53_dice0.6523.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_54_dice0.6404.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_55_dice0.6395.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_56_dice0.6305.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_57_dice0.6032.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_58_dice0.6541.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_59_dice0.6242.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_60_dice0.6362.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_61_dice0.6469.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_62_dice0.6340.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_63_dice0.6420.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_64_dice0.6307.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_65_dice0.6382.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_66_dice0.6389.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_67_dice0.6450.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_68_dice0.6503.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_69_dice0.6343.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_70_dice0.6488.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_71_dice0.6659.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_72_dice0.6486.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_73_dice0.6461.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_74_dice0.6458.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_75_dice0.6606.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_76_dice0.6254.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_77_dice0.6196.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_78_dice0.6349.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_79_dice0.6352.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_80_dice0.6431.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_81_dice0.6389.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_82_dice0.6198.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_83_dice0.6205.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_84_dice0.6462.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_85_dice0.6098.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_86_dice0.6425.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_87_dice0.6434.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_88_dice0.6442.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_89_dice0.6401.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_90_dice0.6121.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_91_dice0.6167.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_92_dice0.6452.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_93_dice0.6199.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_94_dice0.6407.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_95_dice0.6487.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_96_dice0.5953.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_97_dice0.6483.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_98_dice0.6412.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_99_dice0.6404.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_100_dice0.6422.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_101_dice0.6239.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_102_dice0.6470.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_103_dice0.6252.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_104_dice0.6434.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_105_dice0.6511.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_106_dice0.6349.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_107_dice0.6327.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_108_dice0.6290.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_109_dice0.6357.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_110_dice0.6347.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_111_dice0.6393.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_112_dice0.6296.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_113_dice0.6464.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_114_dice0.6358.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_115_dice0.6400.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_116_dice0.6459.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_117_dice0.6138.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_118_dice0.6408.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_119_dice0.6224.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_120_dice0.5953.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_121_dice0.6585.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_122_dice0.6336.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_123_dice0.6144.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_124_dice0.6294.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_125_dice0.6351.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_126_dice0.6445.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_127_dice0.6421.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_128_dice0.6331.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_129_dice0.6333.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_130_dice0.6652.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_131_dice0.6358.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_132_dice0.6437.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_133_dice0.6488.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_134_dice0.6378.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_135_dice0.5992.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_136_dice0.6419.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_137_dice0.6320.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_138_dice0.6413.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_139_dice0.6421.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_140_dice0.6451.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_141_dice0.6380.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_142_dice0.6440.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_143_dice0.6422.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_144_dice0.6152.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_145_dice0.6444.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_146_dice0.6091.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_147_dice0.6247.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_148_dice0.6461.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_149_dice0.6245.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_150_dice0.6486.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_151_dice0.6406.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_152_dice0.6352.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_153_dice0.6412.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_154_dice0.6416.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_155_dice0.6461.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_156_dice0.6328.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_157_dice0.6123.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_158_dice0.6695.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_159_dice0.6430.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_160_dice0.6387.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_161_dice0.6384.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_162_dice0.6142.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_163_dice0.6485.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_164_dice0.6353.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_165_dice0.6362.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_166_dice0.6492.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_167_dice0.6366.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_168_dice0.6150.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_169_dice0.6224.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_170_dice0.6290.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_171_dice0.6360.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_172_dice0.6397.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_173_dice0.6409.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_174_dice0.6375.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_175_dice0.6031.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_176_dice0.6369.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_177_dice0.6417.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_178_dice0.6374.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_179_dice0.6393.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_180_dice0.6423.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_181_dice0.6389.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_182_dice0.6392.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_183_dice0.6298.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_184_dice0.6456.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_185_dice0.6336.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_186_dice0.6406.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_187_dice0.6177.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_188_dice0.6384.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_189_dice0.6332.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_190_dice0.6415.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_191_dice0.6170.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_192_dice0.6274.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_193_dice0.6393.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_194_dice0.6236.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_195_dice0.6373.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_196_dice0.6447.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_197_dice0.6429.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_198_dice0.6135.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_199_dice0.6309.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_200_dice0.6582.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_201_dice0.6404.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_202_dice0.6429.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_203_dice0.6185.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_204_dice0.6243.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_205_dice0.6214.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_206_dice0.6602.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_207_dice0.6416.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_208_dice0.6439.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_209_dice0.6234.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_210_dice0.6101.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_211_dice0.6367.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_212_dice0.6366.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_213_dice0.6145.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_214_dice0.6107.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_215_dice0.6440.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_216_dice0.6350.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_217_dice0.6302.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_218_dice0.6522.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_219_dice0.6164.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_220_dice0.6220.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_221_dice0.6444.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_222_dice0.6129.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_223_dice0.6434.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_224_dice0.6404.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_225_dice0.6437.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_226_dice0.6382.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_227_dice0.6107.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_228_dice0.6167.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_229_dice0.6428.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_230_dice0.6429.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_231_dice0.6445.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_232_dice0.6403.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_233_dice0.6427.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_234_dice0.6247.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_235_dice0.6556.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_236_dice0.6070.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_237_dice0.6390.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_238_dice0.6420.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_239_dice0.6143.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_240_dice0.6409.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_241_dice0.6340.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_242_dice0.6388.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_243_dice0.6277.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_244_dice0.6067.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_245_dice0.6149.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_246_dice0.6330.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_247_dice0.6418.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_248_dice0.6377.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_249_dice0.6109.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_250_dice0.5728.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_251_dice0.6386.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_252_dice0.6370.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_253_dice0.6384.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_254_dice0.6149.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_255_dice0.6421.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_256_dice0.6427.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_257_dice0.6092.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_258_dice0.6156.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_259_dice0.6609.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_260_dice0.6384.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_261_dice0.6416.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_262_dice0.6342.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_263_dice0.6080.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_264_dice0.6311.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_265_dice0.6414.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_266_dice0.6397.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_267_dice0.6413.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_268_dice0.6359.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_269_dice0.6301.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_270_dice0.6359.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_271_dice0.6442.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_272_dice0.6454.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_273_dice0.6179.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_274_dice0.6118.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_275_dice0.6151.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_276_dice0.6388.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_277_dice0.6439.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_278_dice0.6174.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_279_dice0.6130.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_280_dice0.6413.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_281_dice0.6374.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_282_dice0.6330.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_283_dice0.6372.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_284_dice0.6418.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_285_dice0.6302.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_286_dice0.6346.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_287_dice0.6413.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_288_dice0.6347.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_289_dice0.6321.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_290_dice0.6353.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_291_dice0.6387.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_292_dice0.6403.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_293_dice0.6370.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_294_dice0.6143.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_295_dice0.6346.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_296_dice0.6330.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_297_dice0.6223.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_298_dice0.6247.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_299_dice0.6328.pth
    output/msd_task03+grpo+icl+include_cond_ptr/2026-03-15-15-42-08/epoch_300_dice0.6149.pth
)

# export CUDA_VISIBLE_DEVICES=0

for idx in ${!ckpt[@]}
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain ${ckpt[idx]} \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "Task03" \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot
            # -no_agent
    done
done
