#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=24:00:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-task01-%j.out"

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    # output/msd_task10+grpo+icl/2026-03-10-19-22-49/best.pth

    # NOTE: TESTING
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_0_dice0.4249.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_1_dice0.4652.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_2_dice0.4938.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_3_dice0.4600.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_4_dice0.4405.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_5_dice0.4635.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_6_dice0.4845.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_7_dice0.4867.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_8_dice0.5032.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_9_dice0.5044.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_10_dice0.4980.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_11_dice0.4831.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_12_dice0.5159.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_13_dice0.4933.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_14_dice0.4748.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_15_dice0.5007.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_16_dice0.4975.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_17_dice0.5113.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_18_dice0.5078.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_19_dice0.5020.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_20_dice0.5046.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_21_dice0.5290.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_22_dice0.5069.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_23_dice0.5043.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_24_dice0.5110.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_25_dice0.5062.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_26_dice0.5154.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_27_dice0.5224.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_28_dice0.5077.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_29_dice0.5231.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_30_dice0.5255.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_31_dice0.5055.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_32_dice0.5138.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_33_dice0.4968.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_34_dice0.5117.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_35_dice0.5142.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_36_dice0.5102.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_37_dice0.5258.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_38_dice0.5223.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_39_dice0.5238.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_40_dice0.5145.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_41_dice0.5215.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_42_dice0.5050.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_43_dice0.5273.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_44_dice0.5245.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_45_dice0.5102.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_46_dice0.5177.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_47_dice0.5275.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_48_dice0.5296.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_49_dice0.5221.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_50_dice0.5271.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_51_dice0.5303.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_52_dice0.5292.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_53_dice0.5013.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_54_dice0.4954.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_55_dice0.5347.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_56_dice0.5393.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_57_dice0.5325.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_58_dice0.5190.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_59_dice0.5256.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_60_dice0.5285.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_61_dice0.5021.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_62_dice0.5105.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_63_dice0.5203.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_64_dice0.5152.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_65_dice0.5297.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_66_dice0.5319.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_67_dice0.5077.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_68_dice0.5282.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_69_dice0.5279.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_70_dice0.5250.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_71_dice0.5226.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_72_dice0.5216.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_73_dice0.5301.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_74_dice0.5306.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_75_dice0.5243.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_76_dice0.5214.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_77_dice0.5328.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_78_dice0.5241.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_79_dice0.5257.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_80_dice0.5203.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_81_dice0.5216.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_82_dice0.5305.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_83_dice0.5310.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_84_dice0.5322.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_85_dice0.5241.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_86_dice0.5416.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_87_dice0.5301.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_88_dice0.5267.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_89_dice0.5282.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_90_dice0.5047.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_91_dice0.5336.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_92_dice0.5222.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_93_dice0.5390.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_94_dice0.5301.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_95_dice0.5226.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_96_dice0.5306.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_97_dice0.5270.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_98_dice0.5297.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_99_dice0.5289.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_100_dice0.5338.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_101_dice0.5373.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_102_dice0.5235.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_103_dice0.5315.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_104_dice0.5235.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_105_dice0.5177.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_106_dice0.5177.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_107_dice0.5288.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_108_dice0.5243.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_109_dice0.5235.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_110_dice0.5225.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_111_dice0.5232.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_112_dice0.5307.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_113_dice0.5230.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_114_dice0.5294.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_115_dice0.5231.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_116_dice0.5297.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_117_dice0.5301.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_118_dice0.5298.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_119_dice0.5265.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_120_dice0.5266.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_121_dice0.5283.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_122_dice0.5299.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_123_dice0.5317.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_124_dice0.5277.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_125_dice0.5249.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_126_dice0.5262.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_127_dice0.5307.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_128_dice0.5276.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_129_dice0.5299.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_130_dice0.5150.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_131_dice0.5272.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_132_dice0.5261.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_133_dice0.5262.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_134_dice0.5260.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_135_dice0.5254.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_136_dice0.5165.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_137_dice0.5264.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_138_dice0.5260.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_139_dice0.5262.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_140_dice0.5283.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_141_dice0.5219.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_142_dice0.5244.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_143_dice0.5212.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_144_dice0.5158.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_145_dice0.5247.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_146_dice0.5255.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_147_dice0.5230.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_148_dice0.5192.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_149_dice0.5262.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_150_dice0.5247.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_151_dice0.5307.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_152_dice0.5182.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_153_dice0.5192.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_154_dice0.5217.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_155_dice0.5220.pth
    output/msd_task01+grpo+icl+cw_gating+semantic_filtering/2026-03-17-17-26-01/epoch_156_dice0.5237.pth
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
            -task "Task01" \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot
            # -no_agent
    done
done
