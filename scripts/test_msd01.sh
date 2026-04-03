#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=12:00:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-task03-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

DIR=/data/rlsam2/msd01/DQN-MedSAM2/output/msd_task01+grpo+icl+cw_soft_gating+obj_ptr_gating+highres_gating_by_lowres/2026-03-31-10-03-47

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    # output/msd_task10+grpo+icl/2026-03-10-19-22-49/best.pth

    # NOTE: TESTING
    ${DIR}/epoch_0_dice0.3956.pth
    ${DIR}/epoch_1_dice0.4033.pth
    ${DIR}/epoch_2_dice0.3741.pth
    ${DIR}/epoch_3_dice0.4373.pth
    ${DIR}/epoch_4_dice0.4812.pth
    ${DIR}/epoch_5_dice0.4340.pth
    ${DIR}/epoch_6_dice0.4388.pth
    ${DIR}/epoch_7_dice0.4830.pth
    ${DIR}/epoch_8_dice0.4827.pth
    ${DIR}/epoch_9_dice0.4309.pth
    ${DIR}/epoch_10_dice0.4485.pth
    ${DIR}/epoch_11_dice0.4801.pth
    ${DIR}/epoch_12_dice0.4634.pth
    ${DIR}/epoch_13_dice0.4899.pth
    ${DIR}/epoch_14_dice0.4726.pth
    ${DIR}/epoch_15_dice0.4993.pth
    ${DIR}/epoch_16_dice0.4913.pth
    ${DIR}/epoch_17_dice0.4987.pth
    ${DIR}/epoch_18_dice0.5017.pth
    ${DIR}/epoch_19_dice0.5023.pth
    ${DIR}/epoch_20_dice0.4954.pth
    ${DIR}/epoch_21_dice0.5077.pth
    ${DIR}/epoch_22_dice0.4839.pth
    ${DIR}/epoch_23_dice0.5120.pth
    ${DIR}/epoch_24_dice0.4924.pth
    ${DIR}/epoch_25_dice0.4988.pth
    ${DIR}/epoch_26_dice0.5205.pth
    ${DIR}/epoch_27_dice0.5210.pth
    ${DIR}/epoch_28_dice0.5180.pth
    ${DIR}/epoch_29_dice0.5155.pth
    ${DIR}/epoch_30_dice0.5227.pth
    ${DIR}/epoch_31_dice0.5158.pth
    ${DIR}/epoch_32_dice0.5095.pth
    ${DIR}/epoch_33_dice0.5091.pth
    ${DIR}/epoch_34_dice0.5163.pth
    ${DIR}/epoch_35_dice0.5259.pth
    ${DIR}/epoch_36_dice0.5234.pth
    ${DIR}/epoch_37_dice0.5218.pth
    ${DIR}/epoch_38_dice0.5212.pth
    ${DIR}/epoch_39_dice0.5248.pth
    ${DIR}/epoch_40_dice0.5268.pth
    ${DIR}/epoch_41_dice0.5277.pth
    ${DIR}/epoch_42_dice0.5266.pth
    ${DIR}/epoch_43_dice0.5153.pth
    ${DIR}/epoch_44_dice0.5156.pth
    ${DIR}/epoch_45_dice0.5225.pth
    ${DIR}/epoch_46_dice0.5343.pth
    ${DIR}/epoch_47_dice0.5129.pth
    ${DIR}/epoch_48_dice0.5261.pth
    ${DIR}/epoch_49_dice0.5330.pth
    ${DIR}/epoch_50_dice0.5332.pth
    ${DIR}/epoch_51_dice0.5163.pth
    ${DIR}/epoch_52_dice0.5167.pth
    ${DIR}/epoch_53_dice0.5267.pth
    ${DIR}/epoch_54_dice0.5204.pth
    ${DIR}/epoch_55_dice0.5255.pth
    ${DIR}/epoch_56_dice0.5141.pth
    ${DIR}/epoch_57_dice0.5218.pth
    ${DIR}/epoch_58_dice0.5240.pth
    ${DIR}/epoch_59_dice0.5276.pth
    ${DIR}/epoch_60_dice0.5314.pth
    ${DIR}/epoch_61_dice0.5268.pth
    ${DIR}/epoch_62_dice0.5147.pth
    ${DIR}/epoch_63_dice0.5227.pth
    ${DIR}/epoch_64_dice0.5262.pth
    ${DIR}/epoch_65_dice0.5354.pth
    ${DIR}/epoch_66_dice0.5325.pth
    ${DIR}/epoch_67_dice0.5227.pth
    ${DIR}/epoch_68_dice0.5306.pth
    ${DIR}/epoch_69_dice0.5207.pth
    ${DIR}/epoch_70_dice0.5264.pth
    ${DIR}/epoch_71_dice0.5192.pth
    ${DIR}/epoch_72_dice0.5289.pth
    ${DIR}/epoch_73_dice0.5176.pth
    ${DIR}/epoch_74_dice0.5274.pth
    ${DIR}/epoch_75_dice0.5293.pth
    ${DIR}/epoch_76_dice0.5271.pth
    ${DIR}/epoch_77_dice0.5195.pth
    ${DIR}/epoch_78_dice0.5329.pth
    ${DIR}/epoch_79_dice0.5365.pth
    ${DIR}/epoch_80_dice0.5121.pth
    ${DIR}/epoch_81_dice0.5423.pth
    ${DIR}/epoch_82_dice0.5222.pth
    ${DIR}/epoch_83_dice0.5408.pth
    ${DIR}/epoch_84_dice0.5364.pth
    ${DIR}/epoch_85_dice0.5312.pth
    ${DIR}/epoch_86_dice0.5246.pth
    ${DIR}/epoch_87_dice0.5243.pth
    ${DIR}/epoch_88_dice0.5326.pth
    ${DIR}/epoch_89_dice0.5433.pth
    ${DIR}/epoch_90_dice0.5363.pth
    ${DIR}/epoch_91_dice0.5208.pth
    ${DIR}/epoch_92_dice0.5257.pth
    ${DIR}/epoch_93_dice0.5227.pth
    ${DIR}/epoch_94_dice0.5358.pth
    ${DIR}/epoch_95_dice0.5310.pth
    ${DIR}/epoch_96_dice0.5329.pth
    ${DIR}/epoch_97_dice0.5231.pth
    ${DIR}/epoch_98_dice0.5185.pth
    ${DIR}/epoch_99_dice0.5336.pth
    ${DIR}/epoch_100_dice0.5404.pth
    ${DIR}/epoch_101_dice0.5358.pth
    ${DIR}/epoch_102_dice0.5139.pth
    ${DIR}/epoch_103_dice0.5322.pth
    ${DIR}/epoch_104_dice0.5367.pth
    ${DIR}/epoch_105_dice0.5332.pth
    ${DIR}/epoch_106_dice0.5397.pth
    ${DIR}/epoch_107_dice0.5365.pth
    ${DIR}/epoch_108_dice0.5334.pth
    ${DIR}/epoch_109_dice0.5386.pth
    ${DIR}/epoch_110_dice0.5306.pth
    ${DIR}/epoch_111_dice0.5371.pth
    ${DIR}/epoch_112_dice0.5414.pth
    ${DIR}/epoch_113_dice0.5330.pth
    ${DIR}/epoch_114_dice0.5285.pth
    ${DIR}/epoch_115_dice0.5362.pth
    ${DIR}/epoch_116_dice0.5178.pth
    ${DIR}/epoch_117_dice0.5413.pth
    ${DIR}/epoch_118_dice0.5410.pth
    ${DIR}/epoch_119_dice0.5351.pth
    ${DIR}/epoch_120_dice0.5440.pth
    ${DIR}/epoch_121_dice0.5301.pth
    ${DIR}/epoch_122_dice0.5460.pth
    ${DIR}/epoch_123_dice0.5381.pth
    ${DIR}/epoch_124_dice0.5485.pth
    ${DIR}/epoch_125_dice0.5409.pth
    ${DIR}/epoch_126_dice0.5435.pth
    ${DIR}/epoch_127_dice0.5496.pth
    ${DIR}/epoch_128_dice0.5416.pth
    ${DIR}/epoch_129_dice0.5471.pth
    ${DIR}/epoch_130_dice0.5432.pth
    ${DIR}/epoch_131_dice0.5440.pth
    ${DIR}/epoch_132_dice0.5419.pth
    ${DIR}/epoch_133_dice0.5495.pth
    ${DIR}/epoch_134_dice0.5459.pth
    ${DIR}/epoch_135_dice0.5306.pth
    ${DIR}/epoch_136_dice0.5385.pth
    ${DIR}/epoch_137_dice0.5495.pth
    ${DIR}/epoch_138_dice0.5392.pth
    ${DIR}/epoch_139_dice0.5380.pth
    ${DIR}/epoch_140_dice0.5477.pth
    ${DIR}/epoch_141_dice0.5457.pth
    ${DIR}/epoch_142_dice0.5361.pth
    ${DIR}/epoch_143_dice0.5278.pth
    ${DIR}/epoch_144_dice0.5464.pth
    ${DIR}/epoch_145_dice0.5402.pth
    ${DIR}/epoch_146_dice0.5384.pth
    ${DIR}/epoch_147_dice0.5356.pth
    ${DIR}/epoch_148_dice0.5519.pth
    ${DIR}/epoch_149_dice0.5387.pth
    ${DIR}/epoch_150_dice0.5449.pth
    ${DIR}/epoch_151_dice0.5431.pth
    ${DIR}/epoch_152_dice0.5446.pth
    ${DIR}/epoch_153_dice0.5311.pth
    ${DIR}/epoch_154_dice0.5344.pth
    ${DIR}/epoch_155_dice0.5378.pth
    ${DIR}/epoch_156_dice0.5423.pth
    ${DIR}/epoch_157_dice0.5452.pth
    ${DIR}/epoch_158_dice0.5314.pth
    ${DIR}/epoch_159_dice0.5463.pth
    ${DIR}/epoch_160_dice0.5443.pth
    ${DIR}/epoch_161_dice0.5386.pth
    ${DIR}/epoch_162_dice0.5418.pth
    ${DIR}/epoch_163_dice0.5517.pth
    ${DIR}/epoch_164_dice0.5341.pth
    ${DIR}/epoch_165_dice0.5399.pth
    ${DIR}/epoch_166_dice0.5380.pth
    ${DIR}/epoch_167_dice0.5466.pth
    ${DIR}/epoch_168_dice0.5457.pth
    ${DIR}/epoch_169_dice0.5363.pth
    ${DIR}/epoch_170_dice0.5400.pth
    ${DIR}/epoch_171_dice0.5379.pth
    ${DIR}/epoch_172_dice0.5310.pth
    ${DIR}/epoch_173_dice0.5374.pth
    ${DIR}/epoch_174_dice0.5411.pth
    ${DIR}/epoch_175_dice0.5477.pth
    ${DIR}/epoch_176_dice0.5453.pth
    ${DIR}/epoch_177_dice0.5426.pth
    ${DIR}/epoch_178_dice0.5460.pth
    ${DIR}/epoch_179_dice0.5492.pth
    ${DIR}/epoch_180_dice0.5480.pth
    ${DIR}/epoch_181_dice0.5339.pth
    ${DIR}/epoch_182_dice0.5344.pth
    ${DIR}/epoch_183_dice0.5444.pth
    ${DIR}/epoch_184_dice0.5410.pth
    ${DIR}/epoch_185_dice0.5442.pth
    ${DIR}/epoch_186_dice0.5460.pth
    ${DIR}/epoch_187_dice0.5451.pth
    ${DIR}/epoch_188_dice0.5392.pth
    ${DIR}/epoch_189_dice0.5452.pth
    ${DIR}/epoch_190_dice0.5452.pth
    ${DIR}/epoch_191_dice0.5445.pth
    ${DIR}/epoch_192_dice0.5408.pth
    ${DIR}/epoch_193_dice0.5489.pth
    ${DIR}/epoch_194_dice0.5439.pth
    ${DIR}/epoch_195_dice0.5370.pth
    ${DIR}/epoch_196_dice0.5338.pth
    ${DIR}/epoch_197_dice0.5428.pth
    ${DIR}/epoch_198_dice0.5394.pth
    ${DIR}/epoch_199_dice0.5308.pth
    ${DIR}/epoch_200_dice0.5359.pth
    ${DIR}/epoch_201_dice0.5387.pth
    ${DIR}/epoch_202_dice0.5380.pth
    ${DIR}/epoch_203_dice0.5433.pth
    ${DIR}/epoch_204_dice0.5412.pth
    ${DIR}/epoch_205_dice0.5304.pth
    ${DIR}/epoch_206_dice0.5399.pth
    ${DIR}/epoch_207_dice0.5413.pth
    ${DIR}/epoch_208_dice0.5427.pth
    ${DIR}/epoch_209_dice0.5467.pth
    ${DIR}/epoch_210_dice0.5465.pth
    ${DIR}/epoch_211_dice0.5426.pth
    ${DIR}/epoch_212_dice0.5369.pth
    ${DIR}/epoch_213_dice0.5303.pth
    ${DIR}/epoch_214_dice0.5518.pth
    ${DIR}/epoch_215_dice0.5423.pth
    ${DIR}/epoch_216_dice0.5344.pth
    ${DIR}/epoch_217_dice0.5387.pth
    ${DIR}/epoch_218_dice0.5446.pth
    ${DIR}/epoch_219_dice0.5317.pth
    ${DIR}/epoch_220_dice0.5432.pth
    ${DIR}/epoch_221_dice0.5404.pth
    ${DIR}/epoch_222_dice0.5411.pth
    ${DIR}/epoch_223_dice0.5317.pth
    ${DIR}/epoch_224_dice0.5351.pth
    ${DIR}/epoch_225_dice0.5448.pth
    ${DIR}/epoch_226_dice0.5457.pth
    ${DIR}/epoch_227_dice0.5383.pth
    ${DIR}/epoch_228_dice0.5336.pth
    ${DIR}/epoch_229_dice0.5308.pth
    ${DIR}/epoch_230_dice0.5446.pth
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
            -task "Task01" \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot \
            -gating_dimension "cw" \
            -gating_softness "soft" \
            -obj_ptr_gating \
            -auxiliary_loss "no" \
            -highres_gating "no" \
            # -no_agent
    done
done
