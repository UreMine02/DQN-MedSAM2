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

    # Sarcoma
    # output/sarcoma+icl+grpo+long_horizon+no_augment/2026-04-29-14-22-57/best.pth
    # output/sarcoma+icl+no_agent+long_horizon+no_augment/2026-04-10-19-47-31/best.pth

    # MSD Heart
    # output/msd_task02+icl+no_agent+long_horizon+no_augment/2026-04-15-19-42-29/best.pth
    # output/msd_task02+icl+grpo+lazy_pen1.0+long_horizon+flip_augment/2026-04-23-16-01-08/best.pth
    # output/msd_task02+icl+grpo+long_horizon+no_augment/2026-04-30-18-07-03/best.pth
    # output/msd_task02+icl+grpo+long_horizon+no_augment+lr2e-4/2026-05-01-07-12-31/best.pth

    # MSD Liver
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_0_dice0.3693.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_1_dice0.4290.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_2_dice0.4723.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_3_dice0.5137.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_4_dice0.3241.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_5_dice0.5124.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_6_dice0.5775.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_7_dice0.5916.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_8_dice0.6122.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_9_dice0.5870.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_10_dice0.6190.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_11_dice0.6164.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_12_dice0.6245.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_13_dice0.6308.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_14_dice0.6181.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_15_dice0.6368.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_16_dice0.6301.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_17_dice0.6654.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_18_dice0.6402.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_19_dice0.6453.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_20_dice0.6375.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_21_dice0.6416.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_22_dice0.6413.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_23_dice0.6308.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_24_dice0.6372.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_25_dice0.6454.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_26_dice0.6417.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_27_dice0.6325.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_28_dice0.6434.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_29_dice0.6440.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_30_dice0.6409.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_31_dice0.6466.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_32_dice0.6465.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_33_dice0.6411.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_34_dice0.6426.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_35_dice0.6430.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_36_dice0.6456.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_37_dice0.6434.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_38_dice0.6455.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_39_dice0.6080.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_40_dice0.6391.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_41_dice0.6464.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_42_dice0.6472.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_43_dice0.6358.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_44_dice0.6444.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_45_dice0.6433.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_46_dice0.6374.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_47_dice0.6412.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_48_dice0.6387.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_49_dice0.6390.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_50_dice0.6389.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_51_dice0.6366.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_52_dice0.6409.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_53_dice0.6401.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_54_dice0.6329.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_55_dice0.6328.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_56_dice0.6400.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_57_dice0.6338.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_58_dice0.6367.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_59_dice0.6355.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_60_dice0.6360.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_61_dice0.6345.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_62_dice0.6330.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_63_dice0.6291.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_64_dice0.6416.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_65_dice0.6317.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_66_dice0.6187.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_67_dice0.6290.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_68_dice0.6083.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_69_dice0.6408.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_70_dice0.6373.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_71_dice0.6372.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_72_dice0.6388.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_73_dice0.6333.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_74_dice0.6410.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_75_dice0.6303.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_76_dice0.6334.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_77_dice0.6264.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_78_dice0.6361.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_79_dice0.6415.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_80_dice0.6382.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_81_dice0.6435.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_82_dice0.6348.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_83_dice0.6101.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_84_dice0.6328.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_85_dice0.6363.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_86_dice0.6395.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_87_dice0.6331.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_88_dice0.6338.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_89_dice0.6369.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_90_dice0.6385.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_91_dice0.6315.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_92_dice0.6368.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_93_dice0.6348.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_94_dice0.6347.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_95_dice0.6135.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_96_dice0.6347.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_97_dice0.6354.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_98_dice0.6298.pth
    output/msd_task03+icl+grpo+long_horizon+no_augment/2026-04-30-13-28-46/epoch_99_dice0.6345.pth
    
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

    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_0_dice0.3176.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_1_dice0.4057.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_2_dice0.4577.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_3_dice0.4779.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_4_dice0.4871.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_5_dice0.4795.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_6_dice0.4736.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_7_dice0.5167.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_8_dice0.5148.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_9_dice0.5184.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_10_dice0.5283.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_11_dice0.5176.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_12_dice0.5173.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_13_dice0.5244.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_14_dice0.5146.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_15_dice0.5349.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_16_dice0.5260.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_17_dice0.5264.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_18_dice0.5314.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_19_dice0.5304.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_20_dice0.5285.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_21_dice0.5311.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_22_dice0.5360.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_23_dice0.5396.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_24_dice0.5453.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_25_dice0.5383.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_26_dice0.5394.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_27_dice0.5370.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_28_dice0.5381.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_29_dice0.5407.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_30_dice0.5385.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_31_dice0.5362.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_32_dice0.5395.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_33_dice0.5416.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_34_dice0.5291.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_35_dice0.5325.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_36_dice0.5457.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_37_dice0.5309.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_38_dice0.5320.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_39_dice0.5401.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_40_dice0.5301.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_41_dice0.5347.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_42_dice0.5416.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_43_dice0.5359.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_44_dice0.5362.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_45_dice0.5337.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_46_dice0.5463.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_47_dice0.5367.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_48_dice0.5415.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_49_dice0.5347.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_50_dice0.5362.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_51_dice0.5312.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_52_dice0.5348.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_53_dice0.5405.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_54_dice0.5372.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_55_dice0.5360.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_56_dice0.5299.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_57_dice0.5376.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_58_dice0.5279.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_59_dice0.5328.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_60_dice0.5323.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_61_dice0.5380.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_62_dice0.5262.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_63_dice0.5362.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_64_dice0.5327.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_65_dice0.5362.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_66_dice0.5345.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_67_dice0.5242.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_68_dice0.5301.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_69_dice0.5327.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_70_dice0.5377.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_71_dice0.5377.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_72_dice0.5274.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_73_dice0.5214.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_74_dice0.5407.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_75_dice0.5359.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_76_dice0.5328.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_77_dice0.5306.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_78_dice0.5167.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_79_dice0.5376.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_80_dice0.5299.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_81_dice0.5338.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_82_dice0.5401.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_83_dice0.5345.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_84_dice0.5343.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_85_dice0.5335.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_86_dice0.5331.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_87_dice0.5330.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_88_dice0.5348.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_89_dice0.5429.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_90_dice0.5436.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_91_dice0.5469.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_92_dice0.5367.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_93_dice0.5377.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_94_dice0.5364.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_95_dice0.5314.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_96_dice0.5177.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_97_dice0.5216.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_98_dice0.5382.pth
    # output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-30-06-32-06/epoch_99_dice0.5352.pth

    # MSD Spleen
    # output/msd_task09+icl+grpo+long_horizon+no_augment/2026-04-25-19-48-27/best.pth
    # output/msd_task09+icl+no_agent+long_horizon+no_augment/2026-04-11-12-19-00/best.pth

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
    tasks+=("Task03")
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
