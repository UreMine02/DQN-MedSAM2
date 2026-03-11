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

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    # output/msd_task10+grpo+icl/2026-03-10-19-22-49/best.pth

    # NOTE: TESTING
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_0_dice0.4325.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_1_dice0.4434.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_2_dice0.4689.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_3_dice0.4382.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_4_dice0.4366.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_5_dice0.5229.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_6_dice0.5179.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_7_dice0.4946.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_8_dice0.5584.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_9_dice0.5530.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_10_dice0.5217.pth
    # output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_11_dice0.5483.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_12_dice0.5922.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_13_dice0.5855.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_14_dice0.6107.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_15_dice0.5551.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_16_dice0.5646.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_17_dice0.6004.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_18_dice0.5642.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_19_dice0.5706.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_20_dice0.5917.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_21_dice0.5718.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_22_dice0.5785.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_23_dice0.5993.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_24_dice0.6147.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_25_dice0.6298.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_26_dice0.5294.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_27_dice0.5691.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_28_dice0.5691.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_29_dice0.6155.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_30_dice0.5818.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_31_dice0.6277.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_32_dice0.5663.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_33_dice0.6165.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_34_dice0.6402.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_35_dice0.6111.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_36_dice0.6130.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_37_dice0.6199.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_38_dice0.6129.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_39_dice0.6005.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_40_dice0.5722.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_41_dice0.6254.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_42_dice0.5864.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_43_dice0.6331.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_44_dice0.6127.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_45_dice0.5865.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_46_dice0.6293.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_47_dice0.6224.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_48_dice0.6209.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_49_dice0.6272.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_50_dice0.6258.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_51_dice0.6238.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_52_dice0.6252.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_53_dice0.5786.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_54_dice0.5913.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_55_dice0.6358.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_56_dice0.6082.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_57_dice0.5593.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_58_dice0.6181.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_59_dice0.6174.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_60_dice0.5812.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_61_dice0.6166.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_62_dice0.6117.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_63_dice0.6059.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_64_dice0.6204.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_65_dice0.6086.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_66_dice0.6157.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_67_dice0.5676.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_68_dice0.6139.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_69_dice0.5621.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_70_dice0.5803.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_71_dice0.5956.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_72_dice0.6115.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_73_dice0.6149.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_74_dice0.6235.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_75_dice0.6254.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_76_dice0.6070.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_77_dice0.5707.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_78_dice0.6158.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_79_dice0.6254.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_80_dice0.6147.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_81_dice0.6049.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_82_dice0.6271.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_83_dice0.5867.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_84_dice0.6192.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_85_dice0.5791.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_86_dice0.6207.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_87_dice0.6118.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_88_dice0.6192.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_89_dice0.6298.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_90_dice0.6163.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_91_dice0.6330.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_92_dice0.6171.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_93_dice0.6160.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_94_dice0.6101.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_95_dice0.6295.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_96_dice0.5851.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_97_dice0.6340.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_98_dice0.6329.pth
    output/msd_task03+grpo+icl/2026-03-10-19-48-14/epoch_99_dice0.5973.pth
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
            -task "Task03" \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot
            # -no_agent
    done
done
