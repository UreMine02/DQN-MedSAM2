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

conda activate rlsam2
cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
conda init
conda activate rlsam2

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    # output/msd_task10+grpo+icl/2026-03-10-19-22-49/best.pth

    # NOTE: TESTING
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_0_dice0.3825.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_1_dice0.3771.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_2_dice0.4349.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_3_dice0.3938.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_4_dice0.3715.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_5_dice0.3577.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_6_dice0.4521.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_7_dice0.4599.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_8_dice0.4692.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_9_dice0.4266.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_10_dice0.5022.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_11_dice0.5117.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_12_dice0.5008.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_13_dice0.4780.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_14_dice0.5268.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_15_dice0.5489.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_16_dice0.5487.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_17_dice0.5929.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_18_dice0.5324.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_19_dice0.5341.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_20_dice0.5585.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_21_dice0.5501.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_22_dice0.5322.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_23_dice0.5081.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_24_dice0.5470.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_25_dice0.5629.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_26_dice0.5484.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_27_dice0.5316.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_28_dice0.5810.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_29_dice0.6002.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_30_dice0.5839.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_31_dice0.5970.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_32_dice0.5874.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_33_dice0.5750.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_34_dice0.5866.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_35_dice0.6037.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_36_dice0.5824.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_37_dice0.5766.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_38_dice0.5529.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_39_dice0.5889.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_40_dice0.5834.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_41_dice0.5851.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_42_dice0.5977.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_43_dice0.5672.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_44_dice0.5933.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_45_dice0.5649.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_46_dice0.5956.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_47_dice0.5667.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_48_dice0.5591.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_49_dice0.5561.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_50_dice0.5816.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_51_dice0.5334.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_52_dice0.5751.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_53_dice0.5770.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_54_dice0.5557.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_55_dice0.5244.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_56_dice0.5651.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_57_dice0.5442.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_58_dice0.5547.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_59_dice0.5778.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_60_dice0.5622.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_61_dice0.5632.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_62_dice0.5681.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_63_dice0.5714.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_64_dice0.5603.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_65_dice0.5534.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_66_dice0.5763.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_67_dice0.5777.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_68_dice0.5892.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_69_dice0.5704.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_70_dice0.5752.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_71_dice0.5828.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_72_dice0.5664.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_73_dice0.5641.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_74_dice0.5802.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_75_dice0.5813.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_76_dice0.5582.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_77_dice0.5709.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_78_dice0.5733.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_79_dice0.5947.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_80_dice0.5639.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_81_dice0.5533.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_82_dice0.5454.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_83_dice0.5614.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_84_dice0.5511.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_85_dice0.5589.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_86_dice0.6000.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_87_dice0.5693.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_88_dice0.5875.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_89_dice0.5786.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_90_dice0.5943.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_91_dice0.5627.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_92_dice0.5480.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_93_dice0.5739.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_94_dice0.5810.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_95_dice0.5434.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_96_dice0.5854.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_97_dice0.5776.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_98_dice0.5711.pth
    output/msd_task03+grpo+icl/2026-03-11-17-06-11/epoch_99_dice0.5404.pth
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
            -task "Task03" \
            -data_path /hpcfs/users/a1232079/duyanh/MedSAM2/datasets/nii/MSD \
            -num_support $shot
            # -no_agent
    done
done
