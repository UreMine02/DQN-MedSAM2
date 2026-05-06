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
    output/msd_task02+icl+grpo+lazy_pen1.0+long_horizon+flip_augment/2026-04-23-16-01-08/best.pth
    # output/msd_task02+icl+grpo+long_horizon+no_augment/2026-04-30-18-07-03/best.pth
    # output/msd_task02+icl+grpo+long_horizon+no_augment+lr2e-4/2026-05-01-07-12-31/best.pth

    # MSD Liver
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_0_dice0.4855.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_1_dice0.3491.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_2_dice0.4918.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_3_dice0.4613.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_4_dice0.5500.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_5_dice0.5085.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_6_dice0.5515.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_7_dice0.5331.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_8_dice0.5855.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_9_dice0.5984.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_10_dice0.5666.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_11_dice0.6167.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_12_dice0.6169.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_13_dice0.6446.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_14_dice0.6108.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_15_dice0.6111.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_16_dice0.6237.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_17_dice0.6276.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_18_dice0.6316.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_19_dice0.6587.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_20_dice0.6007.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_21_dice0.6187.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_22_dice0.6242.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_23_dice0.6214.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_24_dice0.6449.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_25_dice0.6509.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_26_dice0.6498.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_27_dice0.6353.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_28_dice0.6201.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_29_dice0.6357.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_30_dice0.6481.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_31_dice0.6563.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_32_dice0.6467.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_33_dice0.6453.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_34_dice0.6485.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_35_dice0.6428.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_36_dice0.6212.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_37_dice0.6528.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_38_dice0.6529.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_39_dice0.6332.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_40_dice0.6369.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_41_dice0.6668.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_42_dice0.6629.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_43_dice0.6500.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_44_dice0.6727.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_45_dice0.6163.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_46_dice0.6531.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_47_dice0.6507.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_48_dice0.6016.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_49_dice0.6424.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_50_dice0.6353.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_51_dice0.6334.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_52_dice0.6315.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_53_dice0.6335.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_54_dice0.6338.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_55_dice0.6422.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_56_dice0.6385.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_57_dice0.6262.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_58_dice0.6409.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_59_dice0.6443.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_60_dice0.6417.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_61_dice0.6530.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_62_dice0.6524.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_63_dice0.6442.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_64_dice0.6385.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_65_dice0.6471.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_66_dice0.6435.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_67_dice0.6428.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_68_dice0.6358.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_69_dice0.6202.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_70_dice0.6663.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_71_dice0.6441.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_72_dice0.6131.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_73_dice0.6215.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_74_dice0.6391.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_75_dice0.6330.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_76_dice0.6434.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_77_dice0.6501.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_78_dice0.6130.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_79_dice0.6255.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_80_dice0.6234.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_81_dice0.6409.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_82_dice0.6183.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_83_dice0.6504.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_84_dice0.5861.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_85_dice0.6449.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_86_dice0.6482.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_87_dice0.6467.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_88_dice0.6204.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_89_dice0.6461.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_90_dice0.6406.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_91_dice0.6251.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_92_dice0.6237.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_93_dice0.6294.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_94_dice0.6408.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_95_dice0.6449.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_96_dice0.6484.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_97_dice0.6185.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_98_dice0.6444.pth
    # output/msd_task03+icl+grpo+long_horizon+no_augment+lr2e-4+ep100+cosine_scheduler_/2026-05-04-11-53-03/epoch_99_dice0.6271.pth

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
    # output/msd_task08+icl+grpo+long_horizon+no_augment+lr2e-4+ep200/2026-05-02-20-19-57/epoch_37_dice0.5505.pth

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
    tasks+=("Task02")
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
            -ablation \
            $( [ ${no_agents[idx]} = "no_agent" ] && echo "-no_agent" )
            # -vis \
    done
done
