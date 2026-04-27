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

declare -a ckpt=(
    # MSD Heart
    # output/msd_task02+icl+ppo+skip_penalty+long_horizon+no_augment/2026-04-08-08-07-00/best.pth
    # output/msd_task02+icl+ppo+no_agent+long_horizon+no_augment/2026-04-10-14-46-14/best.pth
    # output/msd_task02+no_agent+icl+no_augment/2026-04-03-19-32-08/best.pth
    # output/msd_task02+icl+ppo+long_horizon+no_augment/2026-04-17-16-43-56/best.pth
    # output/msd_task02+icl+ppo+long_horizon+no_augment/2026-04-16-10-46-24/best.pth
    # output/msd_task02+icl+grpo+lazy_pen1.0+long_horizon+flip_augment/2026-04-23-16-01-08/best.pth

    # Sarcoma
    # output/sarcoma+icl+ppo+long_horizon+no_augment/2026-04-10-19-46-46/best.pth
    # output/sarcoma+icl+no_agent+long_horizon+no_augment/2026-04-10-19-47-31/best.pth

    # MSD Colon
    # output/msd_task10+icl+grpo+long_horizon+no_augment/2026-04-25-13-09-26/best.pth
    # output/msd_task10+icl+no_agent+long_horizon+no_augment/2026-04-12-15-52-38/best.pth
    # output/msd_task10+icl+ppo+long_horizon+no_augment/2026-04-17-18-39-30/best.pth

    # MSD Spleen
    # output/msd_task09+icl+grpo+long_horizon+no_augment/2026-04-25-19-48-27/best.pth
    # output/msd_task09+icl+ppo+long_horizon+no_augment/2026-04-11-12-20-13/best.pth
    # output/msd_task09+icl+no_agent+long_horizon+no_augment/2026-04-11-12-19-00/best.pth
    
    # MSD Prostate
    # output/msd_task05+grpo+long_horizon+no_augment/2026-04-26-11-35-38/best.pth
    # output/msd_task05+no_agent+long_horizon+no_augment/2026-04-13-18-04-27/best.pth
    # output/msd_task05+ppo+long_horizon+no_augment/2026-04-13-14-13-19/best.pth

    # MSD Hepatic
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_0_dice0.2622.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_1_dice0.3736.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_2_dice0.4118.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_3_dice0.4548.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_4_dice0.4716.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_5_dice0.4725.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_6_dice0.5013.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_7_dice0.4890.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_8_dice0.5008.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_9_dice0.4919.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_10_dice0.5035.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_11_dice0.5080.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_12_dice0.5096.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_13_dice0.5172.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_14_dice0.5240.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_15_dice0.5169.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_16_dice0.5299.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_17_dice0.5250.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_18_dice0.5356.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_19_dice0.5279.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_20_dice0.5369.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_21_dice0.5305.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_22_dice0.5377.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_23_dice0.5293.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_24_dice0.5247.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_25_dice0.5270.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_26_dice0.5219.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_27_dice0.5343.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_28_dice0.5169.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_29_dice0.5346.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_30_dice0.5405.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_31_dice0.5357.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_32_dice0.5272.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_33_dice0.5367.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_34_dice0.5361.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_35_dice0.5345.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_36_dice0.5289.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_37_dice0.5404.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_38_dice0.5382.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_39_dice0.5320.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_40_dice0.5302.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_41_dice0.5268.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_42_dice0.5320.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_43_dice0.5314.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_44_dice0.5306.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_45_dice0.5291.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_46_dice0.5319.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_47_dice0.5291.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_48_dice0.5348.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_49_dice0.5304.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_50_dice0.5364.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_51_dice0.5339.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_52_dice0.5413.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_53_dice0.5363.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_54_dice0.5367.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_55_dice0.5278.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_56_dice0.5242.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_57_dice0.5130.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_58_dice0.5126.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_59_dice0.5313.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_60_dice0.5290.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_61_dice0.5309.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_62_dice0.5361.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_63_dice0.5281.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_64_dice0.5277.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_65_dice0.5195.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_66_dice0.5316.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_67_dice0.5190.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_68_dice0.5294.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_69_dice0.5274.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_70_dice0.5213.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_71_dice0.5327.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_72_dice0.5375.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_73_dice0.5224.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_74_dice0.5335.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_75_dice0.5354.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_76_dice0.5388.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_77_dice0.5180.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_78_dice0.5189.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_79_dice0.5357.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_80_dice0.5227.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_81_dice0.5196.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_82_dice0.5331.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_83_dice0.5293.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_84_dice0.5319.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_85_dice0.5294.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_86_dice0.5320.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_87_dice0.5331.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_88_dice0.5218.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_89_dice0.5321.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_90_dice0.5101.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_91_dice0.5352.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_92_dice0.5336.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_93_dice0.5276.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_94_dice0.5175.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_95_dice0.5319.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_96_dice0.5335.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_97_dice0.5300.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_98_dice0.5364.pth
    output/msd_task08+grpo+icl+long_horizon+no_augment/2026-04-26-17-25-02/epoch_99_dice0.5217.pth
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
            -task "Task08" \
            -data_path /data/datasets/nii/ \
            -num_support $shot \
            -memory_bank_size 6 \
            # -ablation \
            # -no_agent
            # -vis \
    done
done
