#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=128GB # ram
#SBATCH --time=30:00 # time
#SBATCH -J btcv_eval # job name
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/btcv_eval-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

ckpt=(
    # output/msd_task02+grpo+icl/2026-02-22-10-36-54/best.pth
    # output/msd_task02+no_agent+icl+freeze/2026-02-19-20-33-41/best.pth
    # output/msd_task02+grpo+icl+agent_only/2026-02-22-15-36-52/best.pth

    # output/msd_task02+grpo+icl/2026-02-26-12-37-46/best.pth
    # output/msd_task02+grpo+icl/2026-02-26-19-17-03/best.pth
    # output/msd_task02+grpo+icl/2026-02-26-17-39-36/best.pth

    
    # output/sarcoma+grpo+icl/2026-02-22-18-05-08/best.pth
    # output/sarcoma+grpo+icl/2026-02-28-18-49-55/best.pth

    # output/sarcoma+grpo+icl/2026-02-28-18-49-55/epoch_7_dice0.8250.pth
    # output/msd_task09+no_agent+icl+augmentation/2026-03-02-17-35-55/best.pth
    # output/msd_task10+no_agent+icl/2026-02-27-18-53-19/best.pth

    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_0_dice0.3657.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_1_dice0.3565.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_2_dice0.4204.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_3_dice0.5204.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_4_dice0.5546.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_5_dice0.5889.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_6_dice0.5732.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_7_dice0.6130.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_8_dice0.5812.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_9_dice0.5465.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_10_dice0.5986.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_11_dice0.5655.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_12_dice0.5899.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_13_dice0.5955.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_14_dice0.5991.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_15_dice0.5864.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_16_dice0.6094.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_17_dice0.6010.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_18_dice0.5851.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_19_dice0.6131.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_20_dice0.6457.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_21_dice0.6269.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_22_dice0.5432.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_23_dice0.6317.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_24_dice0.5472.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_25_dice0.5758.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_26_dice0.6476.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_27_dice0.6463.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_28_dice0.6038.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_29_dice0.5957.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_30_dice0.5607.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_31_dice0.5965.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_32_dice0.5838.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_33_dice0.5931.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_34_dice0.6244.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_35_dice0.6249.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_36_dice0.6339.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_37_dice0.6101.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_38_dice0.6601.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_39_dice0.6561.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_40_dice0.6430.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_41_dice0.6488.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_42_dice0.6048.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_43_dice0.5430.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_44_dice0.6170.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_45_dice0.6013.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_46_dice0.5817.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_47_dice0.6364.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_48_dice0.6160.pth
    output/msd_task10+no_agent+icl/2026-02-27-18-53-19/epoch_49_dice0.6119.pth
)

export CUDA_VISIBLE_DEVICES=0

for pretrain in ${ckpt[@]};
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "Task10" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            -no_agent
            # -vis \
            # -ablation \
    done
done