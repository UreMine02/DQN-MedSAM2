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
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_0_dice0.2312.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_1_dice0.4346.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_2_dice0.3363.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_3_dice0.4381.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_4_dice0.5115.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_5_dice0.5032.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_6_dice0.5029.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_7_dice0.5272.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_8_dice0.4826.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_9_dice0.5964.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_10_dice0.6132.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_11_dice0.5774.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_12_dice0.5235.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_13_dice0.5712.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_14_dice0.5991.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_15_dice0.6099.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_16_dice0.6084.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_17_dice0.6352.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_18_dice0.5601.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_19_dice0.6281.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_20_dice0.5684.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_21_dice0.6380.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_22_dice0.5723.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_23_dice0.6167.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_24_dice0.6185.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_25_dice0.6291.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_26_dice0.5856.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_27_dice0.5832.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_28_dice0.6866.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_29_dice0.6560.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_30_dice0.6346.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_31_dice0.5154.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_32_dice0.6780.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_33_dice0.7034.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_34_dice0.6179.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_35_dice0.6527.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_36_dice0.5831.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_37_dice0.6814.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_38_dice0.5375.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_39_dice0.7002.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_40_dice0.5914.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_41_dice0.6138.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_42_dice0.6027.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_43_dice0.4992.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_44_dice0.5896.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_45_dice0.5374.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_46_dice0.6976.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_47_dice0.6150.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_48_dice0.6054.pth
    output/msd_task05+grpo+icl+support5/2026-01-26-16-49-40/epoch_49_dice0.6101.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task05 \
            -data_path /data/rlsam2/datasets/nii/MSD \
            -num_support $shot
    done
done