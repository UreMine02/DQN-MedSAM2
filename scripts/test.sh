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

declare -a ckpt=(
    # output/msd_task05+grpo+icl/2026-03-08-21-48-51/best.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_0_dice0.4411.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_1_dice0.5408.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_2_dice0.5816.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_3_dice0.4997.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_4_dice0.5545.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_5_dice0.5725.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_6_dice0.5513.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_7_dice0.5927.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_8_dice0.5851.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_9_dice0.6343.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_10_dice0.6025.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_11_dice0.6150.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_12_dice0.5112.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_13_dice0.5403.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_14_dice0.5652.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_15_dice0.5631.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_16_dice0.6831.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_17_dice0.6526.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_18_dice0.5526.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_19_dice0.4828.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_20_dice0.5478.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_21_dice0.5082.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_22_dice0.6099.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_23_dice0.5023.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_24_dice0.6512.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_25_dice0.6399.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_26_dice0.5723.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_27_dice0.6615.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_28_dice0.6389.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_29_dice0.6596.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_30_dice0.6299.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_31_dice0.5713.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_32_dice0.6714.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_33_dice0.6611.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_34_dice0.5748.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_35_dice0.5387.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_36_dice0.6530.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_37_dice0.5729.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_38_dice0.6343.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_39_dice0.5939.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_40_dice0.6117.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_41_dice0.5702.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_42_dice0.5624.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_43_dice0.5811.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_44_dice0.6424.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_45_dice0.5569.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_46_dice0.6189.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_47_dice0.6244.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_48_dice0.5935.pth
    output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_49_dice0.5560.pth
)

export CUDA_VISIBLE_DEVICES=0

for pretrain in ${ckpt[@]}
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -sam_config sam2_hiera_t \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "Task05" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            # -vis
            # -no_agent
    done
done
