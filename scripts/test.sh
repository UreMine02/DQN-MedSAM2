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
    output/msd_task02+grpo+icl/2026-02-28-18-09-06/best.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_0_dice0.8200.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_1_dice0.8655.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_2_dice0.8424.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_3_dice0.8623.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_4_dice0.8547.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_5_dice0.8364.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_6_dice0.8701.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_7_dice0.8784.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_8_dice0.8725.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_9_dice0.8731.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_10_dice0.8814.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_11_dice0.8784.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_12_dice0.8946.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_13_dice0.8798.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_14_dice0.8902.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_15_dice0.9001.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_16_dice0.8987.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_17_dice0.9110.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_18_dice0.9055.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_19_dice0.9019.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_20_dice0.9098.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_21_dice0.8949.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_22_dice0.9111.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_23_dice0.8952.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_24_dice0.9125.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_25_dice0.9116.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_26_dice0.8858.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_27_dice0.9033.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_28_dice0.9060.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_29_dice0.9159.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_30_dice0.9128.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_31_dice0.9111.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_32_dice0.9098.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_33_dice0.9051.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_34_dice0.9005.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_35_dice0.9125.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_36_dice0.9166.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_37_dice0.9113.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_38_dice0.9122.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_39_dice0.9064.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_40_dice0.9071.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_41_dice0.9086.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_42_dice0.9094.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_43_dice0.9048.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_44_dice0.9069.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_45_dice0.9072.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_46_dice0.9085.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_47_dice0.9075.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_48_dice0.9079.pth
    # output/msd_task02+grpo+icl/2026-02-28-18-09-06/epoch_49_dice0.9063.pth
)

export CUDA_VISIBLE_DEVICES=1

for pretrain in ${ckpt[@]}
do
    for shot in 1;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "Task02" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            # -no_agent \
            # -vis
    done
done