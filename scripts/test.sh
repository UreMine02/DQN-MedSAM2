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
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_0_dice0.5948.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_1_dice0.5886.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_2_dice0.5844.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_3_dice0.5941.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_4_dice0.6653.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_5_dice0.5800.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_6_dice0.6341.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_7_dice0.6219.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_8_dice0.6529.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_9_dice0.6802.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_10_dice0.6107.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_11_dice0.6367.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_12_dice0.6787.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_13_dice0.6683.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_14_dice0.6309.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_15_dice0.6814.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_16_dice0.6505.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_17_dice0.6677.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_18_dice0.6634.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_19_dice0.6106.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_20_dice0.6436.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_21_dice0.6747.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_22_dice0.6624.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_23_dice0.7146.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_24_dice0.6363.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_25_dice0.7226.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_26_dice0.6115.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_27_dice0.5932.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_28_dice0.6003.pth
    output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/epoch_29_dice0.7084.pth
)

export CUDA_VISIBLE_DEVICES=1

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
            # -no_agent
            # -vis
    done
done
