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
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    # output/msd_task10+grpo+icl/2026-03-10-19-22-49/best.pth

    # NOTE: TESTING
    # output/btcv+grpo+icl+cw_soft_gating+obj_ptr_gating/2026-03-27-20-13-36/best.pth
    # output/msd_task02+grpo+icl+discri_gating+obj_ptr_gating+highres_gating_by_lowres+rigor_augment/2026-03-29-11-38-40/best.pth
    # output/msd_task02+grpo+icl+cw_gating+semantic_filtering+force_add+highres_gating/2026-03-25-20-05-33/best.pth
    # output/msd_task02+grpo+icl+no_agent+rigor_augment/2026-03-30-09-38-30/best.pth

    # output/msd_task02+no_agent+icl+no_augment/2026-04-03-19-32-08/best.pth
    # output/msd_task02+grpo+icl+no_augment/2026-04-04-10-53-56/best.pth
    output/msd_task02+grpo+icl+cw_soft_gating+obj_ptr_gating+no_augment/2026-04-04-20-00-33/best.pth
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
            -task "Task02" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            -memory_bank_size 6 \
            -gating_dimension "cw" \
            -gating_softness "soft" \
            -obj_ptr_gating \
            # -no_agent \
            # -highres_gating "by_lowres"
    done
done
