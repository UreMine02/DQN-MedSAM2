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

<<<<<<< HEAD
ckpt=(
    # output/msd_task02+grpo+icl/2026-02-22-10-36-54/best.pth
    # output/msd_task02+no_agent+icl+freeze/2026-02-19-20-33-41/best.pth

    # output/sarcoma+grpo+icl/2026-03-03-22-24-08/best.pth
    # output/sarcoma+no_agent+icl/2026-03-03-20-51-26/best.pth

    # output/msd_task09+grpo+icl/2026-03-02-21-58-22/best.pth
    # output/msd_task09+no_agent+icl+augmentation/2026-03-02-17-35-55/best.pth

    # output/msd_task06+grpo+icl/2026-03-03-22-25-28/best.pth
    # output/msd_task06+no_agent+icl/2026-03-03-20-51-01/best.pth

    ## NOTE
    # output/output/msd_task05+grpo+icl/2026-02-28-17-32-43/best.pth # new + not
    # output/output/msd_task05+no_agent+icl/2026-03-04-21-27-17/best.pth # new + rot

    # output/output/msd_task07+grpo+icl/2026-03-02-22-01-45/best.pth # new + rot
    # output/output/msd_task07+no_agent+icl/2026-03-03-16-50-01/best.pth # new + rot

    # output/output/msd_task08+no_agent+icl/2026-03-03-17-16-27/best.pth # new + rot
    # output/output/msd_task08+grpo+icl/2026-02-28-19-16-27/best.pth

    # output/output/msd_task04+no_agent+icl/2026-03-04-21-27-18/best.pth # new + rot
    # output/output/msd_task04+grpo+icl/2026-02-28-21-24-19/best.pth
    # output/msd_task04+grpo+icl/2026-02-22-23-57-22/best.pth

    # output/msd_task10+no_agent+icl/2026-03-04-21-14-49/best.pth
    # output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_24_dice0.5176.pth

    
    # output/output/msd_task03+no_agent+icl/2026-03-04-21-27-05/best.pth
    # output/output/msd_task03+grpo+icl/2026-02-28-19-16-27/best.pth
    # output/output/msd_task03+grpo+icl/2026-02-28-19-16-27/epoch_0_dice0.5179.pth
    output/output/msd_task03+grpo+icl/2026-02-28-19-16-27/epoch_29_dice0.6861.pth
=======
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
>>>>>>> msd01
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
<<<<<<< HEAD
            -task "Task03" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            # -ablation \
            # -no_agent
            # -vis \
=======
            -task "Task02" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            -memory_bank_size 6 \
            -gating_dimension "cw" \
            -gating_softness "soft" \
            -obj_ptr_gating \
            # -no_agent \
            # -highres_gating "by_lowres"
>>>>>>> msd01
    done
done
