#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=1:00:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-task07-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

declare -a ckpt=(
    # NOTE: FINAL
    # output/msd_task07+grpo+icl/2026-03-09-14-50-42/epoch_14_dice0.5667.pth
    
    # output/msd_task02+grpo+icl+test_augment/2026-03-11-15-19-11/best.pth
    # output/msd_task04+grpo+icl/best.pth

    # output/msd_task02+grpo+icl+cw_gating+semantic_filtering_with_proj_before_reshape/2026-03-17-08-01-26/best.pth
    output/msd_task09+grpo+icl+cw_gating+semantic_filtering_with_proj_before_reshape/2026-03-17-08-03-09/best.pth
)

export CUDA_VISIBLE_DEVICES=1

for idx in ${!ckpt[@]}
do
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain ${ckpt[idx]} \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "Task09" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            # -vis
            # -no_agent
    done
done
