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
    output/btcv+grpo+icl+cw_soft_gating+obj_ptr_gating/2026-03-27-20-13-36/best.pth
)

export CUDA_VISIBLE_DEVICES=0

for idx in ${!ckpt[@]}
do
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain ${ckpt[idx]} \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset btcv \
            -task "" \
            -data_path /data/datasets/nii/BTCV \
            -num_support $shot \
            -gating_dimension "cw" \
            -gating_softness "soft" \
            -obj_ptr_gating \
            # -highres_gating "by_lowres" \
            # -no_agent
    done
done
