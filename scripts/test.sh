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
    #NOTE: FINAL
    # output/msd_task02+grpo+icl+correct_reward/2026-03-09-11-11-36/best.pth
    # output/sarcoma+grpo+icl/2026-03-03-22-24-08/best.pth
    # output/msd_task05+grpo+icl/2026-03-08-21-48-51/epoch_24_dice0.6512.pth
    # output/msd_task06+grpo+icl/2026-03-03-22-25-28/best.pth

    output/msd_task09+grpo+icl/2026-03-10-10-29-32/best.pth
)

export CUDA_VISIBLE_DEVICES=0

for pretrain in ${ckpt[@]}
do
    for shot in 1 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -sam_config sam2_hiera_t \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "Task09" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            # -vis
            # -no_agent
    done
done
