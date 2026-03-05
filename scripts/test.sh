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
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_0_dice0.0632.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_1_dice0.2198.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_2_dice0.2662.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_3_dice0.3073.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_4_dice0.2959.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_5_dice0.3732.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_6_dice0.3937.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_7_dice0.3351.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_8_dice0.4013.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_9_dice0.4665.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_10_dice0.3911.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_11_dice0.4495.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_12_dice0.4774.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_13_dice0.4211.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_14_dice0.4613.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_15_dice0.4932.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_16_dice0.4966.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_17_dice0.4825.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_18_dice0.4803.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_19_dice0.4976.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_20_dice0.5129.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_21_dice0.4663.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_22_dice0.4762.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_23_dice0.5354.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_24_dice0.5176.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_25_dice0.4941.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_26_dice0.5001.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_27_dice0.4749.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_28_dice0.5502.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_29_dice0.5286.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_30_dice0.5211.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_31_dice0.5089.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_32_dice0.5175.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_33_dice0.4948.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_34_dice0.5155.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_35_dice0.5083.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_36_dice0.4862.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_37_dice0.5115.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_38_dice0.4659.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_39_dice0.5014.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_40_dice0.5109.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_41_dice0.5184.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_42_dice0.5443.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_43_dice0.5280.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_44_dice0.5029.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_45_dice0.5087.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_46_dice0.5206.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_47_dice0.5382.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_48_dice0.5197.pth
    output/msd_task10+grpo+icl/2026-03-06-00-16-26/epoch_49_dice0.5102.pth
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
            -task "Task10" \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            # -no_agent
            # -vis
    done
done
