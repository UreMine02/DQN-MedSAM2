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
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_0_dice0.2962.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_1_dice0.3935.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_2_dice0.4165.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_3_dice0.3946.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_4_dice0.4200.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_5_dice0.4473.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_6_dice0.4316.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_7_dice0.4080.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_8_dice0.4253.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_9_dice0.4249.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_10_dice0.4294.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_11_dice0.4530.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_12_dice0.4469.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_13_dice0.4285.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_14_dice0.3773.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_15_dice0.4208.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_16_dice0.4097.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_17_dice0.4236.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_18_dice0.4221.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_19_dice0.4613.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_20_dice0.4273.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_21_dice0.4368.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_22_dice0.4582.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_23_dice0.4659.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_24_dice0.4170.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_25_dice0.4460.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_26_dice0.4403.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_27_dice0.4556.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_28_dice0.4611.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_29_dice0.4559.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_30_dice0.4501.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_31_dice0.4493.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_32_dice0.3926.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_33_dice0.3833.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_34_dice0.4484.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_35_dice0.4335.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_36_dice0.4288.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_37_dice0.0000.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_38_dice0.0166.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_39_dice0.0806.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_40_dice0.0423.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_41_dice0.0979.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_42_dice0.0996.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_43_dice0.0612.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_44_dice0.0869.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_45_dice0.1281.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_46_dice0.1306.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_47_dice0.1399.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_48_dice0.1019.pth
    output/dpc/msd_task01+grpo+icl/2026-01-11-16-34-10/epoch_49_dice0.0678.pth
)

export CUDA_VISIBLE_DEVICES=0

for pretrain in ${ckpt[@]};
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task01 \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot
            # -vis
    done
done