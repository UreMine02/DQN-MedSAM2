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
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_0_dice0.7556.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_1_dice0.7692.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_2_dice0.8381.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_3_dice0.8550.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_4_dice0.8684.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_5_dice0.8610.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_6_dice0.8246.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_7_dice0.8631.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_8_dice0.8859.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_9_dice0.8465.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_10_dice0.8480.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_11_dice0.8647.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_12_dice0.8885.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_13_dice0.8831.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_14_dice0.8920.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_15_dice0.8892.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_16_dice0.8970.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_17_dice0.9019.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_18_dice0.9005.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_19_dice0.9033.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_20_dice0.8965.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_21_dice0.8981.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_22_dice0.8611.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_23_dice0.8966.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_24_dice0.8959.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_25_dice0.9075.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_26_dice0.9003.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_27_dice0.8501.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_28_dice0.8832.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_29_dice0.9057.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_30_dice0.8961.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_31_dice0.8894.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_32_dice0.9117.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_33_dice0.9117.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_34_dice0.9059.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_35_dice0.9113.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_36_dice0.9113.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_37_dice0.9012.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_38_dice0.9076.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_39_dice0.9150.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_40_dice0.9118.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_41_dice0.9133.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_42_dice0.9042.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_43_dice0.9019.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_44_dice0.9149.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_45_dice0.9109.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_46_dice0.9144.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_47_dice0.9063.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_48_dice0.9104.pth
    output/msd_task02+grpo+icl/2026-02-21-09-06-46/epoch_49_dice0.9066.pth
)

export CUDA_VISIBLE_DEVICES=1

for pretrain in ${ckpt[@]}
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task Task02 \
            -data_path /data/datasets/nii/MSD \
            -num_support $shot \
            # -no_agent \
            # -vis
    done
done