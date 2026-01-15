#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=128GB # ram
#SBATCH --time=30:00 # time
#SBATCH -J msd03_eval # job name
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd03_eval-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

# export CUDA_VISIBLE_DEVICES=1

declare -a ckpt=(        
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_0_dice0.2960.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_1_dice0.2799.pth 
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_2_dice0.3636.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_3_dice0.3693.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_4_dice0.3945.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_5_dice0.3924.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_6_dice0.4083.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_7_dice0.4498.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_8_dice0.4973.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_9_dice0.4239.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_10_dice0.4870.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_11_dice0.5429.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_12_dice0.4884.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_13_dice0.5142.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_14_dice0.4594.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_15_dice0.4685.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_16_dice0.5317.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_17_dice0.5218.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_18_dice0.5167.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_19_dice0.5048.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_20_dice0.4918.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_21_dice0.5368.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_22_dice0.5288.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_23_dice0.5160.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_24_dice0.5421.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_25_dice0.6032.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_26_dice0.4908.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_27_dice0.5262.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_28_dice0.5582.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_29_dice0.5336.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_30_dice0.5417.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_31_dice0.5362.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_32_dice0.5635.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_33_dice0.5988.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_34_dice0.5947.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_35_dice0.6111.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_36_dice0.5514.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_37_dice0.5745.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_38_dice0.5674.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_39_dice0.5994.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_40_dice0.5836.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_41_dice0.5721.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_42_dice0.6234.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_43_dice0.5879.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_44_dice0.5848.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_45_dice0.5614.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_46_dice0.6045.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_47_dice0.5537.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_48_dice0.6029.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_49_dice0.5919.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_50_dice0.6284.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_51_dice0.5708.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_52_dice0.5888.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_53_dice0.6203.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_54_dice0.6176.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_55_dice0.5897.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_56_dice0.5614.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_57_dice0.6013.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_58_dice0.5431.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_59_dice0.5659.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_60_dice0.6225.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_61_dice0.5621.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_62_dice0.6172.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_63_dice0.5811.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_64_dice0.6269.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_65_dice0.6006.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_66_dice0.6085.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_67_dice0.5779.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_68_dice0.6312.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_69_dice0.6508.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_70_dice0.6434.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_71_dice0.6117.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_72_dice0.6103.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_73_dice0.5728.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_74_dice0.6457.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_75_dice0.6312.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_76_dice0.6243.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_77_dice0.5903.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_78_dice0.5727.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_79_dice0.5921.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_80_dice0.6398.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_81_dice0.6285.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_82_dice0.6164.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_83_dice0.6630.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_84_dice0.5936.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_85_dice0.6436.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_86_dice0.5896.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_87_dice0.5895.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_88_dice0.6102.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_89_dice0.6194.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_90_dice0.6337.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_91_dice0.6397.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_92_dice0.5958.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_93_dice0.6699.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_94_dice0.6275.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_95_dice0.6385.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_96_dice0.6399.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_97_dice0.6186.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_98_dice0.5719.pth
    /data/rlsam2/DQN-MedSAM2/output/btcv+grpo+icl/2026-01-15-14-22-54/epoch_99_dice0.5839.pth
)

for pretrain in ${ckpt[@]};
do
    for shot in 1;
    do
        python eval_3d.py \
            -pretrain $pretrain \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset btcv \
            -task Task01 \
            -data_path /data/rlsam2/datasets/nii/BTCV \
            -num_support $shot
    done
done