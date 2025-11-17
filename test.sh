CUDA_VISIBLE_DEVICES=0 python eval_3d.py \
    -pretrain /data/code/DQN-MedSAM2/output/msd-task02-mvalues/mvalues-1gpu/2025-11-15-11-30-29/epoch_29.pth \
    -dataset msd \
    -exp_name msd-task02 \
    -data_path /data/datasets/Combined_Dataset/MSD \
    -task Task02