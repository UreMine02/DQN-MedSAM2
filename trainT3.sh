CUDA_VISIBLE_DEVICES=0 python train_3d.py \
    -exp_name msdTask4+grpo+entropy1e-1+num_support10+clip_grad0.1 \
    -sam_ckpt /mnt/12T/cuong/medsam2-icl-ql/output/msdTask4+bbox+grpo+entropy1e-1+num_support0+clip_grad0.1/2026-02-22-04-45-03/epoch_9_dice0.7489.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -checkpoint_path ./output/msdTask4+bbox+grpo+entropy1e-1+num_support0+clip_grad0.1 \
    -dataset msd \
    -task Task04_Hippocampus \
    -data_path /mnt/12T/AAAI/Combined_Dataset/MSD \
    -lr 1e-4 \
    -val_freq 1 \
    -ep 300 \
    -q_updates_per_step 3 \
    -lazy_penalty 0 \
    -invalid_penalty 0.0 \
    -num_support 0 \
    
