export CUDA_VISIBLE_DEVICES=1

python eval_3d.py \
    -checkpoint_path output/msd_task10+no_icl+10fg10bg/2025-12-31-20-12-27 \
    -pretrain output/msd_task10+no_icl+10fg10bg/2025-12-31-20-12-27/best.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -dataset msd \
    -task Task10 \
    -data_path /data/datasets/MSD \
    -num_support 0 \
    -val_fg_point 10 \
    -val_bg_point 5 \
    -val_num_prompted_frame -1 \
    # -vis
