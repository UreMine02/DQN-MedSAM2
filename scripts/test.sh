CUDA_VISIBLE_DEVICES=0 python eval_3d.py \
    -pretrain output/msd_task02+grpo/2026-01-04-20-04-36/best.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -dataset msd \
    -task Task02 \
    -data_path /data/datasets/MSD \
    -num_support 1