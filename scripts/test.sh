python eval_3d.py \
    -pretrain output/hpc/msd_task03+grpo+entropy1e-1+num_support10+clip_grad0.1/2025-12-30-10-37-01/best.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -dataset msd \
    -task Task03 \
    -data_path /data/datasets/MSD \
    -num_support 5