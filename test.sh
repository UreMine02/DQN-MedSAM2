export CUDA_VISIBLE_DEVICES=1

python eval_3d.py \
    -pretrain output/sarcoma+grpo+no_icl/2025-12-26-19-24-41/best.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -dataset sarcoma \
    -task Task02_Heart \
    -data_path /data/datasets/Sarcoma \
    -num_support 0 \
    -eval_fg_point 5 \
    -eval_bg_point 5