CUDA_VISIBLE_DEVICES=1 python eval_3d.py \
    -pretrain output/sarcoma+grpo+entropy1e-3+num_support3+clip_grad0.1+lazy_pen0.1+invalid_pen1/2026-01-03-16-12-05/best.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -dataset sarcoma \
    -task Task02 \
    -data_path /data/datasets/Sarcoma \
    -num_support 1