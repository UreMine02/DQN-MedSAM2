CUDA_VISIBLE_DEVICES=0 python eval_3d.py \
    -pretrain output/sarcoma+a2c+gae0/2025-11-28-17-25-42/epoch_0.pth \
    -rl_config rl_modules/config/a2c_po_agent.yaml \
    -dataset sarcoma \
    -exp_name sarcoma \
    -data_path /data/datasets/Sarcoma