CUDA_VISIBLE_DEVICES=0 python eval_3d.py \
    -pretrain output/sarcoma+a2c+gae1.0/2025-11-29-17-49-20/epoch_25.pth \
    -rl_config rl_modules/config/a2c_po_agent.yaml \
    -dataset sarcoma \
    -exp_name sarcoma \
    -data_path /data/datasets