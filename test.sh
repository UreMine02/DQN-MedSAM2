CUDA_VISIBLE_DEVICES=1 python eval_3d.py \
    -pretrain /data/code/DQN-MedSAM2/output/sarcoma+mvalues+no_invalid+freeze+obj_ptr+double_dqn+qr/2025-11-20-10-18-20/epoch_15.pth \
    -rl_config /data/code/DQN-MedSAM2/sam2_train/qr_agent.yaml \
    -dataset sarcoma \
    -exp_name sarcoma \
    -data_path /data/datasets/ \
    -vis