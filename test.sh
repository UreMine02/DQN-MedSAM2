CUDA_VISIBLE_DEVICES=1 python eval_3d.py \
    -pretrain output/sarcoma+mvalues+no_invalid+freeze+obj_ptr+double_dqn+vanilla/2025-11-19-10-19-13/epoch_20.pth \
    -rl_config rl_modules/config/vanilla_q_agent.yaml \
    -dataset sarcoma \
    -exp_name sarcoma \
    -data_path /data/datasets