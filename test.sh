export CUDA_VISIBLE_DEVICES=1

python eval_3d.py \
    -pretrain output/test/2025-12-10-17-12-22/epoch_0_dice0.6363.pth \
    -rl_config rl_modules/config/ppo_po_agent.yaml \
    -dataset msd \
    -task Task02 \
    -data_path /data/datasets/MSD \
    -num_support 3