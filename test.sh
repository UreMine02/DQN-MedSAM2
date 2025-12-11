export CUDA_VISIBLE_DEVICES=1


CHECKPOINT=output/msd_task02+ppo+normalized0.5_gae0.99+entropy1e-3+num_support3+clip_grad0.1/2025-12-11-15-43-13/epoch_281_dice0.8571.pth
python eval_3d.py \
    -pretrain $CHECKPOINT \
    -rl_config rl_modules/config/ppo_po_agent.yaml \
    -dataset msd \
    -task Task02 \
    -data_path /data/datasets/MSD \
    -num_support 1