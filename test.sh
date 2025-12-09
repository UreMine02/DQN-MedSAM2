export CUDA_VISIBLE_DEVICES=1

# for shot in 1 5;
# do
#     for i in 213 269 282;
#     do
#     python eval_3d.py \
#         -pretrain output/msd_task02+ppo+normalized_gae0.99+entropy1e-1+num_support3/2025-12-07-20-10-15/epoch_${i}.pth \
#         -rl_config rl_modules/config/ppo_po_agent.yaml \
#         -dataset msd \
#         -task Task02 \
#         -data_path /data/datasets/MSD \
#         -wandb_enabled \
#         -num_support $shot \
#         -exp_name msd_task02_epoch${i}_shot${shot}
#     done
# done

python eval_3d.py \
    -pretrain output/msd_task02+ppo+normalized0.5_gae0.99+entropy1e-3+num_support3/2025-12-09-14-48-31/epoch_150_dice0.8531.pth \
    -rl_config rl_modules/config/ppo_po_agent.yaml \
    -dataset msd \
    -task Task02 \
    -data_path /data/datasets/MSD \
    -num_support 1