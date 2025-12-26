export CUDA_VISIBLE_DEVICES=0

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

# python eval_3d.py \
#     -pretrain /mnt/12T/cuong/medsam2-icl-ql/output/sarcoma+grpo+entropy1e-1+num_support10+clip_grad0.1/2025-12-24-23-45-34/epoch_2_dice0.8581.pth \
#     -rl_config rl_modules/config/grpo_po_agent.yaml \
#     -dataset sarcoma \
#     -task Sarcoma \
#     -data_path /mnt/12T/fred/medical_image \
#     -num_support 5

python eval_3d.py \
<<<<<<< HEAD
    -pretrain output/msd_task04+grpo+entropy1e-1+num_support10+clip_grad0.1/2025-12-24-14-50-57/best.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -dataset msd \
    -task Task04 \
    -data_path /data/datasets/MSD \
    -num_support 5
=======
    -pretrain /mnt/12T/cuong/medsam2-icl-ql/output/msd+grpo+entropy1e-1+num_support10+clip_grad0.1/2025-12-25-16-45-46/epoch_93_dice0.9132.pth \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -dataset msd \
    -task Task02_Heart \
    -data_path /mnt/12T/cuong/AAAI/Combined_Dataset/MSD \
    -num_support 0
>>>>>>> 56b11a7769ed7ae5a89227036fbc446cee63d881
