# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import time
import pytz
import numpy as np
from datetime import datetime
from functools import partial
from tqdm import tqdm

import cfg
from conf import settings
from func_3d import function
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb # NOTE: WANDB

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12348'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank=0, world_size=0):
    args = cfg.parse_args()

    if args.distributed:
        setup(rank, world_size)
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)
        
    # NOTE: WANDB
    if args.wandb_enabled:
        wandb.init(
            project="dqn-medsam2",
            name=args.exp_name,              # Experiment name from args
            config=args
        )

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    net.to(dtype=torch.bfloat16)
    agent = getattr(net, "agent", None)
    if agent is not None:
        agent.to_dtype(torch.bfloat16)
        
    if args.wandb_enabled:
        wandb.watch(net)
        
    if args.pretrain:
        print(args.pretrain)
        weights = torch.load(args.pretrain, map_location=GPUdevice)
        net.load_state_dict(weights["model"], strict=False)
        if "agent" in weights.keys() and not args.no_agent:
            agent.load_state_dict(weights["agent"])

    for name, param in net.named_parameters():
        if "image_encoder" in name:
            param.requires_grad_(False)
        elif "sam_prompt_encoder" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    agent_n_params = 0
    if agent is not None:
        agent_n_params = agent.num_parameters()

    n_parameters_tot = sum(p.numel() for p in net.parameters())
    print(f'Number of sam2 params: {n_parameters_tot:,}')
    print(f'Number of agent params: {agent_n_params:,}')

    head, fix = [], []
    for k, v in net.named_parameters():
        (head if v.requires_grad else fix).append(v)

    print(f'Trainable parameters: {sum(p.numel() for p in head) + agent_n_params:,}')
    print(f'Parameters fixed: {sum(p.numel() for p in fix):,}')

    if args.distributed:
        net = DDP(net, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        if not args.no_agent:
            net.module.agent.to_distributed(rank=rank)
            print("Wrapped agent for distributed training")

    param_list = [{'params': head, 'initial_lr': args.lr}]
    optimizer = torch_optim.AdamW(param_list, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.ep, eta_min=args.lr/10)
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    nice_train_loader, nice_test_loader = get_dataloader(args, rank=rank, world_size=world_size)
    
    
    for epoch in range(100):
        net.train()
        net.image_encoder.eval()
        net.sam_prompt_encoder.eval()
        with tqdm(total=len(nice_train_loader), unit='img', position=0, miniters=10) as pbar:
            for batch_idx, packs in enumerate(nice_train_loader): #metric_logger.log_every(train_loader, print_freq, header=header):
                whole_imgs_tensor = packs["image"].squeeze(0).to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
                whole_masks_tensor = packs["label"].squeeze(0).to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
                whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
                whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
                task = packs["task"][0]
                
                imgs_tensor = F.interpolate(whole_imgs_tensor, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
                masks_tensor = F.interpolate(whole_masks_tensor.unsqueeze(1), size=(args.image_size, args.image_size), mode="nearest").squeeze(1)
                
                support_imgs_tensor = F.interpolate(whole_support_imgs_tensor, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
                support_masks_tensor = F.interpolate(whole_support_masks_tensor.unsqueeze(1), size=(args.image_size, args.image_size), mode="nearest").squeeze(1)
                
                train_state = net.train_init_state(
                    args=args,
                    imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor
                )
                
                # print(imgs_tensor.shape, masks_tensor.shape, support_imgs_tensor.shape, support_masks_tensor.shape)
                
                for frame_idx in range(support_masks_tensor.shape[0]):
                    mask = support_masks_tensor[frame_idx]
                    _, _, _ = net.train_add_new_mask(
                        inference_state=train_state,
                        frame_idx=frame_idx,
                        obj_id=packs["obj_id"][0],
                        mask=mask.to(device=GPUdevice),
                    )
                    
                train_state["support_set_stage"] = False
                for frame_idx in range(imgs_tensor.shape[0]):
                    (
                        _,
                        _,
                        current_vision_feats,
                        current_vision_pos_embeds,
                        feat_sizes,
                    ) = net._get_image_feature(train_state, frame_idx, 1)
                    
                pbar.update()   
                
        net.eval()


def main():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = cfg.parse_args()

    if args.distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train()

if __name__ == '__main__':
    main()
    
