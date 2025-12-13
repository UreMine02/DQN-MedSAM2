# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import time

import torch
import torch.optim as optim

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
import wandb
from datetime import datetime
import pytz
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np

import warnings
warnings.filterwarnings("ignore")

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

    if args.wandb_enabled:
        wandb.init(
            project="dqn-medsam2",
            name=args.exp_name              # Experiment name from args
        )

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    net.to(dtype=torch.bfloat16)
    
    if args.pretrain:
        print(args.pretrain)
        weights = torch.load(args.pretrain, map_location=GPUdevice)
        net.load_state_dict(weights["model"], strict=False)
        if "agent" in weights.keys():
            net.agent.load_state_dict(weights["agent"])
    
    if args.distributed:
        net = DDP(net, device_ids=[rank])
        net.module.agent.q_net = DDP(net.module.agent.q_net, device_ids=[rank])
    
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, fused=True)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    #create checkpoint folder to save model
    root_path = args.checkpoint_path
    current_time = datetime.now(pytz.timezone("Australia/Adelaide")).strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_path = os.path.join(root_path, current_time)
    if not os.path.exists(checkpoint_path) and args.save_ckpt and rank == 0:
        os.makedirs(checkpoint_path)
        print(f"checkpoint saved in {checkpoint_path}")

    '''begain training'''
    best_dice = 0.0
    for epoch in range(args.ep):
        net.train()
        time_start = time.time()
        loss, dice_loss, focal_loss, mae_loss, bce_loss, agent_loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, rank=rank)
        loss_dict = {
            'train/loss': loss, 
            'train/dice loss': dice_loss, 
            'train/focal loss': focal_loss, 
            'train/mae_loss': mae_loss, 
            'train/bce_loss': bce_loss, 
            "train/actor_loss": agent_loss["actor_loss"],
            "train/lr": scheduler.get_last_lr()[0],
        }
        
        if args.wandb_enabled and loss is not None:
            wandb.log(loss_dict, step=epoch)
        time_end = time.time()
        print(loss_dict)
        print('time_for_training ', time_end - time_start)
        
        if args.distributed:
            torch.distributed.barrier()
        
        net.eval()
        new_best = False
        if epoch % args.val_freq == 0 or epoch == args.ep-1:
            
            iou, dice = function.validation_sam(args, nice_test_loader, epoch, net, rank=rank)

            if args.distributed:
                dist.all_reduce(iou), dist.all_reduce(dice)
                iou, dice = iou/world_size, dice/world_size
                print(f"val/IOU: {iou}, val/dice : {dice}")
            else:
                print(f"val/IOU: {iou}, val/dice : {dice}")
                
            if dice > best_dice:
                best_dice = dice
                new_best = True
            
            if args.wandb_enabled:
                wandb.log({'val/IOU' : iou, 'val/dice' : dice}, step=epoch)
        
        scheduler.step()
        
        if args.save_ckpt:
            if args.distributed and dist.get_rank() == 0:
                torch.save({
                    'model': net.module.state_dict(),
                    'agent': net.module.agent.q_net.module.state_dict(),
                },
                os.path.join(checkpoint_path, f"epoch_{epoch}_dice{dice:.4f}.pth"))
                
                if new_best:
                    print(f"Achieve best Dice: {dice} > {best_dice}")
                    torch.save({
                        'model': net.module.state_dict(),
                        'agent': net.module.agent.q_net.module.state_dict(),
                    },
                    os.path.join(checkpoint_path, f"best.pth"))
                    
            elif not args.distributed:
                torch.save({
                    'model': net.state_dict(),
                    'agent': net.agent.state_dict(),
                },
                os.path.join(checkpoint_path, f"epoch_{epoch}_dice{dice:.4f}.pth"))
                
                if new_best:
                    print(f"Achieve best Dice: {dice} > {best_dice}")
                    torch.save({
                        'model': net.state_dict(),
                        'agent': net.agent.state_dict(),
                    },
                    os.path.join(checkpoint_path, f"best.pth"))
            
    if args.distributed:
        cleanup()         

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