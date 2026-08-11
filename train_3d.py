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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR

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
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    nice_train_loader, nice_test_loader = get_dataloader(args, rank=rank, world_size=world_size)

    '''checkpoint path and tensorboard'''
    #create checkpoint folder to save model
    root_path = args.checkpoint_path
    current_time = datetime.now(pytz.timezone("Australia/Adelaide")).strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_path = os.path.join(root_path, current_time)
    if not os.path.exists(checkpoint_path) and args.save_ckpt and rank == 0:
        os.makedirs(checkpoint_path)
        print(f"checkpoint saved in {checkpoint_path}")

    if args.stop_sam2_ep >= 0 and rank == 0:
        if agent is None or args.no_agent:
            print(f"WARNING: -stop_sam2_ep {args.stop_sam2_ep} without an agent: nothing is trained from that epoch on")
        elif args.stop_sam2_ep <= args.warmup_ep:
            print(f"WARNING: -stop_sam2_ep {args.stop_sam2_ep} <= -warmup_ep {args.warmup_ep}: "
                  f"SAM2 stops before the agent starts training")

    '''begin training'''
    best_dice = 0.0
    for epoch in range(args.ep):
        # From stop_sam2_ep on, SAM2 is frozen and only the RL agent keeps training.
        # It also switches to eval() so the agent learns against the same forward path
        # validation uses (memory-mask binarization, obj-ptr selection, mask-decoder
        # stability branch, no dropout).
        train_sam2 = args.stop_sam2_ep < 0 or epoch < args.stop_sam2_ep
        if not train_sam2 and epoch == args.stop_sam2_ep:
            for param in net.parameters():
                param.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            if rank == 0:
                print(f"Epoch {epoch}: SAM2 frozen and set to eval, training the RL agent only")

        net.train() if train_sam2 else net.eval()
        if args.distributed:
            nice_train_loader.sampler.set_epoch(epoch)
        #     net.module.image_encoder.eval()
        #     net.module.sam_prompt_encoder.eval()
        # else:
        #     net.image_encoder.eval()
        #     net.sam_prompt_encoder.eval()

        if agent is not None:
            agent.set_epoch(epoch, distributed=args.distributed)

        time_start = time.time()
        (
            loss,
            dice_loss,
            focal_loss,
            mae_loss,
            bce_loss,
            aux_loss,
            agent_loss
        ) = function.train_sam(args, net, optimizer, nice_train_loader, epoch, rank=rank, train_sam2=train_sam2)
        loss_dict = {
            'train/loss': loss,
            'train/dice loss': dice_loss,
            'train/focal loss': focal_loss,
            'train/mae_loss': mae_loss,
            'train/bce_loss': bce_loss,
            'train/aux_loss': aux_loss,
            'train/lr': optimizer.param_groups[0]['lr'] if train_sam2 else 0.0,
        }
        # RL diagnostics: whatever the agent's update() returned this epoch (actor/critic
        # loss, PPO ratio/entropy/KL/explained-variance/advantage stats, episodic return
        # stats, ...), logged under rl/* rather than train/* since they're not SAM2
        # training signals. agent_loss is empty when the agent never ran an update this
        # epoch (still warming up, or the replay buffer hasn't filled) -- skip logging
        # entirely rather than log a stale/flat 0 under rl/*.
        if agent_loss:
            loss_dict.update({f"rl/{key}": value for key, value in agent_loss.items()})
        # No point annealing an LR that no longer drives any update.
        if train_sam2:
            scheduler.step()
        
        # NOTE: WANDB
        if args.wandb_enabled and loss is not None:
            wandb.log(loss_dict, step=epoch)
            
        time_end = time.time()
        print(loss_dict)
        print('time_for_training ', time_end - time_start)

        net.eval()
        new_best = False
        if epoch % args.val_freq == 0 or epoch == args.ep-1:
            iou, dice = function.validation_sam(args, nice_test_loader, epoch, net, rank=rank)
            # iou, dice = net(args, nice_test_loader, epoch, net, rank=rank, training=False)

            if args.distributed:
                dist.all_reduce(iou), dist.all_reduce(dice)
                iou, dice = iou.item(), dice.item()
                iou, dice = iou/world_size, dice/world_size
                if rank == 0:
                    print(f"val/IOU: {iou}, val/dice : {dice}")
            else:
                iou, dice = iou.item(), dice.item()
                print(f"val/IOU: {iou}, val/dice : {dice}")

            if dice > best_dice and rank==0:
                print(f"Achieve best Dice: {dice:4f} > {best_dice:4f}")
                best_dice = dice
                new_best = True
            
            # NOTE: WANDB
            if args.wandb_enabled:
                wandb.log({'val/IOU' : iou, 'val/dice' : dice}, step=epoch)
            
        if args.save_ckpt:
            if args.distributed and rank == 0:
                ckpt = {
                    'dice': dice,
                    'epoch': epoch,
                    'model': net.module.state_dict(),
                }
                if not args.no_agent:
                    ckpt['agent'] = net.module.agent.state_dict()
                torch.save(ckpt, os.path.join(checkpoint_path, f"epoch_{epoch}_dice{dice:.4f}.pth"))

                if new_best:
                    torch.save(ckpt, os.path.join(checkpoint_path, f"best.pth"))

            elif not args.distributed:
                ckpt = {
                    'dice': dice,
                    'epoch': epoch,
                    'model': net.state_dict(),
                }
                if not args.no_agent:
                    ckpt['agent'] = net.agent.state_dict()

                torch.save(ckpt, os.path.join(checkpoint_path, f"epoch_{epoch}_dice{dice:.4f}.pth"))

                if new_best:
                    torch.save(ckpt, os.path.join(checkpoint_path, f"best.pth"))

        if args.distributed:
            torch.distributed.barrier()

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