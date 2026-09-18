# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import time
import random
import pytz
import numpy as np
from datetime import datetime, timedelta
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

def setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    # Pin this process to its GPU before the process group is built. Without it every
    # rank's NCCL collectives and any device-less .cuda() default to device 0, which
    # either serialises the run onto one GPU or deadlocks.
    torch.cuda.set_device(rank)
    # The default 30-minute collective timeout is too tight here: on validation epochs
    # every other rank sits in evaluate()'s broadcast for as long as rank 0 takes to score
    # the entire split by itself, and rank 0 then writes checkpoints while they wait at the
    # end-of-epoch barrier.
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        timeout=timedelta(hours=6),
    )

def cleanup():
    dist.destroy_process_group()

def is_main(rank):
    return rank == 0

def reduce_mean(values, device, world_size):
    """Average a list of python scalars across every rank.

    All ranks must call this with the same-length list, which is why it is only used for
    the SAM2 losses -- those keys exist every epoch. The agent's metric dict does not:
    a rank whose replay buffer has not filled returns nothing, so reducing it key-by-key
    would deadlock on the ranks that do have keys.
    """
    reduced = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return (reduced / world_size).tolist()

def train(rank=0, world_size=1):
    args = cfg.parse_args()

    if args.distributed:
        setup(args, rank, world_size)
        GPUdevice = torch.device('cuda', rank)
    else:
        rank, world_size = 0, 1
        GPUdevice = torch.device('cuda', args.gpu_device)
        torch.cuda.set_device(GPUdevice)

    set_seed(args.seed + rank)
    
    args.wandb_enabled = args.wandb_enabled and is_main(rank)
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
        if is_main(rank):
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

    head, fix = [], []
    for k, v in net.named_parameters():
        (head if v.requires_grad else fix).append(v)

    if is_main(rank):
        print(f'Number of sam2 params: {n_parameters_tot:,}')
        print(f'Number of agent params: {agent_n_params:,}')
        print(f'Trainable parameters: {sum(p.numel() for p in head) + agent_n_params:,}')
        print(f'Parameters fixed: {sum(p.numel() for p in fix):,}')
        if args.distributed:
            print(f'Distributed over {world_size} GPU(s); evaluation runs on rank 0 alone')

    if args.distributed:
        net = DDP(net, device_ids=None, output_device=None, find_unused_parameters=True)
        if not args.no_agent:
            net.module.agent.to_distributed(rank=rank)
            if is_main(rank):
                print("Wrapped agent for distributed training")

    param_list = [{'params': head, 'initial_lr': args.lr}]
    optimizer = torch_optim.AdamW(param_list, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    splits = ("train", "val", "test") if args.eval_test else ("train", "val")
    nice_train_loader, nice_val_loader, nice_test_loader = get_dataloader(
        args, rank=rank, world_size=world_size, splits=splits,
    )
    if args.eval_test and args.test_ckpt == "best" and args.fold < 0 and rank == 0:
        print("WARNING: -eval_test -test_ckpt best with -fold < 0: there is no val split to "
              "select on, so the test pass scores the final epoch's weights instead")
    if nice_val_loader is None and rank == 0:
        print("WARNING: -fold < 0 leaves no validation split; nothing is evaluated during "
              "this run and best.pth will not be written -- only per-epoch checkpoints")

    if agent is not None and hasattr(agent, "set_lr_schedule"):
        agent_lr_T_max = args.agent_lr_T_max or (args.ep * len(nice_train_loader))
        agent.set_lr_schedule(agent_lr_T_max)
        if is_main(rank):
            print(f"Agent LR cosine schedule spans {agent_lr_T_max} update(s)")

    '''checkpoint path and tensorboard'''
    #create checkpoint folder to save model
    root_path = args.checkpoint_path
    current_time = datetime.now(pytz.timezone("Australia/Adelaide")).strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_path = os.path.join(root_path, current_time)
    if args.distributed:
        # Every rank timestamps itself, so without this the ranks disagree on the
        # directory and only rank 0's exists -- the path rank 0 prints would then not be
        # the one another rank means, and eval_3d.py would be pointed at nothing.
        shared = [checkpoint_path]
        dist.broadcast_object_list(shared, src=0)
        checkpoint_path = shared[0]
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
        # Each rank only saw its own shard of the epoch, so its losses describe a subset
        # of the data. Average them so the logged curve is the epoch's, not rank 0's.
        if args.distributed:
            loss, dice_loss, focal_loss, mae_loss, bce_loss, aux_loss = reduce_mean(
                [loss, dice_loss, focal_loss, mae_loss, bce_loss, aux_loss], GPUdevice, world_size,
            )
        loss_dict = {
            'train/loss': loss,
            'train/dice loss': dice_loss,
            'train/focal loss': focal_loss,
            'train/mae_loss': mae_loss,
            'train/bce_loss': bce_loss,
            'train/aux_loss': aux_loss,
            'train/lr': optimizer.param_groups[0]['lr'] if train_sam2 else 0.0,
        }

        if agent_loss:
            loss_dict.update({f"rl/{key}": value for key, value in agent_loss.items()})
        # No point annealing an LR that no longer drives any update.
        if train_sam2:
            scheduler.step()
        
        # NOTE: WANDB
        if args.wandb_enabled and loss is not None:
            wandb.log(loss_dict, step=epoch)
            
        time_end = time.time()
        if is_main(rank):
            print(loss_dict)
            print('time_for_training ', time_end - time_start)

        net.eval()
        new_best = False
        val_dice = None
        if nice_val_loader is not None and (epoch % args.val_freq == 0 or epoch == args.ep-1):
            iou, dice = evaluate(args, nice_val_loader, epoch, net, rank, world_size)
            val_dice = dice
            if rank == 0:
                print(f"val/IOU: {iou}, val/dice : {dice}")

            if dice > best_dice and rank==0:
                print(f"Achieve best Dice: {dice:4f} > {best_dice:4f}")
                best_dice = dice
                new_best = True

            # NOTE: WANDB
            if args.wandb_enabled:
                wandb.log({'val/IOU' : iou, 'val/dice' : dice}, step=epoch)

        if args.save_ckpt and rank == 0:
            # Only tag the filename with a score from this epoch -- on non-val epochs the
            # previous fix reused a stale dice from up to val_freq epochs earlier.
            tag = f"dice{val_dice:.4f}" if val_dice is not None else "noval"
            save_checkpoint(args, net, epoch, val_dice, os.path.join(checkpoint_path, f"epoch_{epoch}_{tag}.pth"))
            if new_best:
                save_checkpoint(args, net, epoch, val_dice, os.path.join(checkpoint_path, "best.pth"))

        if args.distributed:
            torch.distributed.barrier()

    # Training ends here. Whatever the test split is worth is read from here on only --
    # never inside the loop above -- so no test number can have moved an epoch, an LR or a
    # saved checkpoint. eval_3d.py remains the way to score a checkpoint in a separate job.
    if rank == 0:
        best_ckpt = os.path.join(checkpoint_path, "best.pth")
        if args.save_ckpt and os.path.exists(best_ckpt):
            print(f"Best val dice {best_dice:.4f}; selected checkpoint: {best_ckpt}")
        else:
            print(f"Best val dice {best_dice:.4f}; no best.pth written "
                  f"(checkpoints in {checkpoint_path})")
        if args.wandb_enabled:
            wandb.summary['val/best_dice'] = best_dice

    if args.eval_test:
        test_sam(args, nice_test_loader, net, rank, world_size, checkpoint_path)

    if args.distributed:
        cleanup()


def evaluate(args, loader, epoch, net, rank, world_size):
    """Score the whole validation split on one GPU and hand the result to every rank.

    Evaluation is not sharded. Splitting it across ranks gives each one a different set
    of (task, obj_id) classes to average over, and DistributedSampler pads the tail by
    repeating volumes, so the mean-of-means the old all_reduce computed drifted from the
    single-GPU number and drifted differently for every world size -- which is exactly
    what makes checkpoints picked by one run incomparable to another's. Rank 0 runs the
    full split against its (unwrapped) replica, which is parameter-identical to the
    others, and broadcasts the scores so the ranks stay in step for the barrier and the
    best-checkpoint decision below.
    """
    if not args.distributed:
        iou, dice = function.validation_sam(args, loader, epoch, net, rank=rank)
        return iou.item(), dice.item()

    scores = torch.zeros(2, dtype=torch.float32, device=torch.device('cuda', rank))
    if is_main(rank):
        iou, dice = function.validation_sam(
            args, loader, epoch, function.unwrap(net),
            rank=rank, device=torch.device('cuda', rank),
        )
        scores[0], scores[1] = iou, dice
    dist.broadcast(scores, src=0)
    return scores[0].item(), scores[1].item()


def test_sam(args, loader, net, rank, world_size, checkpoint_path):
    """Score the held-out test split once, after the last epoch.

    Two things stop this becoming a second selection signal. It runs after the training
    loop, so nothing it prints can have reached an epoch, the LR schedule or a saved
    checkpoint; and by default it scores the checkpoint *val* chose rather than whichever
    epoch happens to test best -- the run never gets to pick its test number. `-test_ckpt
    last` scores the final epoch's weights as they stand, which is also the fallback when
    -fold < 0 left no val split and hence no best.pth; that number is an unselected
    model's, and the run says so rather than passing it off as a selected one's.

    Like validation this is never sharded: rank 0 makes one pass over the whole split and
    broadcasts the result, for the reasons in evaluate()'s docstring.
    """
    if loader is None:
        if rank == 0:
            print("WARNING: -eval_test was passed but no test loader was built; nothing scored")
        return

    best_ckpt = os.path.join(checkpoint_path, "best.pth")
    if rank == 0:
        if args.test_ckpt == "best" and os.path.exists(best_ckpt):
            # Loaded on rank 0 alone: it is the only rank that scores, and training is
            # over, so the replicas are free to diverge from here.
            print(f"Test pass on the val-selected checkpoint {best_ckpt}")
            weights = torch.load(best_ckpt, map_location=torch.device('cuda', rank))
            target = function.unwrap(net)
            target.load_state_dict(weights["model"], strict=False)
            if "agent" in weights and not args.no_agent:
                target.agent.load_state_dict(weights["agent"])
        elif args.test_ckpt == "best":
            print(f"Test pass on the final epoch's weights: no {best_ckpt} to score "
                  f"(no val split, or -save_ckpt off). This model was never selected.")
        else:
            print("Test pass on the final epoch's weights (-test_ckpt last)")

    net.eval()
    iou, dice = evaluate(args, loader, args.ep, net, rank, world_size)
    if rank == 0:
        print(f"test/IOU: {iou}, test/dice : {dice}")
        if args.wandb_enabled:
            wandb.summary['test/IOU'] = iou
            wandb.summary['test/dice'] = dice


def save_checkpoint(args, net, epoch, dice, path):
    target = net.module if args.distributed else net
    ckpt = {'dice': dice, 'epoch': epoch, 'model': target.state_dict()}
    if not args.no_agent:
        ckpt['agent'] = target.agent.state_dict()
    torch.save(ckpt, path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def main():
    args = cfg.parse_args()
    set_seed(args.seed)

    if args.distributed:
        world_size = torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError("-distributed needs at least one visible CUDA device")
        if world_size == 1:
            print("WARNING: -distributed with a single visible GPU; set CUDA_VISIBLE_DEVICES "
                  "to the GPUs you want, or drop -distributed")
        print(f"Spawning {world_size} training rank(s) on port {args.master_port}")
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train()

if __name__ == '__main__':
    main()