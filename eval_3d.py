# eval_3d.py
#!/usr/bin/env	python3

""" Score a trained checkpoint on one split, on a single GPU.

    Evaluation deliberately never runs distributed, even when the checkpoint was trained
    with -distributed. Sharding the split across ranks gives each rank a different set of
    (task, obj_id) classes to average over, and DistributedSampler pads the tail by
    repeating volumes, so the reported dice would depend on how many GPUs happened to be
    visible. One GPU, one pass over the whole split, so two runs are comparable.

    Yunli Qi
"""

import os
import random

import numpy as np
import torch

import cfg
from func_3d import function
from func_3d.dataset import get_dataloader
from func_3d.utils import get_network


def evaluate(args):
    GPUdevice = torch.device('cuda', args.gpu_device)
    torch.cuda.set_device(GPUdevice)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=False)
    net.to(dtype=torch.bfloat16)
    agent = getattr(net, "agent", None)
    if agent is not None:
        agent.to_dtype(torch.bfloat16)

    if args.pretrain:
        print(args.pretrain)
        weights = torch.load(args.pretrain, map_location=GPUdevice)
        net.load_state_dict(weights["model"], strict=False)
        if "agent" in weights.keys() and not args.no_agent:
            net.agent.load_state_dict(weights["agent"])
            print("Loaded Agent weights")
        elif "q_agent" in weights.keys() and not args.no_agent:
            net.agent.load_state_dict(weights["q_agent"])
            print("Loaded DQN weights")

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(GPUdevice).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # This is the only place the held-out test manifest is ever scored -- train_3d.py
    # never loads it. Supports for the queries are drawn from the training fold, so pass
    # the same -fold/-n_folds/-split_seed/-fold_csv the checkpoint was trained with, or
    # the support pool will not be the data the model saw.
    _, nice_val_loader, nice_test_loader = get_dataloader(
        args, rank=0, world_size=1, splits=(args.eval_split,),
    )

    net.eval()

    loader = nice_val_loader if args.eval_split == "val" else nice_test_loader
    if loader is None:
        raise ValueError("-eval_split val requires -fold >= 0")

    iou, dice = function.validation_sam(args, loader, 0, net, rank=0, device=GPUdevice)

    print(f"{args.eval_split}/IOU: {iou.item()}, {args.eval_split}/dice : {dice.item()}")


def main():
    args = cfg.parse_args()
    if args.distributed:
        print("WARNING: ignoring -distributed; evaluation always runs on a single GPU so "
              "that scores stay comparable across runs. Use CUDA_VISIBLE_DEVICES or "
              "-gpu_device to choose which one.")
        args.distributed = False

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    evaluate(args)


if __name__ == '__main__':
    main()
