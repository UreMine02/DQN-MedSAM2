""" function for training and validation in one epoch
    Yunli Qi
"""
import os
import copy
import time
import random
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import draw_segmentation_masks, save_image

from monai.losses import DiceLoss

import cfg
from conf import settings
from func_3d.utils import (
    eval_seg, iou_score, CombinedLoss, update_loss, average_loss, update_score,
    average_score, extract_object, sample_diverse_support, calculate_bounding_box,
    extract_object_multiple
)
from func_3d.misc import MetricLogger, reduce_dict

import wandb

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
paper_loss = CombinedLoss(focal_weight=1, dice_weight=1)
seed = torch.randint(1,11,(1,7))

torch.backends.cudnn.benchmark = True
# scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def train_sam(args, net: nn.Module, optimizer, train_loader, epoch, rank=None):
    if args.distributed:
        # net = net.module
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)

    net.train()

    video_length = args.video_length
    train_agent = not args.no_agent
    agent_act = not args.no_agent # and epoch >= 0
    generate_rl_samples = not args.no_agent
    dice_loss_per_class = {}

    lossfunc = paper_loss
    aux_lossfunc = DiceLoss(sigmoid=False)

    # Record total loss thoroughout the entire epoch
    total_loss = {
        "total_loss": 0,
        "focal_loss": 0,
        "dice_loss": 0,
        "mae_loss": 0,
        "bce_loss": 0,
        "aux_loss": 0,
        "num_step": 0
    }
    target_class = 4
    agent_loss = {"actor_loss": 0, "critic_loss": 0}
    agent_step = 0
    metric_logger = MetricLogger(delimiter=" ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img', position=0, miniters=10) as pbar:
        for batch_idx, packs in enumerate(train_loader): #metric_logger.log_every(train_loader, print_freq, header=header):
            whole_imgs_tensor = packs["image"].squeeze(0)
            whole_masks_tensor = packs["label"].squeeze(0)
            whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
            whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
            task = packs["task"][0]

            obj_list = torch.unique(whole_masks_tensor)[1:].int().tolist()
            instance_loss = {
                "total_loss": 0,
                "focal_loss": 0,
                "dice_loss": 0,
                "mae_loss": 0,
                "bce_loss": 0,
                "aux_loss": 0,
                "num_step": 0,
            }
            # print(obj_list)
            for obj_id in obj_list:
                # pack = extract_object(whole_imgs_tensor, whole_masks_tensor, whole_support_imgs_tensor, whole_support_masks_tensor, \
                #                         obj_id=obj_id, video_length=args.video_length, num_support=args.num_support)
                # if pack is None:
                #     print(f"[PACK SKIP] obj_id={obj_id}\n")
                #     continue
                # torch.cuda.empty_cache()
                pack = {
                    "image": whole_imgs_tensor,
                    "label": whole_masks_tensor,
                    "support_image": whole_support_imgs_tensor,
                    "support_label": whole_support_masks_tensor,
                }
                if obj_id not in dice_loss_per_class.keys():
                    dice_loss_per_class[obj_id] = {"dice_loss":0, "num_step": 0}
                imgs_tensor = pack['image']
                masks_tensor = pack['label']

                support_imgs_tensor = pack["support_image"]
                support_masks_tensor = pack["support_label"]

                support_imgs_tensor = F.interpolate(support_imgs_tensor, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
                support_masks_tensor = F.interpolate(support_masks_tensor.unsqueeze(1), size=(args.image_size, args.image_size), mode="nearest").squeeze(1)

                # if imgs_tensor.numel() == 0 or masks_tensor.numel() == 0:
                #     print(f"[Query] Warning: Empty image or mask tensor for obj_id={obj_id} in {task}. Skipping...")
                #     continue  # Skip empty tensors
                # if support_imgs_tensor.numel() == 0 or support_masks_tensor.numel() == 0:
                #     print(f"[Support] Warning: Empty support image or mask tensor for obj_id={obj_id} in {task}. Skipping...")
                #     continue

                rounded_length = (pack['image'].shape[0] // args.video_length) * args.video_length
                start_slice = random.randint(0, pack['image'].shape[0] - rounded_length)
                sliding_window = [
                    slice(i, i+args.video_length) 
                    for i in range(start_slice, start_slice+rounded_length, args.video_length)
                ]
                
                # local_size = len(sliding_window)
                if args.distributed:
                    local_size = torch.tensor([len(sliding_window)], device=GPUdevice)
                    dist.all_reduce(local_size, op=dist.ReduceOp.MIN)
                    sliding_window = sliding_window[:local_size]
                    # print(batch_idx, rank, len(sliding_window), local_size)
                
                for slide_idx, slide in enumerate(sliding_window):
                    slide_imgs_tensor = imgs_tensor[slide].to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
                    slide_masks_tensor = masks_tensor[slide].to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
                    slide_imgs_tensor = F.interpolate(slide_imgs_tensor, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
                    slide_masks_tensor = F.interpolate(slide_masks_tensor.unsqueeze(1), size=(args.image_size, args.image_size), mode="nearest").squeeze(1)
                    if not args.distributed:
                        train_state = net.train_init_state(
                            args=args,
                            imgs_tensor=slide_imgs_tensor, masks_tensor=slide_masks_tensor, support_imgs_tensor=support_imgs_tensor
                        )
                    else:
                        train_state = net.module.train_init_state(
                            args=args,
                            imgs_tensor=slide_imgs_tensor, masks_tensor=slide_masks_tensor, support_imgs_tensor=support_imgs_tensor
                        )

                    with torch.cuda.amp.autocast():
                        video_segments = net(
                            slide_imgs_tensor, slide_masks_tensor,
                            support_masks_tensor, train_state,
                            obj_id,
                            train_agent=train_agent,
                            agent_act=agent_act,
                            generate_rl_samples=generate_rl_samples,
                            start_trajectory=(slide_idx == 0),
                            end_trajectory=(slide_idx == len(sliding_window)-1),
                            device=GPUdevice
                        )
                        # Record the loss in this step
                        class_loss = {
                            "total_loss":0,
                            "focal_loss": 0,
                            "dice_loss": 0,
                            "mae_loss": 0,
                            "bce_loss": 0,
                            "aux_loss": 0,
                            "num_step": 0
                        }

                        for frame_idx in video_segments.keys():
                            pred = video_segments[frame_idx][obj_id]["pred_mask"].squeeze(0)
                            mask = video_segments[frame_idx][obj_id]["image_label"]
                            if mask is not None:
                                mask = mask == obj_id
                                mask = mask.to(dtype=torch.float32, device=GPUdevice)
                            else:
                                mask = torch.zeros_like(pred).to(device=GPUdevice)

                            # NOTE: TEST AUXILIARY LOSS
                            if args.auxiliary_loss == "dice":
                                cond_gating_score = video_segments[frame_idx][obj_id]["gating_score_dict"]["cond_frames"]
                                cond_gating_score = F.interpolate(cond_gating_score, size=support_masks_tensor.shape[-2:], mode="nearest")
                                aux_loss = aux_lossfunc(cond_gating_score, support_masks_tensor.unsqueeze(0))

                                non_cond_gating_score = video_segments[frame_idx][obj_id]["gating_score_dict"]["non_cond_frames"].values()
                                non_cond_gating_score = list(non_cond_gating_score)
                                if len(non_cond_gating_score) > 0:
                                    non_cond_gating_score = torch.cat(list(non_cond_gating_score), dim=0).unsqueeze(0)
                                    aux_label = []
                                    for prev_frame_idx in video_segments[frame_idx][obj_id]["gating_score_dict"]["non_cond_frames"].keys():
                                        aux_label.append(video_segments[prev_frame_idx][obj_id]["pred_mask"])
                                    aux_label = torch.cat(aux_label, dim=0).unsqueeze(0)

                                    non_cond_gating_score = F.interpolate(non_cond_gating_score, size=aux_label.shape[-2:], mode="nearest")
                                    aux_loss += aux_lossfunc(non_cond_gating_score, aux_label)

                                aux_loss = 0.2 * aux_loss
                            else:
                                aux_loss = torch.Tensor([0]).to(device=GPUdevice)

                            # Calculate the loss
                            obj_pred = video_segments[frame_idx][obj_id]["object_score_logits"]
                            iou_pred = video_segments[frame_idx][obj_id]["iou"]
                            pred_mask = (torch.sigmoid(pred.detach()) > 0.5).float()
                            iou_gt = iou_score(pred_mask, mask, smoothing=1e-8)
                            dice_loss, focal_loss, mae_loss, bce_loss = lossfunc(pred, mask, iou_pred, iou_gt.reshape(1), obj_pred)
                            class_loss["num_step"] += 1
                            # Update the loss of the class
                            update_loss(class_loss, focal_loss, dice_loss, mae_loss, bce_loss, aux_loss)

                            dice_loss_per_class[obj_id]["dice_loss"] += dice_loss.item()
                            dice_loss_per_class[obj_id]["num_step"] += 1

                        accum_step = 1
                        # Average loss of this class
                        average_loss(class_loss)
                        avg_loss = class_loss["total_loss"] / accum_step
                        avg_loss.backward()
                        for k, v in class_loss.items():
                            print(k,v)
                        
                        for name, param in net.named_parameters():
                            if param.grad is None:
                                continue
                            
                            if param.grad.isnan().any():
                                raise AssertionError(f"{name} grad is nan")


                        if (batch_idx + 1) % accum_step == 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
                            optimizer.step()
                            optimizer.zero_grad()

                    # to_reduce = {k: class_loss[k] for k in class_loss.keys() if k not in ["num_step", "total_loss"]}
                    # losses_reduced = reduce_dict(to_reduce)
                    # loss_value = sum(losses_reduced.values()).item()

                    # metric_logger.update(loss=loss_value, **losses_reduced)
                    # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                    # metric_logger.update(grad_norm=grad_total_norm)

                    if not args.distributed:
                        agent = getattr(net, "agent", None)
                    else:
                        agent = getattr(net.module, "agent", None)

                    if agent is not None:
                        q_updates_per_step = getattr(args, "q_updates_per_step", 0)
                        agent_step_loss = agent.update(q_updates_per_step)
                        if agent_step_loss is not None:
                            # metric_logger.update(actor_loss=agent_step_loss["actor_loss"].item())
                            # metric_logger.update(actor_gradnorm=agent_step_loss["actor_gradnorm"].item())
                            agent_loss["actor_loss"] += agent_step_loss["actor_loss"]
                            if "critic_loss" in agent_step_loss.keys():
                                # metric_logger.update(critic_loss=agent_step_loss["critic_loss"].item())
                                # metric_logger.update(critic_gradnorm=agent_step_loss["critic_gradnorm"].item())
                                agent_loss["critic_loss"] += agent_step_loss["critic_loss"]
                            agent_step += 1

                    # Add the loss of the class to the instance
                    update_loss(
                        instance_loss,
                        class_loss["focal_loss"].item(),
                        class_loss["dice_loss"].item(),
                        class_loss["mae_loss"].item(),
                        class_loss["bce_loss"].item(),
                        class_loss["aux_loss"].item(),
                    )
                    instance_loss["num_step"] += 1

            average_loss(instance_loss)

            update_loss(total_loss,
                instance_loss["focal_loss"],
                instance_loss["dice_loss"],
                instance_loss["mae_loss"],
                instance_loss["bce_loss"],
                instance_loss["aux_loss"],
            )
            total_loss["num_step"] += 1
            pbar.update()

    average_loss(total_loss)
    dice_loss_per_class = {f"{class_}":dice_loss_output["dice_loss"]/dice_loss_output["num_step"] for class_, dice_loss_output in dice_loss_per_class.items()}

    if agent_step > 0:
        avg_agent_loss = {}
        avg_agent_loss["actor_loss"] = agent_loss["actor_loss"] / agent_step
        avg_agent_loss["critic_loss"] = agent_loss["critic_loss"] / agent_step
    else:
        avg_agent_loss = {
            "actor_loss": 0,
            "critic_loss": 0,
        }

    return (
        total_loss["total_loss"],
        total_loss["dice_loss"],
        total_loss["focal_loss"],
        total_loss["mae_loss"],
        total_loss["bce_loss"],
        total_loss["aux_loss"],
        avg_agent_loss
    )

def validation_sam(args, val_loader, epoch, net: nn.Module, inferencing=False, clean_dir=True, rank=None):
    if args.distributed:
        # net = net.module
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)

    # eval mode
    net.eval()
    n_val = len(val_loader)

    total_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
    ths = np.arange(0, 1.0, 0.001)
    score_per_class = {}
    masks = {}
    preds = {}
    agent_act = not args.no_agent
    # lossfunc = paper_loss

    for packs in val_loader:
        whole_imgs_tensor = packs["image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
        whole_masks_tensor = packs["label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
        whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
        whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
        task = packs["task"][0]
        name = packs["name"][0]
        # cls_id = packs["obj_id"][0]
        # Log initial slice stats for validation
        # print(f"[VALIDATION PACK] Name: {name}")
        # print(f"  Query Total Slices: {whole_masks_tensor.shape[0]}, Classes: {torch.unique(whole_masks_tensor)}")
        # print(f"  Support Total Slices: {whole_support_masks_tensor.shape[0]}, Classes: {torch.unique(whole_support_masks_tensor)}")

        obj_list = torch.unique(whole_masks_tensor)[1:].int().tolist()
        instance_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
        for obj_id in obj_list:
            pack = extract_object(whole_imgs_tensor, whole_masks_tensor, whole_support_imgs_tensor, whole_support_masks_tensor, \
                                    obj_id=obj_id, video_length=None, num_support=args.num_support)

            pack = {
                "image": whole_imgs_tensor,
                "label": whole_masks_tensor,
                "support_image": whole_support_imgs_tensor,
                "support_label": whole_support_masks_tensor,
            }
            # if pack is None:
            #     print(f"[Validation] [PACK]: No valid for pack for obj_id={obj_id}. Skipping...")
            #     # print(f"[DEBUG - QUERY] Slices: {whole_imgs_tensor.shape[0]}, Unique Classes: {torch.unique(whole_masks_tensor)}")
            #     # print(f"[DEBUG - SUPPORT] Slices: {whole_support_imgs_tensor.shape[0]}, Unique Classes: {torch.unique(whole_support_masks_tensor)}")
            #     continue
            if f"{task}_{obj_id}" not in score_per_class.keys():
                score_per_class[f"{task}_{obj_id}"] = {
                    "iou": torch.FloatTensor([]).to(device=GPUdevice),
                    "dice": torch.FloatTensor([]).to(device=GPUdevice),
                    "fb_iou": torch.FloatTensor([]).to(device=GPUdevice),
                }
                preds[f"{task}_{obj_id}"] = []
                masks[f"{task}_{obj_id}"] = []

            imgs_tensor = pack['image']
            masks_tensor = pack['label']

            support_imgs_tensor = pack["support_image"]
            support_masks_tensor = pack["support_label"]
            
            support_imgs_tensor = F.interpolate(support_imgs_tensor, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            support_masks_tensor = F.interpolate(support_masks_tensor.unsqueeze(1), size=(args.image_size, args.image_size), mode="nearest").squeeze(1)

            imgs_tensor = F.interpolate(imgs_tensor, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            masks_tensor = F.interpolate(masks_tensor.unsqueeze(1), size=(args.image_size, args.image_size), mode="nearest").squeeze(1)
                        
            # support_bbox_dict = pack["support_bbox"]
            # if imgs_tensor.numel() == 0 or masks_tensor.numel() == 0:
            #     print(f"VALIDATION: [Query] Warning: Empty image or mask tensor for obj_id={obj_id} in {task}. Skipping...")
            #     continue  # Skip empty tensors

            # if support_imgs_tensor.numel() == 0 or support_masks_tensor.numel() == 0:
            #     print(f"VALIDATION: [Support] Warning: Empty support image or mask tensor for obj_id={obj_id} in {task}. Skipping...")
            #     continue

            if not args.distributed:
                train_state = net.val_init_state(
                    args=args,
                    imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor
                )
            else:
                train_state = net.module.val_init_state(
                    args=args,
                    imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor
                )

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # for frame_idx in range(support_masks_tensor.shape[0]):
                    #     mask = support_masks_tensor[frame_idx]
                    #     _, _, _ = net.train_add_new_mask(
                    #         inference_state=train_state,
                    #         frame_idx=frame_idx,
                    #         obj_id=obj_id,
                    #         mask=mask.to(device=GPUdevice),
                    #     )

                    # video_segments = {}  # video_segments contains the per-frame segmentation results

                    # for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state, agent_act=agent_act):
                    #     video_segments[out_frame_idx] = {
                    #         out_obj_id: {"image_tensor": imgs_tensor[out_frame_idx], "image_label" : masks_tensor[out_frame_idx],
                    #         "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i]}
                    #         for i, out_obj_id in enumerate(out_obj_ids)
                    #     }

                    video_segments = net(imgs_tensor, masks_tensor, support_masks_tensor, train_state, obj_id, agent_act=agent_act, device=GPUdevice)
            # Record the loss in this step
            for frame_idx in video_segments.keys():
                pred = video_segments[frame_idx][obj_id]["pred_mask"].squeeze(0)
                mask = video_segments[frame_idx][obj_id]["image_label"]
                pred_mask = torch.where(torch.sigmoid(pred) >= 0.5, 1, 0)
                if mask is not None:
                    mask = mask == obj_id
                    mask = mask.to(dtype=torch.float32, device=GPUdevice)

                    (
                        iou,
                        dice,
                        fb_iou,
                    ) = eval_seg(pred, mask)

                    score_dict = score_per_class[f"{task}_{obj_id}"]

                    score_dict["iou"] = torch.cat([score_dict["iou"], iou.detach()])
                    score_dict["dice"] = torch.cat([score_dict["dice"], dice.detach()])
                    score_dict["fb_iou"] = torch.cat([score_dict["fb_iou"], fb_iou.detach()])
                else:
                    mask = torch.zeros_like(pred).to(device=GPUdevice, dtype=torch.float32)

    avg = {
        "iou": torch.FloatTensor([]).to(device=GPUdevice),
        "dice": torch.FloatTensor([]).to(device=GPUdevice),
        "fb_iou": torch.FloatTensor([]).to(device=GPUdevice),
    }

    table_data = []

    for name, metrics_dict in score_per_class.items():
        table_data.append((
            name,
            metrics_dict["iou"].mean(),
            metrics_dict["dice"].mean(),
            metrics_dict["fb_iou"].mean(),
            0.5
        ))

        avg["iou"] = torch.cat([avg["iou"], metrics_dict["iou"].mean(dim=0, keepdim=True)])
        avg["dice"] = torch.cat([avg["dice"], metrics_dict["dice"].mean(dim=0, keepdim=True)])
        avg["fb_iou"] = torch.cat([avg["fb_iou"], metrics_dict["fb_iou"].mean(dim=0, keepdim=True)])
        avg["th"] = 0.5

        if args.wandb_enabled:
            wandb.log({f"val/{name}.Dice": metrics_dict["dice"].mean()}, step=epoch)

    avg["iou"] = avg["iou"].mean()
    avg["dice"] = avg["dice"].mean()
    avg["fb_iou"] = avg["fb_iou"].mean()

    table_data.append((
        "Average",
        avg["iou"].item(),
        avg["dice"].item(),
        avg["fb_iou"].item(),
        avg["th"]
    ))

    print(tabulate(table_data, headers=["name", "iou", "dice", "fb_iou", "th"], floatfmt=".4f", tablefmt="grid"))

    return avg['iou'], avg['dice']