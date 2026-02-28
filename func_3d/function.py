""" function for training and validation in one epoch
    Yunli Qi
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchshow as ts
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
import wandb

import cfg
from conf import settings
from func_3d.utils import (
    eval_seg, iou_score, CombinedLoss, update_loss, average_loss, update_score,
    average_score, extract_object, sample_diverse_support, calculate_bounding_box,
    extract_object_multiple
)
from func_3d.misc import MetricLogger, reduce_dict

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
paper_loss = CombinedLoss(focal_weight=20, dice_weight=1)
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
        net = net.module
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

    # Record total loss thoroughout the entire epoch
    total_loss = {"total_loss": 0, "focal_loss": 0, "dice_loss": 0, "mae_loss": 0, "bce_loss": 0, "num_step": 0}
    target_class = 4
    agent_loss = {"actor_loss": 0, "critic_loss": 0}
    agent_step = 0
    metric_logger = MetricLogger(delimiter=" ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img', position=0) as pbar:
    for packs in metric_logger.log_every(train_loader, print_freq, header=header):
        whole_imgs_tensor = packs["image"].squeeze(0).to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
        whole_masks_tensor = packs["label"].squeeze(0).to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
        whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
        whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype=torch.float32, device=GPUdevice, non_blocking=True)
        task = packs["task"][0]

        obj_list = torch.unique(whole_masks_tensor)[1:].int().tolist()
        instance_loss = {"total_loss": 0, "focal_loss": 0, "dice_loss": 0, "mae_loss": 0, "bce_loss": 0, "num_step": 0}
        for obj_id in obj_list:
            pack = extract_object(whole_imgs_tensor, whole_masks_tensor, whole_support_imgs_tensor, whole_support_masks_tensor, \
                                    obj_id=obj_id, video_length=args.video_length, num_support=args.num_support)
            if pack is None:
                print(f"[PACK SKIP] obj_id={obj_id}\n")
                continue
            # torch.cuda.empty_cache()
            if obj_id not in dice_loss_per_class.keys():
                dice_loss_per_class[obj_id] = {"dice_loss":0, "num_step": 0}
            imgs_tensor = pack['image']
            masks_tensor = pack['label']

            support_imgs_tensor = pack["support_image"]
            support_masks_tensor = pack["support_label"]
            if imgs_tensor.numel() == 0 or masks_tensor.numel() == 0:
                print(f"[Query] Warning: Empty image or mask tensor for obj_id={obj_id} in {task}. Skipping...")
                continue  # Skip empty tensors
            if support_imgs_tensor.numel() == 0 or support_masks_tensor.numel() == 0:
                print(f"[Support] Warning: Empty support image or mask tensor for obj_id={obj_id} in {task}. Skipping...")
                continue

            train_state = net.train_init_state(
                args=args,
                imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor
            )

            with torch.cuda.amp.autocast():
                for frame_idx in range(support_masks_tensor.shape[0]):
                    mask = support_masks_tensor[frame_idx]
                    _, _, _ = net.train_add_new_mask(
                        inference_state=train_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=mask.to(device=GPUdevice),
                    )

                video_segments = {}  # video_segments contains the per-frame segmentation results

                for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state, train_agent=train_agent, agent_act=agent_act, generate_rl_samples=generate_rl_samples):
                    video_segments[out_frame_idx] = {
                        out_obj_id: {"image_tensor": imgs_tensor[out_frame_idx], "image_label" : masks_tensor[out_frame_idx],
                        "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i]}
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # Record the loss in this step
                class_loss = {"total_loss":0, "focal_loss": 0, "dice_loss": 0, "mae_loss": 0, "bce_loss": 0, "num_step": 0}

                for frame_idx in video_segments.keys():
                    pred = video_segments[frame_idx][obj_id]["pred_mask"].squeeze(0)
                    mask = video_segments[frame_idx][obj_id]["image_label"]
                    if mask is not None:
                        mask = mask.to(dtype=torch.float32, device=GPUdevice)
                    else:
                        mask = torch.zeros_like(pred).to(device=GPUdevice)
                    # Calculate the loss
                    obj_pred = video_segments[frame_idx][obj_id]["object_score_logits"]
                    iou_pred = video_segments[frame_idx][obj_id]["iou"]
                    pred_mask = (torch.sigmoid(pred) > 0.5).float()
                    assert not pred_mask.isnan().any()
                    iou_gt = iou_score(pred_mask, mask)
                    dice_loss, focal_loss, mae_loss, bce_loss = lossfunc(pred, mask, iou_pred, iou_gt.reshape(1), obj_pred)
                    class_loss["num_step"] += 1
                    # Update the loss of the class
                    update_loss(class_loss, focal_loss, dice_loss, mae_loss, bce_loss)

                    dice_loss_per_class[obj_id]["dice_loss"] += dice_loss.item()
                    dice_loss_per_class[obj_id]["num_step"] += 1

                # Average loss of this class
                average_loss(class_loss)
                avg_loss = class_loss["total_loss"]
                # avg_loss = class_loss["focal_loss"] + class_loss["dice_loss"] + class_loss["mae_loss"] 

                to_reduce = {k: class_loss[k] for k in class_loss.keys() if k not in ["num_step", "total_loss"]}
                losses_reduced = reduce_dict(to_reduce)
                loss_value = sum(losses_reduced.values()).item()

                optimizer.zero_grad()
                avg_loss.backward()
                grad_total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
                optimizer.step()

                metric_logger.update(loss=loss_value, **losses_reduced)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                metric_logger.update(grad_norm=grad_total_norm)

                agent = getattr(net, "agent", None)
                if agent is not None and epoch >= args.warmup_ep:
                    q_updates_per_step = getattr(args, "q_updates_per_step", 0)
                    agent_step_loss = agent.update(q_updates_per_step)
                    if agent_step_loss is not None:
                        metric_logger.update(actor_loss=agent_step_loss["actor_loss"].item())
                        metric_logger.update(actor_gradnorm=agent_step_loss["actor_gradnorm"].item())
                        agent_loss["actor_loss"] += agent_step_loss["actor_loss"]
                        if "critic_loss" in agent_step_loss.keys():
                            metric_logger.update(critic_loss=agent_step_loss["critic_loss"].item())
                            metric_logger.update(critic_gradnorm=agent_step_loss["critic_gradnorm"].item())
                            agent_loss["critic_loss"] += agent_step_loss["critic_loss"]
                        agent_step += 1

                # Add the loss of the class to the instance
                update_loss(
                    instance_loss,
                    class_loss["focal_loss"].item(),
                    class_loss["dice_loss"].item(),
                    class_loss["mae_loss"].item(),
                    class_loss["bce_loss"].item()
                )
                instance_loss["num_step"] += 1

        average_loss(instance_loss)

        update_loss(total_loss,
            instance_loss["focal_loss"],
            instance_loss["dice_loss"],
            instance_loss["mae_loss"],
            instance_loss["bce_loss"]
        )
        total_loss["num_step"] += 1
        # pbar.update()

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

    return total_loss["total_loss"], total_loss["dice_loss"], total_loss["focal_loss"], total_loss["mae_loss"], total_loss["bce_loss"], avg_agent_loss

def validation_sam(args, val_loader, epoch, net: nn.Module, inferencing=False, clean_dir=True, rank=None):
    if args.distributed:
        net = net.module
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)

    # eval mode
    net.eval()
    n_val = len(val_loader)

    total_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
    score_per_class = {}
    masks = {}
    preds = {}
    agent_act = not args.no_agent
    # lossfunc = paper_loss

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for packs in val_loader:
            whole_imgs_tensor = packs["image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_masks_tensor = packs["label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            task = packs["task"][0]
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
                if pack is None:
                    print(f"[Validation] [PACK]: No valid for pack for obj_id={obj_id}. Skipping...")
                    # print(f"[DEBUG - QUERY] Slices: {whole_imgs_tensor.shape[0]}, Unique Classes: {torch.unique(whole_masks_tensor)}")
                    # print(f"[DEBUG - SUPPORT] Slices: {whole_support_imgs_tensor.shape[0]}, Unique Classes: {torch.unique(whole_support_masks_tensor)}")
                    continue
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

                # selected_support_frames = torch.randint(0, len(masks_tensor), size=(10,)).tolist()
                # support_imgs_tensor = pack["image"][selected_support_frames]
                # support_masks_tensor = pack["label"][selected_support_frames]

                support_imgs_tensor = pack["support_image"]
                support_masks_tensor = pack["support_label"]
                # support_bbox_dict = pack["support_bbox"]
                if imgs_tensor.numel() == 0 or masks_tensor.numel() == 0:
                    print(f"VALIDATION: [Query] Warning: Empty image or mask tensor for obj_id={obj_id} in {task}. Skipping...")
                    continue  # Skip empty tensors

                if support_imgs_tensor.numel() == 0 or support_masks_tensor.numel() == 0:
                    print(f"VALIDATION: [Support] Warning: Empty support image or mask tensor for obj_id={obj_id} in {task}. Skipping...")
                    continue

                train_state = net.val_init_state(
                    args=args,
                    imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor
                )

                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        for frame_idx in range(support_masks_tensor.shape[0]):
                            mask = support_masks_tensor[frame_idx]
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                mask=mask.to(device=GPUdevice),
                            )

                        video_segments = {}  # video_segments contains the per-frame segmentation results

                        for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state, agent_act=agent_act):
                            video_segments[out_frame_idx] = {
                                out_obj_id: {"image_tensor": imgs_tensor[out_frame_idx], "image_label" : masks_tensor[out_frame_idx],
                                "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i]}
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }

                # Record the loss in this step
                volume_masks = []
                volume_preds = []
                for frame_idx in video_segments.keys():
                    pred = video_segments[frame_idx][obj_id]["pred_mask"].squeeze(0)
                    mask = video_segments[frame_idx][obj_id]["image_label"]
                    pred_mask = torch.where(torch.sigmoid(pred) >= 0.5, 1, 0)
                    if mask is not None:
                        mask = mask.to(dtype=torch.float32, device=GPUdevice)
                        # masks[f"{task}_{obj_id}"].append(mask.cpu())
                        # preds[f"{task}_{obj_id}"].append(pred.cpu())
                        
                        volume_masks.append(mask.cpu())
                        volume_preds.append(pred.cpu())
                        
                        (
                            iou,
                            dice,
                            fb_iou,
                        ) = eval_seg(pred, mask)

                        # score_dict = score_per_class[f"{task}_{obj_id}"]

                        # score_dict["iou"] = torch.cat([score_dict["iou"], iou.detach()])
                        # score_dict["dice"] = torch.cat([score_dict["dice"], dice.detach()])
                        # score_dict["fb_iou"] = torch.cat([score_dict["fb_iou"], fb_iou.detach()])
                    else:
                        mask = torch.zeros_like(pred).to(device=GPUdevice)

                    if args.vis:
                        save_dir = "/".join(args.pretrain.split("/")[:-1])
                        save_prefix = f"{save_dir}/vis/{packs['name'][0]}_{obj_id}_idx{frame_idx}_dice{dice.item():.4f}"
                        mask *= 2
                        ts.overlay(
                            [imgs_tensor[frame_idx], pred_mask, mask], [1, 0.4, 0.4],
                            save_as=save_prefix + ".png",
                            cmap="jet"
                        )
                
                volume_masks = torch.stack(volume_masks).flatten(1) # [D,H,W]
                volume_preds = torch.stack(volume_preds).flatten(1) # [D,H,W]
                
                masks[f"{task}_{obj_id}"].append(volume_masks)
                preds[f"{task}_{obj_id}"].append(volume_preds)
                
                # print(f"Name: {task}_{obj_id} Dice score: {dice_score} IoU score: {iou_score}")

            pbar.update()

    ths = np.arange(0, 1.0, 0.01)
    # ths = [0.5]
    for name in preds.keys():
        best_iou = 0
        best_dice = 0
        best_fbiou = 0
        best_th = ths[0]
        
        for th in ths:
            ious = torch.FloatTensor([]).to(device=GPUdevice)
            dices = torch.FloatTensor([]).to(device=GPUdevice)
            fb_ious = torch.FloatTensor([]).to(device=GPUdevice)
            
            for i in range(len(preds[name])):
                pred = preds[name][i].to(GPUdevice)
                mask = masks[name][i].to(GPUdevice)
                iou, dice, fb_iou = eval_seg(pred, mask, thr=th)
                
                ious = torch.cat([ious, iou])
                dices = torch.cat([dices, dice])
                fb_ious = torch.cat([fb_ious, fb_iou])
            
            # print(ious)
            ious = ious.mean(dim=0, keepdim=True)
            dices = dices.mean(dim=0, keepdim=True)
            fb_ious = fb_ious.mean(dim=0, keepdim=True)
            # print(th, ious)
            
            if dices > best_dice:
                best_iou = ious
                best_dice = dices
                best_fbiou = fb_ious
                best_th = th
                
        score_per_class[name]["iou"] = best_iou
        score_per_class[name]["dice"] = best_dice
        score_per_class[name]["fb_iou"] = best_fbiou
        score_per_class[name]["th"] = best_th

    avg = {
        "iou": torch.FloatTensor([]).to(device=GPUdevice),
        "dice": torch.FloatTensor([]).to(device=GPUdevice),
        "fb_iou": torch.FloatTensor([]).to(device=GPUdevice),
    }

    table_data = []

    for name, metrics_dict in score_per_class.items():
        table_data.append((
            name,
            metrics_dict["iou"],
            metrics_dict["dice"],
            metrics_dict["fb_iou"],
            metrics_dict["th"]
        ))

        avg["iou"] = torch.cat([avg["iou"], metrics_dict["iou"]])
        avg["dice"] = torch.cat([avg["dice"], metrics_dict["dice"]])
        avg["fb_iou"] = torch.cat([avg["fb_iou"], metrics_dict["fb_iou"]])
        avg["th"] = None
        
    # print(avg["iou"])
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