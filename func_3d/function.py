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
import pandas as pd

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
paper_loss = CombinedLoss()
seed = torch.randint(1,11,(1,7))

torch.backends.cudnn.benchmark = True
# scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def min_max_scaling(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

def check(a, s=6):
    min_inbank = a[-s:].min()
    max_outbank = a[:-s].max()
    return min_inbank < max_outbank

def train_sam(args, net: nn.Module, optimizer, train_loader, epoch, rank=None):
    if args.distributed:
        net = net.module
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)

    net.train()

    video_length = args.video_length
    train_agent = not args.no_agent
    agent_act = not args.no_agent
    dice_loss_per_class = {}

    lossfunc = paper_loss

    # Record total loss thoroughout the entire epoch
    total_loss = {"total_loss": 0, "focal_loss": 0, "dice_loss": 0, "mae_loss": 0, "bce_loss": 0, "num_step": 0}
    target_class = 4
    agent_loss = {"actor_loss": 0, "critic_loss": 0}
    agent_step = 0
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    # with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img', position=0) as pbar:
    for packs in metric_logger.log_every(train_loader, print_freq, header=header) :
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

                for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state, train_agent=train_agent, agent_act=agent_act):
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
                # avg_loss = class_loss["dice_loss"] + class_loss["focal_loss"]

                to_reduce = {k: class_loss[k] for k in class_loss.keys() if k != "num_step"}
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
                if agent is not None:
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
    agent_act = not args.no_agent
    # lossfunc = paper_loss

    dropped_frames_allres_sim_rank = []
    dropped_frames_lowres_sim_rank = []
    dropped_frames_ious_rank = []
    dropped_frames_dice_rank = []

    total_global_allres_sim = {}
    total_global_lowres_sim = {}
    total_global_masked_allres_sim = {}
    total_global_masked_lowres_sim = {}
    total_local_allres_sim = {}
    total_local_lowres_sim = {}
    total_local_masked_allres_sim = {}
    total_local_masked_lowres_sim = {}
    total_iou_sim = {}
    vol_avg_dice = {}

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
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
                if pack is None:
                    print(f"[Validation] [PACK]: No valid for pack for obj_id={obj_id}. Skipping...")
                    # print(f"[DEBUG - QUERY] Slices: {whole_imgs_tensor.shape[0]}, Unique Classes: {torch.unique(whole_masks_tensor)}")
                    # print(f"[DEBUG - SUPPORT] Slices: {whole_support_imgs_tensor.shape[0]}, Unique Classes: {torch.unique(whole_support_masks_tensor)}")
                    continue
                if obj_id not in score_per_class.keys():
                    score_per_class[f"{task}_{obj_id}"] = {
                        "iou": torch.FloatTensor([]).to(device=GPUdevice),
                        "dice": torch.FloatTensor([]).to(device=GPUdevice),
                        "fb_iou": torch.FloatTensor([]).to(device=GPUdevice),
                    }

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

                # dropped_frames_allres_sim_rank.extend(train_state["output_dict"]["dropped_frames_allres_sim_rank"])
                # dropped_frames_lowres_sim_rank.extend(train_state["output_dict"]["dropped_frames_lowres_sim_rank"])
                # dropped_frames_ious_rank.extend(train_state["output_dict"]["dropped_frames_ious_rank"])
                # dropped_frames_dice_rank.extend(train_state["output_dict"]["dropped_frames_dice_rank"])

                # Record the loss in this step
                if args.ablation:
                    vol_avg_dice[f"{name}_{obj_id}"] = []
                    
                class_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
                for frame_idx in video_segments.keys():
                    pred = video_segments[frame_idx][obj_id]["pred_mask"].squeeze(0)
                    mask = video_segments[frame_idx][obj_id]["image_label"]
                    pred_mask = torch.where(torch.sigmoid(pred) >= 0.5, 1, 0)
                    if mask is not None:
                        mask = mask.to(dtype=torch.float32, device=GPUdevice)
                        (
                            iou,
                            dice,
                            fb_iou,
                        ) = eval_seg(pred, mask)
                        update_score(class_score, dice.item(), iou.item())
                        class_score["num_step"] += 1

                        score_dict = score_per_class[f"{task}_{obj_id}"]

                        score_dict["iou"] = torch.cat([score_dict["iou"], iou.detach()])
                        score_dict["dice"] = torch.cat([score_dict["dice"], dice.detach()])
                        score_dict["fb_iou"] = torch.cat([score_dict["fb_iou"], fb_iou.detach()])
                        
                        # Record the loss in this step
                        if args.ablation:
                            vol_avg_dice[f"{name}_{obj_id}"].append(dice)
                    else:
                        mask = torch.zeros_like(pred).to(device=GPUdevice)

                    if args.vis:
                        save_dir = "/".join(args.pretrain.split("/")[:-1])
                        save_prefix = f"{save_dir}/vis/{name}_{obj_id}_idx{frame_idx}_"
                        ts.save(imgs_tensor[frame_idx], save_prefix + "image.png")
                        ts.overlay(
                            [save_prefix + "image.png", pred_mask], [1, 0.4],
                            save_as=save_prefix + "pred.png",
                            cmap="jet"
                        )
                        ts.overlay(
                            [save_prefix + "image.png", mask], [1, 0.4],
                            save_as=save_prefix + "mask.png",
                            cmap="jet"
                        )

                if args.ablation:
                    vol_avg_dice[f"{name}_{obj_id}"] = torch.mean(torch.tensor(vol_avg_dice[f"{name}_{obj_id}"]))

                average_score(class_score)
                update_score(instance_score, class_score["dice_score"], class_score["iou_score"])

                instance_score["num_step"] += 1

                # HYPOTHESIS TESTING
                if args.ablation:
                    total_global_allres_sim[f"{name}_{obj_id}"] = [0,0]
                    total_global_lowres_sim[f"{name}_{obj_id}"] = [0,0]
                    total_global_masked_allres_sim[f"{name}_{obj_id}"] = [0,0]
                    total_global_masked_lowres_sim[f"{name}_{obj_id}"] = [0,0]
                    total_local_allres_sim[f"{name}_{obj_id}"] = [0,0]
                    total_local_lowres_sim[f"{name}_{obj_id}"] = [0,0]
                    total_local_masked_allres_sim[f"{name}_{obj_id}"] = [0,0]
                    total_local_masked_lowres_sim[f"{name}_{obj_id}"] = [0,0]
                    total_iou_sim[f"{name}_{obj_id}"] = [0,0]
                    
                    for frame_idx in train_state["output_dict"]["image_features"].keys():
                        curr_gt = train_state["gt_masks"][frame_idx].float().to(GPUdevice, non_blocking=True)
                        curr_local_feats = train_state["output_dict"]["image_features"][frame_idx]
                        curr_local_masked_feats = train_state["output_dict"]["masked_image_features"][frame_idx]
                        curr_global_feats = [feat.mean(0) for feat in curr_local_feats]
                        curr_global_masked_feats = [feat.mean(0) for feat in curr_local_masked_feats]
                        
                        prev_idx_list = []
                        global_allres_sim_list = []
                        global_lowres_sim_list = []
                        global_masked_allres_sim_list = []
                        global_masked_lowres_sim_list = []
                        local_allres_sim_list = []
                        local_lowres_sim_list = []
                        local_masked_allres_sim_list = []
                        local_masked_lowres_sim_list = []
                        gt_iou_list = []
                        for prev_idx in train_state["output_dict"]["image_features"].keys():
                            if prev_idx >= frame_idx:
                                continue
                            
                            prev_local_feats = train_state["output_dict"]["image_features"][prev_idx]
                            prev_local_masked_feats = train_state["output_dict"]["masked_image_features"][prev_idx]
                            prev_global_feats = [feat.mean(0) for feat in prev_local_feats]
                            prev_global_masked_feats = [feat.mean(0) for feat in prev_local_masked_feats]
                            prev_gt = train_state["gt_masks"][prev_idx].float()
                            iou = iou_score(prev_gt, curr_gt)

                            sum_global_sim, sum_global_masked_sim, sum_local_sim, sum_local_masked_sim  = 0, 0, 0 ,0
                            for res in range(len(curr_global_feats)):
                                curr_local_feat = curr_local_feats[res]
                                curr_global_feat = curr_global_feats[res]
                                prev_local_feat = prev_local_feats[res]
                                prev_global_feat = prev_global_feats[res]
                                curr_local_masked_feat = curr_local_masked_feats[res]
                                curr_global_masked_feat = curr_global_masked_feats[res]
                                prev_local_masked_feat = prev_local_masked_feats[res]
                                prev_global_masked_feat = prev_global_masked_feats[res]
                                
                                curr_local_feat = F.normalize(curr_local_feat, p=2, dim=-1)
                                curr_global_feat = F.normalize(curr_global_feat, p=2, dim=-1)
                                prev_local_feat = F.normalize(prev_local_feat, p=2, dim=-1)
                                prev_global_feat = F.normalize(prev_global_feat, p=2, dim=-1)
                                curr_local_masked_feat = F.normalize(curr_local_masked_feat, p=2, dim=-1)
                                curr_global_masked_feat = F.normalize(curr_global_masked_feat, p=2, dim=-1)
                                prev_local_masked_feat = F.normalize(prev_local_masked_feat, p=2, dim=-1)
                                prev_global_masked_feat = F.normalize(prev_global_masked_feat, p=2, dim=-1)
                                
                                local_sim = curr_local_feat @ prev_local_feat.transpose(-2, -1)
                                local_masked_sim = curr_local_masked_feat @ prev_local_masked_feat.transpose(-2, -1)
                                global_sim = curr_global_feat @ prev_global_feat.transpose(-2, -1)
                                global_masked_sim = curr_global_masked_feat @ prev_global_masked_feat.transpose(-2, -1)
                                
                                local_sim = local_sim.mean()
                                local_masked_sim = local_masked_sim.mean()
                                
                                if res == len(curr_global_feats) - 1:
                                    local_lowres_sim_list.append(local_sim)
                                    local_masked_lowres_sim_list.append(local_masked_sim)
                                    global_lowres_sim_list.append(global_sim)
                                    global_masked_lowres_sim_list.append(global_masked_sim)
                                    
                                sum_global_sim += global_sim 
                                sum_global_masked_sim += global_masked_sim 
                                sum_local_sim += local_sim 
                                sum_local_masked_sim += local_masked_sim 

                            global_allres_sim_list.append(sum_global_sim)
                            global_masked_allres_sim_list.append(sum_global_masked_sim)
                            local_allres_sim_list.append(sum_local_sim)
                            local_masked_allres_sim_list.append(sum_local_masked_sim)
                            prev_idx_list.append(prev_idx)
                            gt_iou_list.append(iou)

                        
                        if len(global_allres_sim_list) > 6:
                            global_allres_sim_list = torch.Tensor(global_allres_sim_list)
                            global_lowres_sim_list = torch.Tensor(global_lowres_sim_list)
                            global_masked_allres_sim_list = torch.Tensor(global_masked_allres_sim_list)
                            global_masked_lowres_sim_list = torch.Tensor(global_masked_lowres_sim_list)
                            local_allres_sim_list = torch.Tensor(local_allres_sim_list)
                            local_lowres_sim_list = torch.Tensor(local_lowres_sim_list)
                            local_masked_allres_sim_list = torch.Tensor(local_masked_allres_sim_list)
                            local_masked_lowres_sim_list = torch.Tensor(local_masked_lowres_sim_list)
                            
                            global_allres_sim_list = min_max_scaling(global_allres_sim_list)
                            global_lowres_sim_list = min_max_scaling(global_lowres_sim_list)
                            global_masked_allres_sim_list = min_max_scaling(global_masked_allres_sim_list)
                            global_masked_lowres_sim_list = min_max_scaling(global_masked_lowres_sim_list)
                            local_allres_sim_list = min_max_scaling(local_allres_sim_list)
                            local_lowres_sim_list = min_max_scaling(local_lowres_sim_list)
                            local_masked_allres_sim_list = min_max_scaling(local_masked_allres_sim_list)
                            local_masked_lowres_sim_list = min_max_scaling(local_masked_lowres_sim_list)
                            
                            total_global_allres_sim[f"{name}_{obj_id}"][check(global_allres_sim_list)] += 1
                            total_global_lowres_sim[f"{name}_{obj_id}"][check(global_lowres_sim_list)] += 1
                            total_global_masked_allres_sim[f"{name}_{obj_id}"][check(global_masked_allres_sim_list)] += 1
                            total_global_masked_lowres_sim[f"{name}_{obj_id}"][check(global_masked_lowres_sim_list)] += 1
                            total_local_allres_sim[f"{name}_{obj_id}"][check(local_allres_sim_list)] += 1
                            total_local_lowres_sim[f"{name}_{obj_id}"][check(local_lowres_sim_list)] += 1
                            total_local_masked_allres_sim[f"{name}_{obj_id}"][check(local_masked_allres_sim_list)] += 1
                            total_local_masked_lowres_sim[f"{name}_{obj_id}"][check(local_masked_lowres_sim_list)] += 1
                            total_iou_sim[f"{name}_{obj_id}"][check(torch.tensor(gt_iou_list))] += 1
                        
                    # size = torch.tensor(size)
                    # size = size[torch.where(size > 0)]
                    # n_frame[f"{name}_{obj_id}"] = torch.std(size)
                    # print(name, torch.std(size))

            average_score(instance_score)
            # print(f"Name: {task}_{obj_id} Dice score: {instance_score['dice_score']} IoU score: {instance_score['iou_score']}")
            update_score(total_score, instance_score["dice_score"], instance_score["iou_score"])
            total_score["num_step"] += 1
            pbar.update()

    average_score(total_score)
    
    # HYPOTHESIS TESTING
    if args.ablation:
        data = []
        for obj_id in total_global_allres_sim.keys():
            print(obj_id, vol_avg_dice[obj_id])
            print("global_allres_sim", total_global_allres_sim[obj_id])
            print("global_lowres_sim", total_global_lowres_sim[obj_id])
            print("global_masked_allres_sim", total_global_masked_allres_sim[obj_id])
            print("global_masked_lowres_sim", total_global_masked_lowres_sim[obj_id])
            print("local_allres_sim", total_local_allres_sim[obj_id])
            print("local_lowres_sim", total_local_lowres_sim[obj_id])
            print("local_masked_allres_sim", total_local_masked_allres_sim[obj_id])
            print("local_masked_lowres_sim", total_local_masked_lowres_sim[obj_id])
            print("iou_sim", total_iou_sim[obj_id])
            data.append((
                vol_avg_dice[obj_id],
                total_global_allres_sim[obj_id],
                total_global_lowres_sim[obj_id],
                total_global_masked_allres_sim[obj_id],
                total_global_masked_lowres_sim[obj_id],
                total_local_allres_sim[obj_id],
                total_local_lowres_sim[obj_id],
                total_local_masked_allres_sim[obj_id],
                total_local_masked_lowres_sim[obj_id],
            ))
        columns = [
            "vol_avg_dice",
            "global_allres_sim",
            "global_lowres_sim",
            "global_masked_allres_sim",
            "global_masked_lowres_sim",
            "local_allres_sim",
            "local_lowres_sim",
            "local_masked_allres_sim",
            "local_masked_lowres_sim",
        ]
        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv("msd03_ablation.csv")
            

    avg = {
        "iou": torch.FloatTensor([]).to(device=GPUdevice),
        "dice": torch.FloatTensor([]).to(device=GPUdevice),
        "fb_iou": torch.FloatTensor([]).to(device=GPUdevice),
    }

    table_data = []

    for name, metrics_dict in score_per_class.items():
        miou = metrics_dict["iou"].mean(dim=0, keepdim=True)
        mdice = metrics_dict["dice"].mean(dim=0, keepdim=True)
        mfb_iou = metrics_dict["fb_iou"].mean(dim=0, keepdim=True)

        table_data.append((
            name,
            miou.item(),
            mdice.item(),
            mfb_iou.item(),
        ))

        avg["iou"] = torch.cat([avg["iou"], miou])
        avg["dice"] = torch.cat([avg["dice"], mdice])
        avg["fb_iou"] = torch.cat([avg["fb_iou"], mfb_iou])

    avg["iou"] = avg["iou"].mean()
    avg["dice"] = avg["dice"].mean()
    avg["fb_iou"] = avg["fb_iou"].mean()

    table_data.append((
        "Average",
        avg["iou"].item(),
        avg["dice"].item(),
        avg["fb_iou"].item(),
    ))

    print(tabulate(table_data, headers=["name", "iou", "dice", "fb_iou"], floatfmt=".4f", tablefmt="grid"))

    return avg['iou'], avg['dice']