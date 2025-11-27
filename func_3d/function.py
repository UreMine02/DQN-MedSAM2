""" function for training and validation in one epoch
    Yunli Qi
"""

import os
import matplotlib.pyplot as plt
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchshow as ts
from einops import rearrange
from tqdm import tqdm
import numpy as np

import cfg
from conf import settings
from func_3d.utils import eval_seg, iou_score, CombinedLoss, update_loss, average_loss, update_score, average_score, extract_object, sample_diverse_support, calculate_bounding_box, extract_object_multiple
import wandb
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import tracemalloc

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

def train_sam(args, net: nn.Module, optimizer, train_loader, epoch, rank=None):
    tracemalloc.start()
    if args.distributed:
        net = net.module
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)
    
    
    agent = getattr(net, "agent", None)
    net.train()
    if agent is not None:
        agent.set_epoch(epoch, distributed=args.distributed)
    for name, param in net.named_parameters():
        if "image_encoder" in name:
            param.requires_grad_(False)
        if "sam_prompt_encoder" in name:
            param.requires_grad_(False)
        if "sam_mask_decoder" in name:
            param.requires_grad_(False)
        
    video_length = args.video_length
    dice_loss_per_class = {}

    lossfunc = paper_loss

    # Record total loss thoroughout the entire epoch
    total_loss = {"total_loss": 0, "focal_loss": 0, "dice_loss": 0, "mae_loss": 0, "bce_loss": 0, "num_step": 0}
    target_class = 4
    agent_loss = 0
    agent_step = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for packs in train_loader:            
            whole_imgs_tensor = packs["image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_masks_tensor = packs["label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            name = packs["name"][0]
            
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
                    dice_loss_per_class[name] = {"dice_loss":0, "num_step": 0} 
                imgs_tensor = pack['image']
                # print('imgs_tensor',imgs_tensor.shape)
                masks_tensor = pack['label']

                support_imgs_tensor = pack["support_image"]
                support_masks_tensor = pack["support_label"]
                if imgs_tensor.numel() == 0 or masks_tensor.numel() == 0:
                    print(f"[Query] Warning: Empty image or mask tensor for obj_id={obj_id} in {name}. Skipping...")
                    continue  # Skip empty tensors
                if support_imgs_tensor.numel() == 0 or support_masks_tensor.numel() == 0:
                    print(f"[Support] Warning: Empty support image or mask tensor for obj_id={obj_id} in {name}. Skipping...")
                    continue
                
                train_state = net.train_init_state(
                    args=args,
                    imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor
                )
                
                if args.wandb_enabled:
                    support_pair = []
                    for frame_idx in range(support_imgs_tensor.shape[0]):
                        support_image = support_imgs_tensor[frame_idx].permute(1, 2, 0).detach().cpu().numpy()
                        # print(support_imgs_tensor[frame_idx].shape)
                        # support_image = support_imgs_tensor[frame_idx].unsqueeze(-1).expand(1024,1024,3).detach().cpu().numpy()
                        # support_image = support_imgs_tensor[frame_idx].detach().cpu().numpy()
                        support_label = support_masks_tensor[frame_idx].detach().cpu().numpy()
                        support_pair.append(wandb.Image(support_image, masks={"ground_truth": {"mask_data": support_label}}, caption=f"support {frame_idx}"))

                # if args.wandb_enabled:
                #     wandb.log({"train/support set": support_pair})
                
                with torch.cuda.amp.autocast():
                    for frame_idx in range(support_masks_tensor.shape[0]):
                        mask = support_masks_tensor[frame_idx]
                        _, _, _ = net.train_add_new_mask(
                            inference_state=train_state,
                            frame_idx=frame_idx,
                            obj_id=obj_id,
                            mask=mask.to(device=GPUdevice),
                        )
                        #     )

                    video_segments = {}  # video_segments contains the per-frame segmentation results

                    for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state, train_agent=True):
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
                        _, iou_gt, _ = eval_seg(pred, mask)
                        dice_loss, focal_loss, mae_loss, bce_loss = lossfunc(pred, mask, iou_pred, iou_gt.reshape(1), obj_pred)
                        class_loss["num_step"] += 1
                        # Update the loss of the class
                        update_loss(class_loss, focal_loss, dice_loss, mae_loss, bce_loss)

                        dice_loss_per_class[name]["dice_loss"] += dice_loss.item()
                        dice_loss_per_class[name]["num_step"] += 1
                    
                    # Average loss of this class
                    average_loss(class_loss)
                    avg_loss = class_loss["total_loss"]

                    optimizer.zero_grad()
                    avg_loss.backward()
                    optimizer.step()

                    # try:
                        # lấy agent từ model (nếu có attach)
                    if args.distributed:
                        torch.distributed.barrier()
                        
                    if agent is not None:
                        q_updates_per_step = getattr(args, "q_updates_per_step", 1)
                        print(f"Updating agents {q_updates_per_step} times")
                        agent_step_loss = agent.update(q_updates_per_step)
                        if agent_step_loss is not None:
                            agent_loss += agent_step_loss
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
            pbar.update()
            
            # process = psutil.Process(os.getpid())
            # memory_info = process.memory_info()
            # print(f"Memory Usage (RSS): {memory_info.rss / (1024 * 1024):.2f} MB")

        
    average_loss(total_loss)
    dice_loss_per_class = {f"{class_}":dice_loss_output["dice_loss"]/dice_loss_output["num_step"] for class_, dice_loss_output in dice_loss_per_class.items()}
    
    if args.wandb_enabled:
        for class_, dice_loss in dice_loss_per_class.items():
            wandb.log({f"train/Class {class_} loss": dice_loss}, step=epoch)

    return total_loss["total_loss"], total_loss["dice_loss"], total_loss["focal_loss"], total_loss["mae_loss"], total_loss["bce_loss"], agent_loss / agent_step

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True, rank=None):
    if args.distributed:
        net = net.module
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)

    # eval mode
    net.eval()
    n_val = len(val_loader)

    total_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
    dice_score_per_class = {}

    lossfunc = paper_loss

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for packs in val_loader:
            torch.cuda.empty_cache()
            whole_imgs_tensor = packs["image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_masks_tensor = packs["label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            name = packs["name"][0]
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
                if name not in dice_score_per_class.keys():
                    dice_score_per_class[name] = {"dice_score":0, "num_step": 0} 
                imgs_tensor = pack['image']
                masks_tensor = pack['label']

                # selected_support_frames = torch.randint(0, len(masks_tensor), size=(10,)).tolist()
                # support_imgs_tensor = pack["image"][selected_support_frames]
                # support_masks_tensor = pack["label"][selected_support_frames]

                support_imgs_tensor = pack["support_image"]
                support_masks_tensor = pack["support_label"]
                # support_bbox_dict = pack["support_bbox"]
                if imgs_tensor.numel() == 0 or masks_tensor.numel() == 0:
                    print(f"VALIDATION: [Query] Warning: Empty image or mask tensor for obj_id={obj_id} in {name}. Skipping...")
                    continue  # Skip empty tensors

                if support_imgs_tensor.numel() == 0 or support_masks_tensor.numel() == 0:
                    print(f"VALIDATION: [Support] Warning: Empty support image or mask tensor for obj_id={obj_id} in {name}. Skipping...")
                    continue
        
                train_state = net.val_init_state(
                    args=args,
                    imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor
                )
                # print('train_st:', train_state)
                support_pair = []
                filtered_support_pair = []
                for frame_idx in range(support_imgs_tensor.shape[0]):
                    support_image = support_imgs_tensor[frame_idx].permute(1, 2, 0).detach().cpu().numpy()
                    # support_image = support_imgs_tensor[frame_idx].detach().cpu().numpy()
                    support_label = support_masks_tensor[frame_idx].detach().cpu().numpy()
                    support_pair.append(wandb.Image(support_image, masks={"ground_truth": {"mask_data": support_label}}, caption=f"support {frame_idx}"))
                     # Add to the filtered support pair if the label contains the current obj_id
                    if (support_label == obj_id).any():
                        filtered_support_pair.append(
                            wandb.Image(
                                support_image,
                                masks={"ground_truth": {"mask_data": support_label}},
                                caption=f"Support {frame_idx} (class {obj_id})"
                            )
                        )
                
                if args.wandb_enabled:
                    wandb.log({"test/support set": support_pair}, step=epoch)
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
                    
                        for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state):
                            video_segments[out_frame_idx] = {
                                out_obj_id: {"image_tensor": imgs_tensor[out_frame_idx], "image_label" : masks_tensor[out_frame_idx],
                                "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i]}
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }
                            
                # Record the loss in this step
                wandb_result = []
                class_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
                for frame_idx in video_segments.keys():
                    pred = video_segments[frame_idx][obj_id]["pred_mask"].squeeze(0)
                    mask = video_segments[frame_idx][obj_id]["image_label"]
                    if mask is not None:
                        mask = mask.to(dtype=torch.float32, device=GPUdevice)
                        dice_score, iou_score, pred_mask = eval_seg(pred, mask)
                        update_score(class_score, dice_score, iou_score)
                        class_score["num_step"] += 1

                        dice_score_per_class[name]["dice_score"] += dice_score
                        dice_score_per_class[name]["num_step"] += 1

                    else:
                        pred_mask = torch.where(torch.sigmoid(pred)>=0.5, 1, 0)
                        mask = torch.zeros_like(pred).to(device=GPUdevice)
                    
                    if args.vis:
                        save_dir = "/".join(args.pretrain.split("/")[:-1])
                        save_prefix = f"{save_dir}/vis/{packs['case']}_{name}_idx{frame_idx}_"
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
                    
                average_score(class_score)
                update_score(instance_score, class_score["dice_score"], class_score["iou_score"])

                instance_score["num_step"] += 1

            average_score(instance_score)
            print(f"Name: {name} Dice score: {instance_score['dice_score']} IoU score: {instance_score['iou_score']}")
            update_score(total_score, instance_score["dice_score"], instance_score["iou_score"])
            total_score["num_step"] += 1
            pbar.update()
        
    average_score(total_score)

    dice_score_per_class = {f"{class_}":dice_score_output["dice_score"]/dice_score_output["num_step"] for class_, dice_score_output in dice_score_per_class.items()}

    if args.wandb_enabled:
        for class_, dice_score in dice_score_per_class.items():
            wandb.log({f"test/Class {class_}": dice_score}, step=epoch)

    dice_score_string = ""
    for class_, dice_score in dice_score_per_class.items():
        dice_score_string += f"Class: {class_} Dice Score: {dice_score}\n" 

    print(dice_score_string)

    return total_score["iou_score"], total_score["dice_score"]