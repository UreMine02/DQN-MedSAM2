""" function for training and validation in one epoch
    Yunli Qi
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tabulate import tabulate
import numpy as np

import cfg
from conf import settings
from func_3d.utils import eval_seg, iou_score, CombinedLoss, update_loss, average_loss, update_score, average_score, extract_object, sample_diverse_support, calculate_bounding_box, extract_object_multiple

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
    if args.distributed:
        net = net.module
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)
    
    net.train()
        
    video_length = args.video_length
    dice_loss_per_class = {}

    lossfunc = paper_loss

    # Record total loss thoroughout the entire epoch
    total_loss = {"total_loss": 0, "focal_loss": 0, "dice_loss": 0, "mae_loss": 0, "bce_loss": 0, "num_step": 0}
    target_class = 4
    agent_loss = {"actor_loss": 0, "critic_loss": 0}
    agent_step = 0
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img', position=0) as pbar:
        for packs in train_loader:
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
                        iou_gt = iou_score(pred, mask)
                        dice_loss, focal_loss, mae_loss, bce_loss = lossfunc(pred, mask, iou_pred, iou_gt.reshape(1), obj_pred)
                        class_loss["num_step"] += 1
                        # Update the loss of the class
                        update_loss(class_loss, focal_loss, dice_loss, mae_loss, bce_loss)

                        dice_loss_per_class[obj_id]["dice_loss"] += dice_loss.item()
                        dice_loss_per_class[obj_id]["num_step"] += 1
                    
                    # Average loss of this class
                    average_loss(class_loss)
                    avg_loss = class_loss["total_loss"]

                    optimizer.zero_grad()
                    avg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
                    optimizer.step()
                    
                    agent = getattr(net, "agent", None)
                    if agent is not None:
                        q_updates_per_step = getattr(args, "q_updates_per_step", 0)
                        agent_step_loss = agent.update(q_updates_per_step)
                        if agent_step_loss is not None:
                            agent_loss["actor_loss"] += agent_step_loss["actor_loss"]
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

    average_loss(total_loss)
    dice_loss_per_class = {f"{class_}":dice_loss_output["dice_loss"]/dice_loss_output["num_step"] for class_, dice_loss_output in dice_loss_per_class.items()}

    avg_agent_loss = {}
    if agent_step > 0:
        avg_agent_loss["actor_loss"] = agent_loss["actor_loss"] / agent_step
    else:
        avg_agent_loss["actor_loss"] = 0
    
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
    
    metrics = {
        "iou": [],
        "dice": [],
        "fb_iou": [],
    }
    total_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
    score_per_class = {}

    # lossfunc = paper_loss

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for packs in val_loader:
            whole_imgs_tensor = packs["image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_masks_tensor = packs["label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            task = packs["task"][0]
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
                    score_per_class[f"{task}_{obj_id}"] = copy.deepcopy(metrics)
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
                    
                        for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state):
                            video_segments[out_frame_idx] = {
                                out_obj_id: {"image_tensor": imgs_tensor[out_frame_idx], "image_label" : masks_tensor[out_frame_idx],
                                "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i]}
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }
                            
                # Record the loss in this step
                class_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
                for frame_idx in video_segments.keys():
                    pred = video_segments[frame_idx][obj_id]["pred_mask"].squeeze(0)
                    mask = video_segments[frame_idx][obj_id]["image_label"]
                    if mask is not None:
                        mask = mask.to(dtype=torch.float32, device=GPUdevice)
                        (
                            iou,
                            dice,
                            fb_iou,
                        ) = eval_seg(pred, mask)
                        update_score(class_score, dice.item(), iou.item())
                        class_score["num_step"] += 1

                        score_per_class[f"{task}_{obj_id}"]["iou"].append(iou.item())
                        score_per_class[f"{task}_{obj_id}"]["dice"].append(dice.item())
                        score_per_class[f"{task}_{obj_id}"]["fb_iou"].append(fb_iou.item())

                    else:
                        pred_mask = torch.where(torch.sigmoid(pred) >= 0.5, 1, 0)
                        mask = torch.zeros_like(pred).to(device=GPUdevice)
                    
                average_score(class_score)
                update_score(instance_score, class_score["dice_score"], class_score["iou_score"])

                instance_score["num_step"] += 1

            average_score(instance_score)
            # print(f"Name: {task}_{obj_id} Dice score: {instance_score['dice_score']} IoU score: {instance_score['iou_score']}")
            update_score(total_score, instance_score["dice_score"], instance_score["iou_score"])
            total_score["num_step"] += 1
            pbar.update()
        
    average_score(total_score)

    score_per_class["Avg"] = copy.deepcopy(metrics)
    for name, metrics_dict in score_per_class.items():
        if name == "Avg":
            continue
        mean_metrics = {k: np.mean(v) for k, v in metrics_dict.items()}
        score_per_class[name] = mean_metrics
        for metric in score_per_class["Avg"].keys():
            score_per_class["Avg"][metric].append(score_per_class[name][metric])
            
    score_per_class["Avg"] = {k: np.mean(v) for k, v in score_per_class["Avg"].items()}

    for name, metrics_dict in score_per_class.items():
        print(f"{name}:")
        print(tabulate([metrics_dict.values()], headers=metrics_dict.keys(), floatfmt=".4f"))

    return torch.Tensor([score_per_class["Avg"]['iou']]).to(GPUdevice), torch.Tensor([score_per_class["Avg"]['dice']]).to(GPUdevice)