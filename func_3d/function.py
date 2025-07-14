""" function for training and validation in one epoch
    Yunli Qi
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
import numpy as np

import cfg
from conf import settings
from func_3d.utils import eval_seg, iou_score, CombinedLoss, update_loss, average_loss, update_score, average_score, extract_object, sample_diverse_support, calculate_bounding_box, extract_object_multiple
import wandb
import matplotlib.pyplot as plt
from torchvision.utils import save_image

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

    # train mode
    net.train()
    video_length = args.video_length
    dice_loss_per_class = {}

    lossfunc = paper_loss

    # Record total loss thoroughout the entire epoch
    total_loss = {"total_loss": 0, "focal_loss": 0, "dice_loss": 0, "mae_loss": 0, "bce_loss": 0, "num_step": 0}
    target_class = 4 

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for packs in train_loader:
            whole_imgs_tensor = packs["image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_masks_tensor = packs["label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            name = packs["name"][0]
            # Log overall slice stats for the pack
            # print(f"[PACK STATS] Name: {name}")
            # print(f"  Query Total Slices: {whole_masks_tensor.shape[0]}, Classes: {torch.unique(whole_masks_tensor)}")
            # print(f"  Support Total Slices: {whole_support_masks_tensor.shape[0]}, Classes: {torch.unique(whole_support_masks_tensor)}")

            obj_list = torch.unique(whole_masks_tensor)[1:].int().tolist()
            instance_loss = {"total_loss": 0, "focal_loss": 0, "dice_loss": 0, "mae_loss": 0, "bce_loss": 0, "num_step": 0} 
            for obj_id in obj_list:
                pack = extract_object(whole_imgs_tensor, whole_masks_tensor, whole_support_imgs_tensor, whole_support_masks_tensor, \
                                      obj_id=obj_id, video_length=args.video_length, num_support=args.num_support)
                if pack is None:
                    print(f"[PACK SKIP] obj_id={obj_id}\n")
                    # print(f"Query Tensor: Slices={whole_imgs_tensor.shape[0]}, Unique Classes={torch.unique(whole_masks_tensor)}")
                    # print(f"Support Tensor: Slices={whole_support_imgs_tensor.shape[0]}, Unique Classes={torch.unique(whole_support_masks_tensor)}")
                    continue
                torch.cuda.empty_cache()
                if obj_id not in dice_loss_per_class.keys():
                    dice_loss_per_class[obj_id] = {"dice_loss":0, "num_step": 0} 
                imgs_tensor = pack['image']
                masks_tensor = pack['label']

                support_imgs_tensor = pack["support_image"]
                support_masks_tensor = pack["support_label"]
                if imgs_tensor.numel() == 0 or masks_tensor.numel() == 0:
                    print(f"[Query] Warning: Empty image or mask tensor for obj_id={obj_id} in {name}. Skipping...")
                    continue  # Skip empty tensors
                if support_imgs_tensor.numel() == 0 or support_masks_tensor.numel() == 0:
                    print(f"[Support] Warning: Empty support image or mask tensor for obj_id={obj_id} in {name}. Skipping...")
                    continue
                # support_bbox_dict = pack["support_bbox"]
                # Sample diverse support images
                # support_imgs_tensor, support_masks_tensor = sample_diverse_support(
                #     support_imgs_tensor, support_masks_tensor, num_samples=args.num_support
                # )
                train_state = net.train_init_state(imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor)
                
                support_pair = []
                for frame_idx in range(support_imgs_tensor.shape[0]):
                    support_image = support_imgs_tensor[frame_idx].permute(1, 2, 0).detach().cpu().numpy()
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

                            # bbox = support_bbox_dict[frame_idx][ann_obj_id]
                            # _, _, _ = net.train_add_new_bbox(
                            #     inference_state=train_state,
                            #     frame_idx=frame_idx,
                            #     obj_id=ann_obj_id,
                            #     bbox=bbox.to(device=GPUdevice),
                            #     clear_old_points=False,
                            # )
                        # else:
                        #     _, _, _ = net.train_add_new_mask(
                        #         inference_state=train_state,
                        #         frame_idx=frame_idx,
                        #         obj_id=obj_id,
                        #         mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                        #     )

                    video_segments = {}  # video_segments contains the per-frame segmentation results
                
                    for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state):
                        video_segments[out_frame_idx] = {
                            out_obj_id: {"image_tensor": imgs_tensor[out_frame_idx], "image_label" : masks_tensor[out_frame_idx],
                            "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i]}
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }

                    # Record the loss in this step
                    class_loss = {"total_loss":0, "focal_loss": 0, "dice_loss": 0, "mae_loss": 0, "bce_loss": 0, "num_step": 0} 
                    wandb_result = []
                    for frame_idx in video_segments.keys():
                        if args.wandb_enabled:
                            whole_gt_mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice)
                            whole_pred_mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice)
                            output = video_segments[frame_idx]
                            original_img_wandb = imgs_tensor[frame_idx, 0, :, :].detach().cpu().numpy()

                            pred_mask = torch.where(torch.sigmoid(output[obj_id]["pred_mask"][0, :, :])>=0.5, obj_id, 0)
                            whole_pred_mask += pred_mask

                            gt_mask = video_segments[frame_idx][obj_id]["image_label"]
                            if gt_mask is not None:
                                gt_mask = torch.where(gt_mask==1, obj_id, 0).to(device=GPUdevice)
                                whole_gt_mask += gt_mask

                            whole_pred_mask = whole_pred_mask.detach().cpu().numpy()
                            whole_gt_mask = whole_gt_mask.detach().cpu().numpy()
                            # Normalize original image
                            original_img_wandb = (original_img_wandb - original_img_wandb.min()) / (original_img_wandb.max() - original_img_wandb.min())
                            original_img_wandb = (original_img_wandb * 255).astype(np.uint8)  # Scale to 0–255
                            # Overlap visualization
                            overlap = np.stack([original_img_wandb] * 3, axis=-1)  # Shape: (H, W, 3)
                            overlap[whole_gt_mask == obj_id] = [0, 255, 0]  # Green for ground truth
                            overlap[whole_pred_mask == obj_id] = [255, 0, 0]  # Red for prediction
                            overlap[(whole_gt_mask == obj_id) & (whole_pred_mask == obj_id)] = [255, 255, 0]  # Yellow for overlap

                            # Append the WandB result
                            wandb_result += [[
                                wandb.Image(original_img_wandb, caption="Image"),
                                wandb.Image(original_img_wandb, masks={"ground_truth": {"mask_data": whole_gt_mask}}, caption="Label"),
                                wandb.Image(original_img_wandb, masks={"predictions": {"mask_data": whole_pred_mask}}, caption="Prediction"),
                                wandb.Image(overlap, caption="Overlap")
                            ] + support_pair]
                            # wandb_result += [[wandb.Image(original_img_wandb, caption="image"),
                            #                 wandb.Image(original_img_wandb, masks={"ground_truth": {"mask_data": whole_gt_mask}}, caption="label"),
                            #                 wandb.Image(original_img_wandb, masks={"predictions": {"mask_data": whole_pred_mask}}, caption="prediction")]
                            #                 ]

                        pred = video_segments[frame_idx][obj_id]["pred_mask"].squeeze(0)
                        mask = video_segments[frame_idx][obj_id]["image_label"]
                        if mask is not None:
                            mask = mask.to(dtype=torch.float32, device=GPUdevice)
                        else:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        # Calculate the loss
                        obj_pred = video_segments[frame_idx][obj_id]["object_score_logits"]
                        iou_pred = video_segments[frame_idx][obj_id]["iou"]
                        _, iou_gt = eval_seg(pred, mask)
                        dice_loss, focal_loss, mae_loss, bce_loss= lossfunc(pred, mask, iou_pred, iou_gt.reshape(1), obj_pred)
                        class_loss["num_step"] += 1
                        # Update the loss of the class
                        update_loss(class_loss, focal_loss, dice_loss, mae_loss, bce_loss)

                        dice_loss_per_class[obj_id]["dice_loss"] += dice_loss
                        dice_loss_per_class[obj_id]["num_step"] += 1

                    if args.wandb_enabled:
                        if len(wandb_result) > 5:
                            sampled_index = np.random.choice(range(len(wandb_result)), size=5)
                        else:
                            sampled_index = range(len(wandb_result))
                        wandb_result = [wandb_inputs for frame_idx, wandb_inputs in enumerate(wandb_result) if frame_idx in sampled_index]
                        wandb_result = [wandb_input for wandb_inputs in wandb_result for wandb_input in wandb_inputs]
                        # wandb.log({f"train image/ {int(obj_id)}": wandb_result})
                        # wandb.log({f"train image": wandb_result})

                    # Average loss of this class
                    average_loss(class_loss)
                    avg_loss = class_loss["total_loss"]

                    optimizer.zero_grad()
                    avg_loss.backward()
                    optimizer.step()

                    # Add the loss of the class to the instance
                    update_loss(instance_loss, class_loss["focal_loss"], class_loss["dice_loss"], class_loss["mae_loss"], class_loss["bce_loss"])
                    instance_loss["num_step"] += 1
            
            average_loss(instance_loss)
            # optimizer.zero_grad()
            # avg_loss.backward()
            # optimizer.step()

            update_loss(total_loss, instance_loss["focal_loss"], instance_loss["dice_loss"], instance_loss["mae_loss"], instance_loss["bce_loss"])
            total_loss["num_step"] += 1

            pbar.update()
        
    average_loss(total_loss)
    dice_loss_per_class = {f"{class_}":dice_loss_output["dice_loss"]/dice_loss_output["num_step"] for class_, dice_loss_output in dice_loss_per_class.items()}

    if args.wandb_enabled:
        for class_, dice_loss in dice_loss_per_class.items():
            wandb.log({f"train/Class {class_}": dice_loss})

    return total_loss["total_loss"], total_loss["dice_loss"], total_loss["focal_loss"], total_loss["mae_loss"], total_loss["bce_loss"]

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
                if obj_id not in dice_score_per_class.keys():
                    dice_score_per_class[obj_id] = {"dice_score":0, "num_step": 0} 
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

                train_state = net.val_init_state(imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor)
                
                support_pair = []
                filtered_support_pair = []
                for frame_idx in range(support_imgs_tensor.shape[0]):
                    support_image = support_imgs_tensor[frame_idx].permute(1, 2, 0).detach().cpu().numpy()
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
                    wandb.log({"test/support set": support_pair})
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

                                # bbox = support_bbox_dict[frame_idx][ann_obj_id]
                                # _, _, _ = net.train_add_new_bbox(
                                #     inference_state=train_state,
                                #     frame_idx=frame_idx,
                                #     obj_id=ann_obj_id,
                                #     bbox=bbox.to(device=GPUdevice),
                                #     clear_old_points=False,
                                # )

                            # else:
                            #     _, _, _ = net.train_add_new_mask(
                            #         inference_state=train_state,
                            #         frame_idx=frame_idx,
                            #         obj_id=ann_obj_id,
                            #         mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            #     )

                        video_segments = {}  # video_segments contains the per-frame segmentation results
                    
                        for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state):
                            video_segments[out_frame_idx] = {
                                out_obj_id: {"image_tensor": imgs_tensor[out_frame_idx], "image_label" : masks_tensor[out_frame_idx],
                                "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i]}
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }
                
                
                # frame_predicted_iou = [(frame_idx, output["iou"]) for frame_idx, object_output in video_segments.items() for obj_id, output in object_output.items()]
                # if imgs_tensor.shape[0] < support_imgs_tensor.shape[0]:
                #     new_support_num_frame = imgs_tensor.shape[0]
                # else:
                #     new_support_num_frame = support_imgs_tensor.shape[0]
                # top10_frame = [frame_idx for frame_idx, iou in sorted(frame_predicted_iou, key=lambda x: x[1], reverse=True)[:new_support_num_frame]]
                # new_support_imgs_tensor = imgs_tensor[top10_frame]
                # new_support_masks_tensor = torch.zeros(new_support_num_frame, 1024, 1024).to(device=GPUdevice)
                # for frame_idx, object_id_output in video_segments.items():
                #     if frame_idx in top10_frame:
                #         for obj_id, output in object_id_output.items():
                #             mask = torch.sigmoid(output["pred_mask"]) >= 0.5
                #             new_support_masks_index = top10_frame.index(frame_idx)
                #             new_support_masks_tensor[new_support_masks_index] = mask
                
                # train_state = net.val_init_state(imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=new_support_imgs_tensor)

                # with torch.no_grad():
                #     with torch.cuda.amp.autocast():
                #         for frame_idx in range(new_support_masks_tensor.shape[0]):
                #             mask = new_support_masks_tensor[frame_idx]
                #             _, _, _ = net.train_add_new_mask(
                #                 inference_state=train_state,
                #                 frame_idx=frame_idx,
                #                 obj_id=obj_id,
                #                 mask=mask.to(device=GPUdevice),
                #             )

                #             #     bbox = support_bbox_dict[frame_idx][ann_obj_id]
                #             #     _, _, _ = net.train_add_new_bbox(
                #             #         inference_state=train_state,
                #             #         frame_idx=frame_idx,
                #             #         obj_id=ann_obj_id,
                #             #         bbox=bbox.to(device=GPUdevice),
                #             #         clear_old_points=False,
                #             #     )

                #             # else:
                #             #     _, _, _ = net.train_add_new_mask(
                #             #         inference_state=train_state,
                #             #         frame_idx=frame_idx,
                #             #         obj_id=ann_obj_id,
                #             #         mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                #             #     )

                #         video_segments = {}  # video_segments contains the per-frame segmentation results
                    
                #         for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.propagate_in_video(train_state):
                #             video_segments[out_frame_idx] = {
                #                 out_obj_id: {"image_tensor": imgs_tensor[out_frame_idx], "image_label" : masks_tensor[out_frame_idx],
                #                 "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i]}
                #                 for i, out_obj_id in enumerate(out_obj_ids)
                #             }


                # Record the loss in this step
                wandb_result = []
                class_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
                for frame_idx in video_segments.keys():
                    if args.wandb_enabled:
                        whole_gt_mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice)
                        whole_pred_mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice)
                        output = video_segments[frame_idx]
                        original_img_wandb = imgs_tensor[frame_idx, 0, :, :].detach().cpu().numpy()

                        pred_mask = torch.where(torch.sigmoid(output[obj_id]["pred_mask"][0, :, :])>=0.5, obj_id, 0)
                        whole_pred_mask += pred_mask

                        gt_mask = output[obj_id]["image_label"]
                        if gt_mask is not None:
                            gt_mask = torch.where(gt_mask==1, obj_id, 0).to(device=GPUdevice)
                            whole_gt_mask += gt_mask

                        whole_pred_mask = whole_pred_mask.detach().cpu().numpy()
                        whole_gt_mask = whole_gt_mask.detach().cpu().numpy()
                        # Normalize original image
                        original_img_wandb = (original_img_wandb - original_img_wandb.min()) / (original_img_wandb.max() - original_img_wandb.min())
                        original_img_wandb = (original_img_wandb * 255).astype(np.uint8)  # Scale to 0–255
                        # Overlap visualization
                        overlap = np.stack([original_img_wandb] * 3, axis=-1)  # Shape: (H, W, 3)
                        overlap[whole_gt_mask == obj_id] = [0, 255, 0]  # Green for ground truth
                        overlap[whole_pred_mask == obj_id] = [255, 0, 0]  # Red for prediction
                        overlap[(whole_gt_mask == obj_id) & (whole_pred_mask == obj_id)] = [255, 255, 0]  # Yellow for overlap

                        wandb_result += [[
                            wandb.Image(original_img_wandb, caption="Image"),
                            wandb.Image(original_img_wandb, masks={"ground_truth": {"mask_data": whole_gt_mask}}, caption="Label"),
                            wandb.Image(original_img_wandb, masks={"predictions": {"mask_data": whole_pred_mask}}, caption="Prediction"),
                            wandb.Image(overlap, caption="Overlap")
                        ] + support_pair]  # Append support_pair directly to the WandB result
                        
                        
                    pred = video_segments[frame_idx][obj_id]["pred_mask"].squeeze(0)
                    mask = video_segments[frame_idx][obj_id]["image_label"]
                    if mask is not None:
                        mask = mask.to(dtype=torch.float32, device=GPUdevice)
                        dice_score, iou_score = eval_seg(pred, mask)
                        update_score(class_score, dice_score, iou_score)
                        class_score["num_step"] += 1

                        dice_score_per_class[obj_id]["dice_score"] += dice_score
                        dice_score_per_class[obj_id]["num_step"] += 1

                    else:
                        mask = torch.zeros_like(pred).to(device=GPUdevice)
                
                if args.wandb_enabled:
                    if len(wandb_result) > 5:
                        sampled_index = np.random.choice(range(len(wandb_result)), size=5)
                    else:
                        sampled_index = range(len(wandb_result))
                    wandb_result = [wandb_inputs for frame_idx, wandb_inputs in enumerate(wandb_result) if frame_idx in sampled_index]
                    wandb_result = [wandb_input for wandb_inputs in wandb_result for wandb_input in wandb_inputs]
                    wandb.log({f"test image/ {int(obj_id)}": wandb_result})

                average_score(class_score)
                update_score(instance_score, class_score["dice_score"], class_score["iou_score"])
                instance_score["num_step"] += 1

            average_score(instance_score)
            print(f"Name: {name} Dice score: {instance_score["dice_score"]} IoU score: {instance_score["iou_score"]}")
            update_score(total_score, instance_score["dice_score"], instance_score["iou_score"])
            total_score["num_step"] += 1
            pbar.update()
        
    average_score(total_score)

    dice_score_per_class = {f"{class_}":dice_score_output["dice_score"]/dice_score_output["num_step"] for class_, dice_score_output in dice_score_per_class.items()}

    if args.wandb_enabled:
        for class_, dice_score in dice_score_per_class.items():
            wandb.log({f"test/Class {class_}": dice_score})

    dice_score_string = ""
    for class_, dice_score in dice_score_per_class.items():
        dice_score_string += f"Class: {class_} Dice Score: {dice_score}\n" 

    print(dice_score_string)

    return total_score["iou_score"], total_score["dice_score"]