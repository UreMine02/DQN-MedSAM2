"""
Q-learning utilities cho SAM2 memory management.
"""

import torch
from monai.losses import DiceLoss, FocalLoss
from sam2_train.q_learning_agent import RLStates

EPS = 1e-6

def prepare_rl_state(
    current_vision_feats,
    current_vision_pos_embeds,
    output_dict,
    frame_idx,
    num_maskmem,
    offload_to_cpu=True,
    training=False
):
    next_image_feat = current_vision_feats[-1] + current_vision_pos_embeds[-1]
    next_image_feat = next_image_feat.permute(1, 2, 0).reshape(1, 256, 64, 64)
    curr_memory_feat = output_dict["await_outputs"][frame_idx-1]
    curr_memory_feat = curr_memory_feat["maskmem_features"] + curr_memory_feat["maskmem_pos_enc"][0]
    curr_obj_ptr = output_dict["await_outputs"][frame_idx-1]["obj_ptr"]
    
    # Add non_cond memory
    cond_bank_list = list(output_dict["cond_frame_outputs"].values())
    non_cond_bank_list = list(output_dict["non_cond_frame_outputs"].values())
    memory_shape = cond_bank_list[0]["maskmem_features"].shape
    obj_ptr_shape = cond_bank_list[0]["obj_ptr"].shape
    device = "cpu" if offload_to_cpu else cond_bank_list[0]["maskmem_features"].device
    prev_memory_bank = []
    prev_obj_ptr = []
    for feat in non_cond_bank_list:
        mem_feat = feat["maskmem_features"] + feat["maskmem_pos_enc"][0]
        obj_ptr = feat["obj_ptr"]
        if offload_to_cpu:
            mem_feat = mem_feat.detach().cpu()
            obj_ptr = obj_ptr.detach().cpu()
        prev_memory_bank.append(mem_feat)
        prev_obj_ptr.append(obj_ptr)
    
    # Append zero memory
    while len(prev_memory_bank) < num_maskmem:
        prev_memory_bank.append(torch.zeros(memory_shape, device=device))
        prev_obj_ptr.append(torch.zeros(obj_ptr_shape, device=device))

    # Add cond memory
    for feat in cond_bank_list:
        mem_feat = feat["maskmem_features"] + feat["maskmem_pos_enc"][0]
        obj_ptr = feat["obj_ptr"]
        if offload_to_cpu:
            mem_feat = mem_feat.detach().cpu()
            obj_ptr = obj_ptr.detach().cpu()
        prev_memory_bank.append(mem_feat)
        prev_obj_ptr.append(obj_ptr)
        
    # Append zero memory
    while len(prev_memory_bank) < 16:
        prev_memory_bank.append(torch.zeros(memory_shape, device=device))
        prev_obj_ptr.append(torch.zeros(obj_ptr_shape, device=device))
    
    prev_memory_bank = torch.stack(prev_memory_bank, dim=1)
    prev_obj_ptr = torch.stack(prev_obj_ptr, dim=1)
    
    if training:
        randperm = torch.randperm(num_maskmem)
        prev_memory_bank[:, :num_maskmem] = prev_memory_bank[:, randperm]
        prev_obj_ptr[:, :num_maskmem] = prev_obj_ptr[:, randperm]
    
    if offload_to_cpu:
        next_image_feat = next_image_feat.detach().cpu()
        curr_memory_feat = curr_memory_feat.detach().cpu()
        curr_obj_ptr = curr_obj_ptr.detach().cpu()
        
    rl_state = {
        "frame_idx": frame_idx,
        "next_image_feat": next_image_feat,
        "curr_memory_feat": {
            "mem_feat": curr_memory_feat,
            "obj_ptr": curr_obj_ptr,
        },
        "prev_memory_bank": {
            "mem_feat": prev_memory_bank,
            "obj_ptr": prev_obj_ptr,
        }
    }
    
    state = RLStates(**rl_state)
    list_frame = list(output_dict["non_cond_frame_outputs"].keys())
    if training:
        avail_index = torch.argsort(randperm)[:len(non_cond_bank_list)].tolist()
        empty_index = torch.argsort(randperm)[len(non_cond_bank_list):].tolist()
        action_frame_map = {action+2:list_frame[i] for i, action in enumerate(avail_index)}
    else:
        action_frame_map = {k+2:v for k, v in enumerate(list_frame)}
        empty_index = [i for i in range(num_maskmem) if i not in action_frame_map.keys()]
        
    return state, action_frame_map

def compute_loss(
    pred_masks,
    gt_masks,
    inference_state
):
    dice_loss_fn = DiceLoss(sigmoid=True)
    focal_loss_fn = FocalLoss()

    video_H = inference_state["video_height"]
    video_W = inference_state["video_width"]
    if pred_masks.shape[-2:] == (video_H, video_W):
        video_res_masks = pred_masks.squeeze()
    else:
        video_res_masks = torch.nn.functional.interpolate(
            pred_masks,
            size=(video_H, video_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        
    loss = dice_loss_fn(video_res_masks, gt_masks) + 20 * focal_loss_fn(video_res_masks, gt_masks)
    
    return loss