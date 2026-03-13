"""
Q-learning utilities cho SAM2 memory management.
"""
import math
import random
import torch
import numpy as np
from functools import partial
from monai.losses import DiceLoss, FocalLoss
from sam2_train.rl_modules.rl_components import RLStates
from sam2_train.modeling.position_encoding import compute_axial_cis

EPS = 1e-6

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_enc(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq).to(xq.device)


def rotary_emb(x, internal_dim=256, num_heads=1, rope_theta=10000.0):
    compute_cis = partial(
        compute_axial_cis, dim=internal_dim // num_heads, theta=rope_theta
    )
    
    # Apply rotary position encoding
    w = h = math.sqrt(x.shape[-2])
    freqs_cis = compute_cis(end_x=w, end_y=h).to(x.device)

    x = apply_rotary_enc(
        x,
        freqs_cis=freqs_cis
    )
    
    return x

def prepare_rl_state(
    current_vision_feats,
    current_vision_pos_embeds,
    output_dict,
    frame_idx,
    num_maskmem,
    num_max_prompt=10,
    offload_to_cpu=True,
    training=False
):
    next_image_feat = current_vision_feats[-1] + current_vision_pos_embeds[-1]
    next_image_feat = rotary_emb(next_image_feat)
    next_image_feat = next_image_feat.permute(1, 2, 0).reshape(1, 256, 64, 64)
    curr_memory_feat = output_dict["await_outputs"][frame_idx-1]
    curr_memory_feat = curr_memory_feat["maskmem_features"] + curr_memory_feat["maskmem_pos_enc"][0]
    curr_memory_feat = rotary_emb(curr_memory_feat)
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
        mem_feat = rotary_emb(mem_feat)
        obj_ptr = feat["obj_ptr"]
        if offload_to_cpu:
            mem_feat = mem_feat.cpu()
            obj_ptr = obj_ptr.cpu()
        prev_memory_bank.append(mem_feat)
        prev_obj_ptr.append(obj_ptr)
    
    # Append zero memory
    while len(prev_memory_bank) < num_maskmem:
        prev_memory_bank.append(torch.zeros(memory_shape, device=device))
        prev_obj_ptr.append(torch.zeros(obj_ptr_shape, device=device))

    # Add cond memory
    for feat in cond_bank_list:
        mem_feat = feat["maskmem_features"] + feat["maskmem_pos_enc"][0]
        mem_feat = rotary_emb(mem_feat)
        obj_ptr = feat["obj_ptr"]
        if offload_to_cpu:
            mem_feat = mem_feat.cpu()
            obj_ptr = obj_ptr.cpu()
        prev_memory_bank.append(mem_feat)
        prev_obj_ptr.append(obj_ptr)
    
    # Append zero memory
    while len(prev_memory_bank) < num_maskmem + num_max_prompt:
        prev_memory_bank.append(torch.zeros(memory_shape, device=device))
        prev_obj_ptr.append(torch.zeros(obj_ptr_shape, device=device))
    
    prev_memory_bank = torch.stack(prev_memory_bank, dim=1)
    prev_obj_ptr = torch.stack(prev_obj_ptr, dim=1)
    
    #TODO: Finding out why this part affecting testing
    # if training:
    #     randperm = torch.randperm(num_maskmem)
    #     prev_memory_bank[:, :num_maskmem] = prev_memory_bank[:, randperm]
    #     prev_obj_ptr[:, :num_maskmem] = prev_obj_ptr[:, randperm]
    
    if offload_to_cpu:
        next_image_feat = next_image_feat.detach().cpu()
        curr_memory_feat = curr_memory_feat.detach().cpu()
        curr_obj_ptr = curr_obj_ptr.detach().cpu()
        
    rl_state = {
        "frame_idx": frame_idx,
        "next_image_feat": next_image_feat.clone().detach(),
        "curr_memory_feat": {
            "mem_feat": curr_memory_feat.clone().detach(),
            "obj_ptr": curr_obj_ptr.clone().detach(),
        },
        "prev_memory_bank": {
            "mem_feat": prev_memory_bank.clone().detach(),
            "obj_ptr": prev_obj_ptr.clone().detach(),
        }
    }
    
    state = RLStates(**rl_state)
    list_frame = list(output_dict["non_cond_frame_outputs"].keys())
    # if training:
    #     avail_index = np.argsort(randperm)[:len(non_cond_bank_list)].tolist()
    #     action_frame_map = {action+2:list_frame[i] for i, action in enumerate(avail_index)}
    # else:
    action_frame_map = {k+2:v for k, v in enumerate(list_frame)}
    
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
        
    loss = dice_loss_fn(video_res_masks, gt_masks)
    
    return loss


def map_action(action, output_dict, storage_key):
    """
    Map an integer action to a drop_key (frame index) for the given storage_key.
    Returns None if no drop should happen (or no candidate to drop).
    Safety: never index into empty lists and prefer per-entry iou if available.
    """
    # get current keys in memory (ordered)
    mem_dict = output_dict.get(storage_key, {})
    sorted_keys = sorted(mem_dict.keys())
    if len(sorted_keys) == 0:
        return None

    # actions:
    # 0: skip
    # 1: add (if full, fallback to drop oldest)
    # 2: add_drop_oldest
    # 3: add_drop_lowest_iou
    # 4: add_drop_random

    # drop oldest (for action 1 or 2)
    if action == 2:
        return sorted_keys[0]

    # drop lowest IoU (action == 3)
    if action == 3:
        # try to collect IoU per stored entry (prefer entry['iou'] or entry['object_score_logits'])
        iou_list = [mem_dict[k]["ious"].item() for k in sorted_keys]
        min_idx = np.argmin(iou_list)
        return sorted_keys[min_idx]

    # drop random (action == 4)
    if action == 4:
        return random.choice(sorted_keys)

    # action == 1
    return None

