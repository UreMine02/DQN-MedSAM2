"""
Q-learning utilities cho SAM2 memory management.
"""

import torch
import torch.nn.functional as F
from func_3d.utils import CombinedLoss

EPS = 1e-6


def compute_relative_loss(model, memory_bank, candidate, dropped=None, lossfunc=None, inputs=None, targets=None):
    """
    Reward = (loss_before - loss_after) / (|loss_before| + eps)

    Parameters
    ----------
    model : nn.Module
    memory_bank : list
    candidate : dict (candidate memory)
    dropped : dict or None (if yes, drop out of bank before add new candidate)
    lossfunc : callable (now, leverage Combined Loss)
    inputs : torch.Tensor
    targets : torch.Tensor (gt to cal Loss)
    """
    loss_before = evaluate_loss(model, memory_bank, lossfunc, inputs, targets)

    temp_bank = memory_bank.copy()
    if dropped is not None and dropped in temp_bank:
        temp_bank.remove(dropped)
    temp_bank.append(candidate)

    loss_after = evaluate_loss(model, temp_bank, lossfunc, inputs, targets)

    reward = (loss_before - loss_after) / (abs(loss_before) + EPS)
    return float(reward)


def evaluate_loss(model, memory_bank, lossfunc, inputs, targets):
    """
    Cal loss by forwarding model with mmbank
    """
    model.eval()
    with torch.no_grad():
        outputs = model(inputs, memory=memory_bank)
        loss = lossfunc(outputs, targets)
    return loss.item()

def compute_loss_for_frame(predictor, inference_state, frame_idx,
                           obj_output_dict=None, temp_mem_list=None, maskmem=True,
                           device="cuda"):
    lossfunc = CombinedLoss()

    # get ground truth from inference_state
    gt_all = inference_state.get("gt_masks", None)
    if gt_all is None or frame_idx >= len(gt_all):
        return 1.0

    gt = gt_all[frame_idx].to(device=device, dtype=torch.float32)

    # get obj_out
    obj_out = None
    if obj_output_dict is not None:
        obj_out = obj_output_dict.get(frame_idx, None)
    if obj_out is None and temp_mem_list is not None and len(temp_mem_list) > 0:
        last = temp_mem_list[-1]
        if isinstance(last, dict):
            obj_out = last
    if obj_out is None or "pred_masks" not in obj_out:
        return 1.0

    pred = obj_out["pred_masks"].to(device=device, dtype=torch.float32)

    # reshape pred
    if pred.ndim == 4 and pred.shape[1] == 1:  # (N,1,H,W)
        gt = gt.unsqueeze(0).unsqueeze(0) if gt.ndim == 2 else gt.unsqueeze(1)  # (N,1,H,W)
    elif pred.ndim == 3:  # (N,H,W)
        gt = gt.unsqueeze(0) if gt.ndim == 2 else gt

    # Resize gt 
    if gt.shape[-2:] != pred.shape[-2:]:
        gt = F.interpolate(gt.float(), size=pred.shape[-2:], mode="nearest")

    # object_score_logits + iou
    obj_pred = obj_out.get("object_score_logits", None)
    iou_pred = obj_out.get("iou", None)
    if obj_pred is None:
        obj_pred = torch.zeros(pred.shape[0], device=device)

    try:
        loss = lossfunc(pred, gt, iou_pred, None, obj_pred)
    except Exception:
        # fallback BCEloss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, gt)

    if isinstance(loss, (tuple, list)):
        return sum(float(x) for x in loss if x is not None)
    return float(loss.item()) if hasattr(loss, "item") else float(loss)

