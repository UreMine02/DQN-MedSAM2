"""Utility functions for training and evaluation.
    Yunli Qi
"""

import logging
import os
import random
import sys
import time
from datetime import datetime

import dateutil.tz
import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss
from torch.nn import L1Loss, BCEWithLogitsLoss
from sklearn.cluster import KMeans
from collections import defaultdict

import cfg

args = cfg.parse_args()
device = torch.device('cuda', args.gpu_device)

def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """

    if net == 'sam2':
        from sam2_train.build_sam import build_sam2_video_predictor

        sam2_checkpoint = args.sam_ckpt
        print(args)
        model_cfg = args.sam_config

        net = build_sam2_video_predictor(config_file=model_cfg, ckpt_path=sam2_checkpoint, mode=None)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        net = net.to(device=gpu_device)
        q_agent = getattr(net, "q_agent", None)
        if q_agent is not None:
            q_agent.to(device=gpu_device)

    return net

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def random_click(mask, point_labels = 1, seed=None):
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label) 
    # return point_labels, indices[np.random.randint(len(indices))]
    if seed is not None:
        rand_instance = random.Random(seed)
        rand_num = rand_instance.randint(0, len(indices) - 1)
    else:
        rand_num = random.randint(0, len(indices) - 1)
    output_index_1 = indices[rand_num][0]
    output_index_0 = indices[rand_num][1]
    return point_labels, np.array([output_index_0, output_index_1])

def generate_bbox(mask, variation=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # check if all masks are black
    if len(mask.shape) != 2:
        current_shape = mask.shape
        raise ValueError(f"Mask shape is not 2D, but {current_shape}")
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    # max agreement position
    indices = np.argwhere(mask == max_label) 
    # return point_labels, indices[np.random.randint(len(indices))]
    # print(indices)
    x0 = np.min(indices[:, 0])
    x1 = np.max(indices[:, 0])
    y0 = np.min(indices[:, 1])
    y1 = np.max(indices[:, 1])
    w = x1 - x0
    h = y1 - y0
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    if variation > 0:
        num_rand = np.random.randn() * variation
        w *= 1 + num_rand[0]
        h *= 1 + num_rand[1]
        x1 = mid_x + w / 2
        x0 = mid_x - w / 2
        y1 = mid_y + h / 2
        y0 = mid_y - h / 2
    return np.array([y0, x0, y1, x1])

def calculate_bounding_box(mask):
    """Calculate bounding box metrics for a given binary mask."""
    coords = torch.nonzero(mask)
    if coords.size(0) == 0:  # No mask present
        return None
    y_min, x_min = coords.min(dim=0).values
    y_max, x_max = coords.max(dim=0).values
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return (width.item(), height.item(), center_x.item(), center_y.item())

def sample_diverse_support(support_imgs_tensor, support_masks_tensor, num_samples=5):
    """
    Sample diverse support images based on size and position metrics.
    Args:
        support_imgs_tensor: Tensor of support images (N, C, H, W).
        support_masks_tensor: Tensor of support masks (N, H, W).
        num_samples: Number of diverse samples to return.
    Returns:
        sampled_imgs: Tensor of sampled support images.
        sampled_masks: Tensor of sampled support masks.
    """
    bbox_metrics = []
    for idx in range(support_masks_tensor.shape[0]):
        bbox = calculate_bounding_box(support_masks_tensor[idx])
        if bbox:
            width, height, center_x, center_y = bbox
            bbox_metrics.append((idx, width, height, center_x, center_y))

    if not bbox_metrics:
        # Fallback to random sampling if no valid masks are found
        sampled_indices = torch.randperm(support_imgs_tensor.shape[0])[:num_samples]
        return support_imgs_tensor[sampled_indices], support_masks_tensor[sampled_indices]

    # Normalize bounding box metrics for clustering
    bbox_metrics = np.array(bbox_metrics)
    normalized_metrics = bbox_metrics[:, 1:] / bbox_metrics[:, 1:].max(axis=0)

    # Cluster metrics to ensure diversity
    num_clusters = min(num_samples, len(bbox_metrics))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_metrics)

    # Sample one image from each cluster
    sampled_indices = []
    for cluster in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) == 0:
            cluster_indices = np.arange(support_imgs_tensor.shape[0])  # Fallback to all indices
        sampled_indices.append(np.random.choice(cluster_indices))

    sampled_indices = [bbox_metrics[idx, 0] for idx in sampled_indices]  # Map back to original indices
    sampled_indices = torch.tensor(sampled_indices, dtype=torch.long)

    return support_imgs_tensor[sampled_indices], support_masks_tensor[sampled_indices]

def eval_seg(pred, mask):
    pred = torch.where(torch.sigmoid(pred)>=0.5, 1, 0)
    dice = dice_score(pred, mask)
    iou = iou_score(pred, mask)
    return dice, iou
    
def dice_score(pred, mask, smoothing=1e-6):
    pred = pred.reshape(-1)
    mask = mask.reshape(-1)
    
    interaction = torch.sum(pred * mask)
    denominator = torch.sum(pred) + torch.sum(mask)

    dice_score = (2*interaction+smoothing)/(denominator+smoothing)
    return dice_score

def iou_score(pred, mask, smoothing=1e-6):
    pred = pred.reshape(-1)
    mask = mask.reshape(-1)
    
    interaction = torch.sum(pred * mask)
    denominator = torch.count_nonzero(pred + mask)

    iou = (interaction+smoothing)/(denominator+smoothing)
    return iou

def precision_score(pred, target, eps=1e-6):
    """
    Precision: TP / (TP + FP)
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    true_positive = torch.sum(pred_flat * target_flat)
    false_positive = torch.sum(pred_flat * (1 - target_flat))
    
    precision = (true_positive + eps) / (true_positive + false_positive + eps)
    return precision

def recall_score(pred, target, eps=1e-6):
    """
    Recall: TP / (TP + FN)
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    true_positive = torch.sum(pred_flat * target_flat)
    false_negative = torch.sum((1 - pred_flat) * target_flat)
    
    recall = (true_positive + eps) / (true_positive + false_negative + eps)
    return recall


# def eval_seg(pred,true_mask_p,threshold):
#     '''
#     threshold: a int or a tuple of int
#     masks: [b,2,h,w]
#     pred: [b,2,h,w]
#     '''
#     b, c, h, w = pred.size()
#     if c == 2:
#         iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
#         for th in threshold:

#             gt_vmask_p = (true_mask_p > th).float()
#             vpred = (pred > th).float()
#             vpred_cpu = vpred.cpu()
#             disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
#             cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

#             disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
#             cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
#             '''iou for numpy'''
#             iou_d += iou(disc_pred,disc_mask)
#             iou_c += iou(cup_pred,cup_mask)

#             '''dice for torch'''
#             disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
#             cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
#         return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
#     elif c > 2: # for multi-class segmentation > 2 classes
#         ious = [0] * c
#         dices = [0] * c
#         for th in threshold:
#             gt_vmask_p = (true_mask_p > th).float()
#             vpred = (pred > th).float()
#             vpred_cpu = vpred.cpu()
#             for i in range(0, c):
#                 pred = vpred_cpu[:,i,:,:].numpy().astype('int32')
#                 mask = gt_vmask_p[:,i,:,:].squeeze(1).cpu().numpy().astype('int32')
        
#                 '''iou for numpy'''
#                 ious[i] += iou(pred,mask)

#                 '''dice for torch'''
#                 dices[i] += dice_coeff(vpred[:,i,:,:], gt_vmask_p[:,i,:,:]).item()
            
#         return tuple(np.array(ious + dices) / len(threshold)) # tuple has a total number of c * 2
#     else:
#         eiou, edice = 0,0
#         for th in threshold:

#             gt_vmask_p = (true_mask_p > th).float()
#             vpred = (pred > th).float()
#             vpred_cpu = vpred.cpu()
#             disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

#             disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
#             '''iou for numpy'''
#             eiou += iou(disc_pred,disc_mask)

#             '''dice for torch'''
#             edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
#         return eiou / len(threshold), edice / len(threshold)
    
def iou(outputs: np.array, labels: np.array):

    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

# class CombinedLoss(nn.Module):
#     def __init__(self, dice_weight=1, focal_weight=20, mae_weight=1):
#         super(CombinedLoss, self).__init__()
#         self.dice_weight = dice_weight
#         self.focal_weight = focal_weight
#         self.mae_weight = mae_weight

#         self.dice_loss = DiceLoss(sigmoid=True)
#         self.focal_loss = FocalLoss()
#         self.mae_loss = L1Loss()
        
#     def forward(self, inputs, targets, iou_pred, iou_gt):
#         dice = self.dice_loss(inputs, targets)
#         focal = self.focal_loss(inputs, targets)
#         mae = self.mae_loss(iou_pred, iou_gt)
#         return self.dice_weight*dice, self.focal_weight*focal, self.mae_weight*mae

# Occlusion score is included in the loss function
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1, focal_weight=20, mae_weight=1, bce_weight=1):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.mae_weight = mae_weight
        self.bce_weight = bce_weight

        self.dice_loss = DiceLoss(sigmoid=True)
        self.focal_loss = FocalLoss()
        self.mae_loss = L1Loss()
        self.bce_loss = FocalLoss()
        
    def forward(self, inputs, targets, iou_pred, iou_gt, obj_pred):
        obj_pred = obj_pred.view(1, -1)

        if (targets == 0).all():
            bce = self.bce_loss(obj_pred, torch.zeros(obj_pred.shape).to(device=obj_pred.device))
        else:
            bce = self.bce_loss(obj_pred, torch.ones(obj_pred.shape).to(device=obj_pred.device))
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        mae = self.mae_loss(iou_pred, iou_gt)
        
        return self.dice_weight*dice, self.focal_weight*focal, self.mae_weight*mae, self.bce_weight*bce

def update_loss(loss_dict, focal_loss, dice_loss, mae_loss, bce_loss):
    loss_dict["focal_loss"] += focal_loss
    loss_dict["dice_loss"] += dice_loss
    loss_dict["mae_loss"] += mae_loss
    loss_dict["bce_loss"] += bce_loss

def average_loss(loss_dict):
    if loss_dict["num_step"] == 0:
        # Avoid division by zero
        loss_dict["focal_loss"] = 0
        loss_dict["dice_loss"] = 0
        loss_dict["mae_loss"] = 0
        loss_dict["bce_loss"] = 0
        loss_dict["total_loss"] = 0
        return
    loss_dict["focal_loss"] = loss_dict["focal_loss"]/(loss_dict["num_step"]+ 1e-6)
    loss_dict["dice_loss"] = loss_dict["dice_loss"]/(loss_dict["num_step"]+ 1e-6)
    loss_dict["mae_loss"] = loss_dict["mae_loss"]/(loss_dict["num_step"]+ 1e-6)
    loss_dict["bce_loss"] = loss_dict["bce_loss"]/(loss_dict["num_step"]+ 1e-6)
    loss_dict["total_loss"] += loss_dict["focal_loss"] + loss_dict["dice_loss"] + loss_dict["mae_loss"] + loss_dict["bce_loss"]

def update_score(score_dict, dice_score, iou_score):
    score_dict["dice_score"] += dice_score
    score_dict["iou_score"] += iou_score

def average_score(score_dict):
    score_dict["dice_score"] = score_dict["dice_score"]/(score_dict["num_step"] + 1e-6)
    score_dict["iou_score"] = score_dict["iou_score"]/(score_dict["num_step"] + 1e-6)
    score_dict["total_score"] += score_dict["dice_score"] + score_dict["iou_score"]

def extract_object(images_tensor, masks_tensor, support_images_tensor, support_masks_tensor, obj_id, video_length, num_support):
    obj_mask = masks_tensor == obj_id
    channels_with_true = torch.argwhere(torch.any(obj_mask, axis=(1, 2))).flatten()
    if channels_with_true.numel() == 0:
        print(f"[EXTRACT QUERY] No valid query slices found for obj_id={obj_id}.")
        return None
    min_slice, max_slice = channels_with_true.min(), channels_with_true.max()
    obj_mask = obj_mask[min_slice:max_slice+1]
    obj_image = images_tensor[min_slice:max_slice+1]

    if video_length is not None and obj_mask.shape[0] > video_length:
        starting_frame = np.random.randint(low=0, high=obj_mask.shape[0]-video_length)
        obj_mask = obj_mask[starting_frame:starting_frame+video_length]
        obj_image = obj_image[starting_frame:starting_frame+video_length]
    
    # channels_class_4 = torch.argwhere(torch.any(obj_mask == 4, axis=(1, 2))).flatten()
    # channels_class_12 = torch.argwhere(torch.any(obj_mask == 12, axis=(1, 2))).flatten()
    # if obj_id in [4, 12]:
    #     print(f"[EXTRACT QUERY - MIDDLE] Class 4={len(channels_class_4)} slices, Class 12={len(channels_class_12)} slices")
    
    output_dict = {
        "image": obj_image,
        "label": obj_mask,
    } 

    # print(f"[EXTRACT - SUPPORT - BEFORE MERGE] obj_id={obj_id}")
    # slice_distribution_support_before_merge = defaultdict(int)
    # slice_distribution_support_after_merge = defaultdict(int)

    # # Count slices before merging for support
    # for frame_index in range(support_masks_tensor.shape[0]):
    #     unique_classes = torch.unique(support_masks_tensor[frame_index])
    #     for cls in unique_classes:
    #         if cls.item() in [4, 12]:
    #             slice_distribution_support_before_merge[int(cls.item())] += 1
    # print(f"  Class 4: {slice_distribution_support_before_merge[4]} slices")
    # print(f"  Class 12: {slice_distribution_support_before_merge[12]} slices")
    # if obj_id in [4, 12]:  # Debugging specific classes
    #     print(f"[DEBUG] Initial obj_mask created for Class {obj_id}:")
    #     print(f"  obj_mask.shape: {obj_mask.shape}")
    #     print(f"  Unique values in obj_mask: {torch.unique(obj_mask)}")
    #     num_slices_with_obj = torch.sum(torch.any(obj_mask, axis=(1, 2))).item()
    #     print(f"  Number of slices containing Class {obj_id}: {num_slices_with_obj}")
    
    # Support Processing
    if obj_id in [4, 12]:
        # print(f"[EXTRACT SUPPORT - BEFORE RESAMPLING] obj_id={obj_id}")
        class_slices_before = torch.sum(support_masks_tensor == obj_id, dim=(1, 2)).nonzero(as_tuple=True)[0].shape[0]
        # print(f"  Total slices containing obj_id={obj_id}: {class_slices_before}")
    obj_mask = support_masks_tensor == obj_id
    channels_with_true = torch.argwhere(torch.any(obj_mask, axis=(1, 2))).flatten()
    if channels_with_true.numel() == 0:  # No valid slices in the support set for the target class
        print(f"[EXTRACT SUPPORT] No valid support slices found for obj_id={obj_id}.")
        return None
    if len(channels_with_true) > num_support:
        selected_indices = torch.tensor(np.random.choice(channels_with_true.cpu(), size=num_support, replace=False))
    else:
        selected_indices = channels_with_true

    selected_obj_mask = obj_mask[selected_indices, ...]
    selected_obj_image = support_images_tensor[selected_indices, ...]
    # Support slices after resampling
    # Debugging slice counts for obj_id 4 and 12
    if obj_id in [4, 12]:
        num_slices = torch.sum(selected_obj_mask == 1, dim=(1, 2)).nonzero(as_tuple=True)[0].shape[0]
        # print(f"[EXTRACT SUPPORT - AFTER RESAMPLING] Class {obj_id}")
        # print(f"  Total Slices: {num_slices}")

    # min_slice, max_slice = channels_with_true.min(), channels_with_true.max()

    # obj_mask = obj_mask[min_slice:max_slice+1]
    # obj_image = support_images_tensor[min_slice:max_slice+1]
    # if obj_id in [4, 12]:  # Debugging after cropping
    #     print(f"[DEBUG] obj_mask after cropping for Class {obj_id}:")
    #     print(f"  obj_mask.shape: {obj_mask.shape}")
    #     num_slices_with_obj_after_crop = torch.sum(torch.any(obj_mask, axis=(1, 2))).item()
    #     print(f"  Number of slices containing Class {obj_id} after cropping: {num_slices_with_obj_after_crop}")

    # obj_num_frame = obj_mask.shape[0]
    # if num_support < obj_num_frame:
    #     support_selected_frame = torch.tensor(np.random.choice(np.arange(0, obj_num_frame), size=num_support, replace=False))
    # else:
    #     support_selected_frame = list(range(obj_num_frame))
    
    

    # Count slices after merging and cropping for support
    # for frame_index in range(selected_obj_mask.shape[0]):
    #     unique_classes = torch.unique(selected_obj_mask[frame_index])
    #     for cls in unique_classes:
    #         if cls.item() in [4, 12]:
    #             slice_distribution_support_after_merge[int(cls.item())] += 1
    # print(f"[EXTRACT - SUPPORT - AFTER MERGE & CROPPING]")
    # print(f"  Class 4: {slice_distribution_support_after_merge[4]} slices")
    # print(f"  Class 12: {slice_distribution_support_after_merge[12]} slices")
    # channels_class_4 = torch.argwhere(torch.any(selected_obj_mask == 4, axis=(1, 2))).flatten()
    # channels_class_12 = torch.argwhere(torch.any(selected_obj_mask == 12, axis=(1, 2))).flatten()

    output_dict.update(
        {
            "support_image": selected_obj_image,
            "support_label": selected_obj_mask,
        }
    )

    return output_dict 

def extract_object_multiple(images_tensor, masks_tensor, support_images_list, support_masks_list, obj_id, video_length, num_support, num_scans):
    """
    Extract support sets for the target class (`obj_id`) from multiple CT scans.
    
    Args:
        images_tensor (torch.Tensor): Query image tensor (single CT scan).
        masks_tensor (torch.Tensor): Query mask tensor (single CT scan).
        support_images_list (list of torch.Tensor): List of image tensors from multiple CT scans.
        support_masks_list (list of torch.Tensor): List of mask tensors from multiple CT scans.
        obj_id (int): Target class ID.
        video_length (int): Number of slices in the query.
        num_support (int): Total number of support slices.
        num_scans (int): Maximum number of CT scans to sample from.
    
    Returns:
        dict: A dictionary containing the query and support tensors.
    """
    # Extract query slices for the target class
    obj_mask = masks_tensor == obj_id
    if obj_mask.dim() < 3:
        raise ValueError(f"Unexpected obj_mask dimensions: {obj_mask.shape}")
    
    channels_with_true = torch.argwhere(torch.any(obj_mask, dim=(1, 2))).flatten()
    if channels_with_true.numel() == 0:
        raise ValueError(f"No slices found for class {obj_id} in the query.")
    
    min_slice, max_slice = channels_with_true.min(), channels_with_true.max()
    obj_mask = obj_mask[min_slice:max_slice+1]
    obj_image = images_tensor[min_slice:max_slice+1]

    if video_length is not None and obj_mask.shape[0] > video_length:
        starting_frame = np.random.randint(low=0, high=obj_mask.shape[0] - video_length)
        obj_mask = obj_mask[starting_frame:starting_frame + video_length]
        obj_image = obj_image[starting_frame:starting_frame + video_length]

    output_dict = {
        "image": obj_image,
        "label": obj_mask,
    }

    # Extract support slices from multiple CT scans
    support_images = []
    support_masks = []

    # Shuffle CT scans to randomize the selection
    combined_data = list(zip(support_images_list, support_masks_list))
    np.random.shuffle(combined_data)

    # Select up to `num_scans` CT scans
    selected_ct_scans = combined_data[:num_scans]

    for support_images_tensor, support_masks_tensor in selected_ct_scans:
        # Extract slices containing the target class
        obj_mask = support_masks_tensor == obj_id
        channels_with_true = torch.argwhere(torch.any(obj_mask, dim=(1, 2))).flatten()

        if channels_with_true.numel() > 0:  # Ensure the target class exists in this scan
            min_slice, max_slice = channels_with_true.min(), channels_with_true.max()
            obj_mask = obj_mask[min_slice:max_slice+1]
            obj_image = support_images_tensor[min_slice:max_slice+1]

            # Add to the support set
            support_images.append(obj_image)
            support_masks.append(obj_mask)

    # Combine support slices from all selected CT scans
    if len(support_images) > 0:
        support_images = torch.cat(support_images, dim=0)
        support_masks = torch.cat(support_masks, dim=0)

        # Randomly sample `num_support` slices if more are available
        obj_num_frame = support_masks.shape[0]
        if num_support < obj_num_frame:
            selected_idx = np.random.choice(obj_num_frame, num_support, replace=False)
            support_images = support_images[selected_idx]
            support_masks = support_masks[selected_idx]
    else:
        raise ValueError(f"No valid support slices found for class {obj_id}.")

    output_dict.update(
        {
            "support_image": support_images,
            "support_label": support_masks,
        }
    )

    return output_dict


# def extract_object_multiple(images_tensor, masks_tensor, support_images_list, support_masks_list, obj_id, video_length, num_support, num_scans):
#     """
#     Extract support sets for the target class (`obj_id`) from multiple CT scans.
#     Args:
#         images_tensor (torch.Tensor): Query image tensor (single CT scan).
#         masks_tensor (torch.Tensor): Query mask tensor (single CT scan).
#         support_images_list (list of torch.Tensor): List of image tensors from multiple CT scans.
#         support_masks_list (list of torch.Tensor): List of mask tensors from multiple CT scans.
#         obj_id (int): Target class ID.
#         video_length (int): Number of slices in the query.
#         num_support (int): Total number of support slices.
#         num_scans (int): Maximum number of CT scans to sample from.
#     Returns:
#         dict: A dictionary containing the query and support tensors.
#     """
#     # Extract query slices for the target class
#     obj_mask = masks_tensor == obj_id
#     channels_with_true = torch.argwhere(torch.any(obj_mask, axis=(1, 2))).flatten()
#     min_slice, max_slice = channels_with_true.min(), channels_with_true.max()

#     obj_mask = obj_mask[min_slice:max_slice + 1]
#     obj_image = images_tensor[min_slice:max_slice + 1]

#     if video_length is not None and obj_mask.shape[0] > video_length:
#         starting_frame = np.random.randint(low=0, high=obj_mask.shape[0] - video_length)
#         obj_mask = obj_mask[starting_frame:starting_frame + video_length]
#         obj_image = obj_image[starting_frame:starting_frame + video_length]

#     output_dict = {
#         "image": obj_image,
#         "label": obj_mask,
#     }

#     # Extract support slices from multiple CT scans
#     support_images = []
#     support_masks = []

#     # Shuffle CT scans to randomize the selection
#     combined_data = list(zip(support_images_list, support_masks_list))
#     np.random.shuffle(combined_data)

#     # Select up to `num_scans` CT scans
#     selected_ct_scans = combined_data[:num_scans]

#     for support_images_tensor, support_masks_tensor in selected_ct_scans:
#         # Extract slices containing the target class
#         obj_mask = support_masks_tensor == obj_id
#         channels_with_true = torch.argwhere(torch.any(obj_mask, axis=(1, 2))).flatten()

#         if channels_with_true.numel() > 0:  # Ensure the target class exists in this scan
#             min_slice, max_slice = channels_with_true.min(), channels_with_true.max()
#             obj_mask = obj_mask[min_slice:max_slice + 1]
#             obj_image = support_images_tensor[min_slice:max_slice + 1]

#             # Add to the support set
#             support_images.append(obj_image)
#             support_masks.append(obj_mask)

#     # Combine support slices from all selected CT scans
#     if len(support_images) > 0:
#         support_images = torch.cat(support_images, dim=0)
#         support_masks = torch.cat(support_masks, dim=0)

#         # Randomly sample `num_support` slices if more are available
#         obj_num_frame = support_masks.shape[0]
#         if num_support < obj_num_frame:
#             selected_idx = np.random.choice(obj_num_frame, num_support, replace=False)
#             support_images = support_images[selected_idx]
#             support_masks = support_masks[selected_idx]

#     output_dict.update(
#         {
#             "support_image": support_images,
#             "support_label": support_masks,
#         }
#     )

#     return output_dict



