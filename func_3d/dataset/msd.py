import os
import glob
import time
import random
import nibabel as nib
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision.transforms import v2
from torchvision import tv_tensors
from monai import transforms


def scaling(image, scale=255):
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min)/(image_max-image_min) * scale
    return image

def remove_negative_samples(image, mask):
    pos_slices = np.sum(mask, axis=(0,1)) > 0
    return image[:, :, pos_slices], mask[:, :, pos_slices]


class MSD(Dataset):
    def __init__(self, args, mode="train"):
        assert mode in ["train", "test"], f"mode must be either 'train' or 'test', got {mode}"
        self.subset = "Tr" if mode == 'train' else 'Ts'
        self.root = args.data_path
        self.mode = mode
        df = []
        
        csv_root = "./data/MSD"
        for csv_path in os.listdir(csv_root):
            if not csv_path.startswith(args.task) or not csv_path.endswith(f"{self.subset}.csv"):
                continue
            
            df.append(pd.read_csv(os.path.join(csv_root, csv_path), index_col=0))
        
        df = pd.concat(df)
        self.gt_path = np.asarray(df["gt_path"])
        self.task = np.asarray(df["task"])
        self.obj_id = np.asarray(df["obj_id"])
        self.n_pos = np.asarray(df["n_pos"])
        
        self.image_size = args.image_size
        self.num_support = args.num_support
        self.max_slices = args.video_length
        
        self.tr_transform = v2.Compose([
            # v2.Resize(size=(self.image_size, self.image_size)),
            v2.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.7, 1.4), ratio=(1.0, 1.0)),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomAffine(degrees=25),
            v2.ColorJitter(brightness=0.25, contrast=0.25)
        ])
        
        self.ts_transform = v2.Compose([
            v2.Resize(size=(self.image_size, self.image_size)),
        ])
        
    def __len__(self):
        return len(self.gt_path)
    
    def __getitem__(self, index):
        task = self.task[index]
        obj_id = self.obj_id[index]
        support_list = (self.task == self.task[index]) & \
                        (self.obj_id == self.obj_id[index]) & \
                        (self.n_pos >= self.num_support)
        support_list = [i for i in np.argwhere(support_list).squeeze() if i != index]
        support_index = np.random.choice(support_list, size=1)[0]

        label_path = os.path.join(self.root, self.gt_path[index])
        image_path = os.path.join(self.root, label_path.replace("label", "image"))

        support_label_path = os.path.join(self.root, self.gt_path[support_index])
        support_image_path = os.path.join(self.root, support_label_path.replace("label", "image"))
        
        (
            image_3d,
            data_seg_3d,
            support_image_3d,
            support_data_seg_3d,
            orig_size
        ) = self.load_data(image_path, label_path, support_image_path, support_label_path, obj_id)
        
        output_dict = {
            "image": image_3d, "label": data_seg_3d,
            "support_image": support_image_3d, "support_label": support_data_seg_3d,
            "task": task, "obj_id": obj_id, 
            "name": os.path.basename(image_path), "support_name": os.path.basename(support_image_path),
            "orig_size": tuple(orig_size)
        }
        
        return output_dict
    
    def load_data(self, image_path, label_path, support_image_path, support_label_path, obj_id):
        image_3d, data_seg_3d = self.load_image_label(
            image_path,
            label_path,
            obj_id = obj_id,
            max_slices=self.max_slices if self.mode == "train" else -1,
            slice_selection='contiguous',
            is_support=False
        )
        support_image_3d, support_data_seg_3d = self.load_image_label(
            support_image_path,
            support_label_path,
            obj_id = obj_id,
            max_slices=self.num_support,
            slice_selection='random' if self.mode == 'train' else 'evenly',
            is_support=True
        )
        
        image_3d = torch.rot90(torch.tensor(image_3d)).permute(2, 0, 1).unsqueeze(1).repeat(1, 3, 1, 1)
        data_seg_3d = torch.rot90(torch.tensor(data_seg_3d)).permute(2, 0, 1)
        support_image_3d = torch.rot90(torch.tensor(support_image_3d)).permute(2, 0, 1).unsqueeze(1).repeat(1, 3, 1, 1)
        support_data_seg_3d = torch.rot90(torch.tensor(support_data_seg_3d)).permute(2, 0, 1)
        
        orig_size = image_3d.shape[-2:]
        
        image_3d = tv_tensors.Image(image_3d)
        data_seg_3d = tv_tensors.Mask(data_seg_3d)
        support_image_3d = tv_tensors.Image(support_image_3d)
        support_data_seg_3d = tv_tensors.Mask(support_data_seg_3d)
        
        if self.mode == "train":
            transform = self.tr_transform
        else:
            transform = self.ts_transform
            
        image_3d, data_seg_3d = transform(image_3d, data_seg_3d)
        support_image_3d, support_data_seg_3d = transform(support_image_3d, support_data_seg_3d)

        return image_3d, data_seg_3d, support_image_3d, support_data_seg_3d, orig_size

    def load_image_label(self, image_path, label_path, obj_id, max_slices=16, slice_selection='contiguous', is_support=False):
        image_3d = nib.load(image_path)
        data_seg_3d = nib.load(label_path)
        image_3d = image_3d.dataobj
        data_seg_3d = data_seg_3d.dataobj
        
        if image_3d.ndim == 4:
            if image_3d.shape[-1] == 4:
                image_3d = image_3d[..., 2]
            elif image_3d.shape[-1] == 2:
                image_3d = image_3d[..., 0]
                
        image_3d = np.asarray(image_3d, dtype=np.float32)
        data_seg_3d = np.asarray(data_seg_3d, dtype=np.float32)
        data_seg_3d[data_seg_3d != obj_id] = 0
        
        if self.mode == "train" and not is_support:
        # if False:
            pos_slices = np.argwhere(np.sum(data_seg_3d, axis=(0,1))).squeeze()
            
            from_idx, to_idx = pos_slices.min() - (max_slices // 2), pos_slices.max() + (max_slices // 2)
            image_3d = image_3d[:, :, max(from_idx, 0):to_idx]
            data_seg_3d = data_seg_3d[:, :, max(from_idx, 0):to_idx]
        else:
            pos_slices = np.sum(data_seg_3d, axis=(0,1)) > 0
            image_3d = image_3d[:, :, pos_slices]
            data_seg_3d = data_seg_3d[:, :, pos_slices]
            
        
        if image_3d.shape[-1] > max_slices and max_slices > 0:
            if slice_selection == 'contiguous':
                start_slice = np.random.choice(range(image_3d.shape[-1] - max_slices + 1))
                image_3d = image_3d[..., start_slice:start_slice+max_slices]
                data_seg_3d = data_seg_3d[..., start_slice:start_slice+max_slices]
            elif slice_selection == 'random':
                n_slice = max_slices if self.mode != 'train' else np.random.randint(1, max_slices + 1)
                slice_indices = np.random.choice(image_3d.shape[-1], size=n_slice, replace=False)
                image_3d = image_3d[..., slice_indices]
                data_seg_3d = data_seg_3d[..., slice_indices]
            elif slice_selection == 'evenly':
                s = image_3d.shape[-1] // (max_slices + 1)
                slice_indices = np.linspace(0, image_3d.shape[-1]-1, max_slices).round().astype(np.int16)
                image_3d = image_3d[..., slice_indices]
                data_seg_3d = data_seg_3d[..., slice_indices]
            else:
                raise ValueError(f"Slice selection method {slice_selection} not supported yet, please provide value in ['contiguous', 'random', 'evenly']")                 

        image_3d = scaling(image_3d, scale=1)
        
        return image_3d, data_seg_3d
    
    def resize(self, image_3d, data_seg_3d):
        image_3d = F.interpolate(image_3d, size=(image_3d.shape[2], self.image_size, self.image_size), mode='trilinear', align_corners=False)
        data_seg_3d = F.interpolate(data_seg_3d, size=(data_seg_3d.shape[2], self.image_size, self.image_size), mode='nearest')
        image_3d = image_3d.squeeze(0).repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
        data_seg_3d = data_seg_3d.squeeze(0).squeeze(0)
        
        return image_3d, data_seg_3d