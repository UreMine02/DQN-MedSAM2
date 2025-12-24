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

SUPPORT_EXCLUDE = [
    "/data/datasets/Combined_Dataset/MSD/Task10_Colon/imagesTr/colon_104.nii.gz",
    "/data/datasets/Combined_Dataset/MSD/Task10_Colon/imagesTr/colon_176.nii.gz",
    "/data/datasets/Combined_Dataset/MSD/Task10_Colon/imagesTr/colon_089.nii.gz",
    "/data/datasets/Combined_Dataset/MSD/Task10_Colon/imagesTr/colon_095.nii.gz",
    "/data/datasets/Combined_Dataset/MSD/Task10_Colon/imagesTr/colon_134.nii.gz",
] # These volumes don't have enough slices to be support (their num of positive slices < args.support_size)

def normalization(image):
    image_min = np.min(image)
    image_max = np.max(image)
    image = ((image - image_min)/(image_max-image_min))*255
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
        
        self.image_size = args.image_size
        self.num_support = args.num_support
        self.max_slices = args.video_length
        
    def __len__(self):
        return len(self.gt_path)
    
    def __getitem__(self, index):
        task = self.task[index]
        obj_id = self.obj_id[index]
        support_list = np.argwhere(np.logical_and(self.task == self.task[index], self.obj_id == self.obj_id[index])).squeeze()
        support_index = np.random.choice(support_list, size=1)

        label_path = os.path.join(self.root, self.gt_path[index])
        image_path = os.path.join(self.root, label_path.replace("label", "image"))

        support_label_path = os.path.join(self.root, self.gt_path[support_index][0])
        support_image_path = os.path.join(self.root, support_label_path.replace("label", "image"))
        
        image_3d, data_seg_3d = self.load_image_label(
            image_path,
            label_path,
            obj_id = obj_id,
            max_slices=self.max_slices if self.mode == "train" else -1
        )
        support_image_3d, support_data_seg_3d = self.load_image_label(
            support_image_path,
            support_label_path,
            obj_id = obj_id,
            max_slices=self.num_support
        )
        
        output_dict = {
            "image": image_3d, "label": data_seg_3d,
            "support_image": support_image_3d, "support_label": support_data_seg_3d,
            "task": task, "obj_id": obj_id
        }
        
        return output_dict

    def load_image_label(self, image_path, label_path, obj_id, max_slices=16):
        image_3d = nib.load(image_path)
        data_seg_3d = nib.load(label_path)
        image_3d = image_3d.dataobj
        data_seg_3d = data_seg_3d.dataobj
        
        if image_3d.ndim == 4:
            if image_3d.shape[-1] == 4:
                image_3d = image_3d[..., 2]
            elif image_3d.shape[-1] == 2:
                image_3d = image_3d[..., 0]
                
        image_3d = np.asanyarray(image_3d)
        data_seg_3d = np.asanyarray(data_seg_3d)
        data_seg_3d[data_seg_3d != obj_id] = 0
        
        pos_slices = np.sum(data_seg_3d, axis=(0,1)) > 0
        image_3d = image_3d[:, :, pos_slices]
        data_seg_3d = data_seg_3d[:, :, pos_slices]
        
        if image_3d.shape[-1] > max_slices and max_slices > 0:
            start_slice = np.random.choice(range(image_3d.shape[-1] - max_slices + 1))
            image_3d = image_3d[..., start_slice:start_slice+max_slices]
            data_seg_3d = data_seg_3d[..., start_slice:start_slice+max_slices]

        image_3d = normalization(image_3d)
        image_3d = torch.rot90(torch.tensor(image_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        data_seg_3d = torch.rot90(torch.tensor(data_seg_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

        image_3d = F.interpolate(image_3d, size=(image_3d.shape[2], self.image_size, self.image_size), mode='trilinear', align_corners=False)
        data_seg_3d = F.interpolate(data_seg_3d, size=(data_seg_3d.shape[2], self.image_size, self.image_size), mode='nearest')
        image_3d = image_3d.squeeze(0).repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
        data_seg_3d = data_seg_3d.squeeze(0).squeeze(0)

        return image_3d, data_seg_3d