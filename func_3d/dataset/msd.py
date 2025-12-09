import os
import glob
import time
import random
import nibabel as nib
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SUPPORT_EXCLUDE = [
    "/data/datasets/Combined_Dataset/MSD/Task10_Colon/imagesTr/colon_104.nii.gz",
    "/data/datasets/Combined_Dataset/MSD/Task10_Colon/imagesTr/colon_176.nii.gz",
    "/data/datasets/Combined_Dataset/MSD/Task10_Colon/imagesTr/colon_089.nii.gz",
    "/data/datasets/Combined_Dataset/MSD/Task10_Colon/imagesTr/colon_095.nii.gz",
    "/data/datasets/Combined_Dataset/MSD/Task10_Colon/imagesTr/colon_134.nii.gz",
] # These volumes don't have enough slices to be support (their num slices < args.support_size)

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
        
        self.root = args.data_path
        self.subset = mode
        self.task_list = glob.glob(f"{args.data_path}/*")
        self.train_volume_list = []
        self.train_last_idx = {}
        self.test_volume_list = []
        self.test_last_idx = {}
        
        for task_path in self.task_list:
            task = task_path.split("/")[-1]
            if not task.startswith(args.task):
                continue
            train_volumes_in_task = glob.glob(f"{task_path}/imagesTr/*")
            test_volumes_in_task = glob.glob(f"{task_path}/imagesTs/*")
                
            self.train_volume_list.extend(train_volumes_in_task)
            self.train_last_idx[task] = len(self.train_volume_list)
            
            self.test_volume_list.extend(test_volumes_in_task)
            self.test_last_idx[task] = len(self.test_volume_list)
        
        self.image_size = args.image_size
        self.num_support = args.num_support
        self.max_slices = args.video_length
        
    def __len__(self):
        if self.subset == "train":
            return len(self.train_volume_list)
        return len(self.test_volume_list)
    
    def __getitem__(self, index):
        last_idx = self.train_last_idx if self.subset == "train" else self.test_last_idx
        volume_list = self.train_volume_list if self.subset == "train" else self.test_volume_list

        name = ""
        support_start_idx = 0
        support_end_idx = 0
        for task, last in last_idx.items():
            name = task
            support_start_idx = support_end_idx
            support_end_idx = last
            
            if last >= index:
                break
            
        if self.subset == "train":
            support_index = random.choice(
                [i for i in range(support_start_idx, support_end_idx) if \
                    i != index and volume_list[i] not in SUPPORT_EXCLUDE]
            )
        else:
            support_index = random.choice([i for i in range(support_start_idx, support_end_idx) if volume_list[i] not in SUPPORT_EXCLUDE])

        image_path = volume_list[index]
        label_path = volume_list[index].replace("image", "label")

        support_image_path = volume_list[support_index]
        support_label_path = volume_list[support_index].replace("image", "label")
        
        image_3d, data_seg_3d = self.load_image_label(
            image_path,
            label_path,
            max_slices=self.max_slices if self.subset == "train" else -1
        )
        support_image_3d, support_data_seg_3d = self.load_image_label(
            support_image_path,
            support_label_path,
            max_slices=self.num_support
        )
        
        output_dict ={
            "image": image_3d, "label": data_seg_3d,
            "support_image": support_image_3d, "support_label": support_data_seg_3d,
            "name": name
        }
        
        return output_dict

    def load_image_label(self, image_path, label_path, max_slices=16):
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