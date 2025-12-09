import os
import glob
import random
import nibabel as nib
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def normalization(image):
    image_min = np.min(image)
    image_max = np.max(image)
    image = ((image - image_min)/(image_max-image_min))*255
    return image

def remove_negative_samples(image_tensor, mask_tensor):
    pos_slices = np.sum(mask_tensor, axis=(0,1)) > 0
    return image_tensor[:, :, pos_slices], mask_tensor[:, :, pos_slices]


class BTCV(Dataset):
    def __init__(self, args, subset="train"):
        self.root = args.data_path
        self.subset = subset

        self.train_list = glob.glob(os.path.join(self.root, "imagesTr", "*"))
        self.test_list = glob.glob(os.path.join(self.root, "imagesTs", "*"))
        
        self.image_size = args.image_size
        self.max_slices = 16
        
    def __len__(self):
        if self.subset == "train":
            return len(self.train_list)
        return len(self.test_list)
    
    def __getitem__(self, index):
        volume_list = self.train_list if self.subset == "train" else self.test_list

        if self.subset == "train":
            support_index = random.choice([i for i in range(len(volume_list)) if i != index])
        else:
            support_index = random.choice([i for i in range(len(volume_list))])

        image_path = volume_list[index]
        label_path = image_path.replace("image", "label")

        support_image_path = volume_list[support_index]
        support_label_path = support_image_path.replace("image", "label")
        
        image_3d, data_seg_3d = self.load_image_label(
            image_path,
            label_path,
            max_slices=self.max_slices if self.subset == "train" else -1
        )
        support_image_3d, support_data_seg_3d = self.load_image_label(
            support_image_path,
            support_label_path,
            max_slices=self.max_slices
        )
        
        output_dict ={
            "image": image_3d, "label": data_seg_3d,
            "support_image": support_image_3d, "support_label": support_data_seg_3d,
            "name": image_path.split("/")[-1]
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