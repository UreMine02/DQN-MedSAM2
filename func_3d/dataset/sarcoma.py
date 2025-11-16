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


class Sarcoma(Dataset):
    def __init__(self, args, subset="train"):
        self.root = args.data_path
        self.subset = subset

        mass_dir = os.path.join(self.root, "16_NIFTI_Soft-tissue-Sarcoma-Mass/MRI")
        edema_dir = os.path.join(self.root, "17_NIFTI_Soft-tissue-Sarcoma-Edema/MRI")
        
        self.train_split = [
            "STS_001",  "STS_008",  "STS_015",  "STS_021",  "STS_028",  "STS_033",  "STS_039",  "STS_046",
            "STS_003",  "STS_010",  "STS_016",  "STS_022",  "STS_029",  "STS_035",  "STS_040",  "STS_047",
            "STS_004",  "STS_011",  "STS_017",  "STS_023",  "STS_030",  "STS_036",  "STS_041",  "STS_048",
            "STS_006",  "STS_012",  "STS_018",  "STS_024",  "STS_031",  "STS_037",  "STS_042",  "STS_049",
            "STS_007",  "STS_013",  "STS_019",  "STS_027",  "STS_032",  "STS_038",  "STS_043",  "STS_051",
        ]

        self.test_split = [
            "STS_002",  "STS_009",  "STS_020",  "STS_026",  "STS_044",  "STS_050",
            "STS_005",  "STS_014",  "STS_025",  "STS_034",  "STS_045",
        ]

        self.train_mass_list = [os.path.join(mass_dir, case) for case in os.listdir(mass_dir) if case in self.train_split]
        self.train_edema_list = [os.path.join(edema_dir, case) for case in os.listdir(edema_dir) if case in self.train_split]

        self.test_mass_list = [os.path.join(mass_dir, case) for case in os.listdir(mass_dir) if case in self.test_split]
        self.test_edema_list = [os.path.join(edema_dir, case) for case in os.listdir(edema_dir) if case in self.test_split]
        
        self.image_size = args.image_size
        self.max_slices = 16
        
    def __len__(self):
        if self.subset == "train":
            return len(self.train_mass_list + self.train_edema_list)
        return len(self.test_mass_list + self.test_edema_list)
    
    def __getitem__(self, index):
        mass_list = self.train_mass_list if self.subset == "train" else self.test_mass_list
        edema_list = self.train_edema_list if self.subset == "train" else self.test_edema_list

        if index < len(mass_list):
            path_list = mass_list
            support_list = self.train_mass_list
            name = "Mass"
        else:
            path_list = edema_list
            support_list = self.train_edema_list
            index = index - len(mass_list)
            name = "Edema"

        if self.subset == "train":
            support_index = random.choice([i for i in range(len(support_list)) if i != index])
        else:
            support_index = random.choice([i for i in range(len(support_list))])

        image_path = os.path.join(path_list[index], "img", "image.nii.gz")
        label_path = os.path.join(path_list[index], "label", f"mask_GTV_{name}.nii.gz")

        support_image_path = os.path.join(support_list[support_index], "img", "image.nii.gz")
        support_label_path = os.path.join(support_list[support_index], "label", f"mask_GTV_{name}.nii.gz")
        
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
            "name": name
        }
        
        return output_dict

    def load_image_label(self, image_path, label_path, max_slices=16):
        image_3d = nib.load(image_path, mmap=True)
        data_seg_3d = nib.load(label_path, mmap=True)
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

        # Convert RGB value to class index
        data_seg_3d = data_seg_3d / 255

        return image_3d, data_seg_3d