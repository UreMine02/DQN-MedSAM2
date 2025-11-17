""" Dataloader for the BTCV dataset
    Yunli Qi
"""
import os
import numpy as np
import torch
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T

from func_3d.utils import random_click, generate_bbox

class Data:
    def __init__(self, dataset):
        self.dataset = dataset
        self.task = []
    
    def add_new_task(self, task):
        self.task.append(task)

class Task:
    def __init__(self, task):
        self.task = task
        self.volume = []
        self.support_instance = None

    def add_new_volume(self, volume):
        self.volume.append(volume)
    
    def get_support_stance(self):
        self.support_instance = self.volume[-1]
        self.volume = self.volume[:-1]
        for volume in self.volume:
            volume.support_volume = self.support_instance

class Volume:
    def __init__(self, volume_name, volume_id, path):
        self.volume_name = volume_name
        self.volume_id = volume_id
        self.train_path = path
        self.label_path = path.replace("imagesTr", "labelsTr").replace("imagesTs", "labelsTs")
        self.support_volume= None

def normalization(image):
    image_min = np.min(image)
    image_max = np.max(image)
    image = ((image - image_min)/(image_max-image_min))*255
    return image

def remove_negative_samples(image, mask):
    pos_slices = np.sum(mask, axis=(0,1)) > 0
    return image[:, :, pos_slices], mask[:, :, pos_slices]

class Combined(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', seed=None, variation=0):

        # Set the data list for training
        self.support_instance = args.support_instance
        if mode == "Training":
            self.dir = "imagesTr"
        else:
            self.dir = "imagesTs"

        self.dataset = []
        volume_id = 0
        self.dataset_list = os.listdir(data_path)
        
        for dataset in self.dataset_list:
            data = Data(dataset)
            for task in os.listdir(os.path.join(data_path, dataset)):
                if not task.startswith("Task01"):
                    continue
                
                if task.startswith("."):
                    continue
                new_task = Task(task)
                data.add_new_task(new_task)
                for volume in os.listdir(os.path.join(data_path, dataset, task, self.dir)):
                    if volume.startswith("."):
                        continue                    
                    new_volume_name = volume
                    new_volume_path = os.path.join(data_path, dataset, task, self.dir, volume)
                    new_volume = Volume(new_volume_name, volume_id, new_volume_path)
                    volume_id += 1
                    new_task.add_new_volume(new_volume)
                new_task.get_support_stance()
            
            self.dataset.append(data)

        self.name_list = [volume for dataset in self.dataset for task in dataset.task for volume in task.volume]
        
        print(len(self.name_list))
        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None
        
        self.newsize = (self.img_size, self.img_size)

        self.num_support = args.num_support
        
    def __len__(self):
        return len(self.name_list)

    def normalization_safe(self, image):
        """Safe normalization to avoid zero-size or flat arrays."""
        if image is None or image.size == 0:
            raise ValueError("Empty image passed to normalization_safe()")
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max == image_min:
            return np.zeros_like(image)
        return (image - image_min) / (image_max - image_min + 1e-8)

    def __getitem__(self, index):
        
        """Get the images"""
        name = self.name_list[index]

        # ====================== QUERY VOLUME ======================
        img_path = name.train_path
        mask_path = name.label_path

        support_img_path = name.support_volume.train_path
        support_mask_path = name.support_volume.label_path
        
        image_3d, data_seg_3d = self.load_image_label(img_path, mask_path)
        support_image_3d, support_data_seg_3d = self.load_image_label(support_img_path, support_mask_path)
        

        # Output dictionary
        output_dict = {
            "image": image_3d,
            "label": data_seg_3d,
            "support_image": support_image_3d,
            "support_label": support_data_seg_3d,
            "name": name.volume_name
        }

        return output_dict
    
    def load_image_label(self, image_path, label_path):
        image_3d = nib.load(image_path, mmap=True)
        data_seg_3d = nib.load(label_path, mmap=True)
        image_3d = image_3d.get_fdata()
        data_seg_3d = data_seg_3d.get_fdata()
        
        if image_3d.ndim == 4:
            if image_3d.shape[-1] == 4:
                image_3d = image_3d[:, :, :, 2]
            elif image_3d.shape[-1] == 2:
                image_3d = image_3d[:, :, :, 0]
            
        image_3d, data_seg_3d = remove_negative_samples(image_3d, data_seg_3d)
        
        max_slices = 16
        if image_3d.shape[-1] > max_slices:
            start_slice = np.random.choice(range(image_3d.shape[-1] - max_slices + 1))
            image_3d = image_3d[:, :, start_slice:start_slice+max_slices]
            data_seg_3d = data_seg_3d[:, :, start_slice:start_slice+max_slices]

        image_3d = self.normalization_safe(image_3d)
        image_3d = torch.rot90(torch.tensor(image_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        data_seg_3d = torch.rot90(torch.tensor(data_seg_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

        image_3d = F.interpolate(image_3d, size=(image_3d.shape[2], self.img_size, self.img_size), mode='trilinear', align_corners=False)
        data_seg_3d = F.interpolate(data_seg_3d, size=(data_seg_3d.shape[2], self.img_size, self.img_size), mode='nearest')
        image_3d = image_3d.squeeze(0).repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
        data_seg_3d = data_seg_3d.squeeze(0).squeeze(0)

        return image_3d, data_seg_3d
    



        
# class Combined(Dataset):
#     def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training',prompt = 'click', seed=None, variation=0):
#         self.mode = mode
#         self.data_path = data_path
#         self.img_size = args.image_size
#         self.transform = transform or T.ToTensor()
#         self.transform_msk = transform_msk or T.ToTensor()
#         self.name_list = []

#         img_folder = "imagesTr" if mode == "Training" else "imagesTs"
#         msk_folder = "labelsTr" if mode == "Training" else "labelsTs"

#         for dataset_name in os.listdir(data_path):
#             if dataset_name.startswith("."):
#                 continue
#             dataset_path = os.path.join(data_path, dataset_name)
#             for task_name in os.listdir(dataset_path):
#                 task_path = os.path.join(dataset_path, task_name)
#                 img_dir = os.path.join(task_path, img_folder)
#                 msk_dir = os.path.join(task_path, msk_folder)

#                 if not os.path.exists(img_dir) or not os.path.exists(msk_dir):
#                     continue

#                 for fname in os.listdir(img_dir):
#                     if not fname.endswith(".png"):
#                         continue
#                     img_path = os.path.join(img_dir, fname)
#                     msk_path = os.path.join(msk_dir, fname)
#                     if os.path.exists(msk_path):
#                         self.name_list.append({
#                             "img_path": img_path,
#                             "mask_path": msk_path
#                         })

#         self.resize = T.Resize((self.img_size, self.img_size))
#         print(f"[INFO] Found {len(self.name_list)} samples in {self.mode} set.")

#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, idx):
#         item = self.name_list[idx]

#         img = Image.open(item["img_path"]).convert("L")
#         msk = Image.open(item["mask_path"]).convert("L")

#         img = self.resize(img)
#         msk = self.resize(msk)

#         img_tensor = self.transform(img)
#         msk_tensor = self.transform_msk(msk)

#         if img_tensor.shape[0] == 1:
#             img_tensor = img_tensor.repeat(3, 1, 1)

#         # support = chính ảnh (bạn có thể thay bằng random khác nếu cần)
#         output = {
#             "image": img_tensor,
#             "label": msk_tensor,
#             "support_image": img_tensor.clone(),
#             "support_label": msk_tensor.clone(),
#             "name": os.path.basename(item["img_path"])
#         }

#         return output