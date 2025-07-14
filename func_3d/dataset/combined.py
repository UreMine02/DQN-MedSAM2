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

def remove_negative_samples(image_tensor, mask_tensor):
    if np.sum(mask_tensor) == 0:
        return image_tensor[..., 0:0], mask_tensor[..., 0:0]
    
    for i in range(mask_tensor.shape[-1]):
        if np.sum(mask_tensor[..., i]) > 0:
            mask_tensor = mask_tensor[..., i:]
            image_tensor = image_tensor[..., i:]
            break

    for j in reversed(range(mask_tensor.shape[-1])):
        if np.sum(mask_tensor[..., j]) > 0:
            mask_tensor = mask_tensor[..., :j+1]
            image_tensor = image_tensor[..., :j+1]
            break

    return image_tensor, mask_tensor

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
            if dataset.startswith("."):
                continue
            data = Data(dataset)
            for task in os.listdir(os.path.join(data_path, dataset)):
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

    def __getitem__(self, index):

        """Get the images"""
        name = self.name_list[index]

        img_path = name.train_path
        mask_path = name.label_path

        image = nib.load(img_path, mmap=True)
        data_seg_3d = nib.load(mask_path, mmap=True)

        image_header = image.header
        data_seg_3d_header = data_seg_3d.header

        if len(image_header.get_data_shape()) == 4:
            num_frame = image_header.get_data_shape()[-2]
        else:
            num_frame = image_header.get_data_shape()[-1]

        image_chunks, mask_chunks = [], []

        for current_chunk in range(0, num_frame, 10):
            if len(image_header.get_data_shape()) == 4:
                if image.shape[-1] > 2:
                    image_chunk = image.slicer[:, :, current_chunk:current_chunk+10, 2].get_fdata()
                else:
                    image_chunk = image.slicer[:, :, current_chunk:current_chunk+10, 0].get_fdata()
            else:
                image_chunk = image.slicer[:, :, current_chunk:current_chunk+10].get_fdata()
            mask_chunk = data_seg_3d.slicer[:, :, current_chunk:current_chunk+10].get_fdata()

            image_chunk, mask_chunk = remove_negative_samples(image_chunk, mask_chunk)

            image_chunks.append(image_chunk)
            mask_chunks.append(mask_chunk)

            current_chunk += 10
        
        image_3d = np.concat(image_chunks, axis=2)
        data_seg_3d =np.concat(mask_chunks, axis=2)

        image_3d = normalization(image_3d)
        image_3d = torch.rot90(torch.tensor(image_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        data_seg_3d = torch.rot90(torch.tensor(data_seg_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

        image_3d = F.interpolate(image_3d, size=(image_3d.shape[2], self.img_size, self.img_size), mode='trilinear', align_corners=False)
        data_seg_3d = F.interpolate(data_seg_3d, size=(data_seg_3d.shape[2], self.img_size, self.img_size), mode='nearest')
        image_3d = image_3d.squeeze(0).repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
        data_seg_3d = data_seg_3d.squeeze(0).squeeze(0)

        support_img_path = name.support_volume.train_path
        support_mask_path = name.support_volume.label_path

        support_image = nib.load(support_img_path, mmap=True)
        support_data_seg_3d = nib.load(support_mask_path, mmap=True)

        support_image_header = support_image.header
        support_data_seg_3d_header = support_data_seg_3d.header

        if len(support_image_header.get_data_shape()) == 4:
            support_num_frame = support_image_header.get_data_shape()[-2]
        else:
            support_num_frame = support_image_header.get_data_shape()[-1]
        
        support_image_chunks, support_mask_chunks = [], []

        for current_chunk in range(0, support_num_frame, 10):
            if len(support_image_header.get_data_shape()) == 4:
                if support_image.shape[-1] > 2:
                    support_image_chunk = support_image.slicer[:, :, current_chunk:current_chunk+10, 2].get_fdata()
                else:
                    support_image_chunk = support_image.slicer[:, :, current_chunk:current_chunk+10, 0].get_fdata()
            else:
                support_image_chunk = support_image.slicer[:, :, current_chunk:current_chunk+10].get_fdata()
            support_mask_chunk = support_data_seg_3d.slicer[:, :, current_chunk:current_chunk+10].get_fdata()

            support_image_chunk, support_mask_chunk = remove_negative_samples(support_image_chunk, support_mask_chunk)

            support_image_chunks.append(support_image_chunk)
            support_mask_chunks.append(support_mask_chunk)

            current_chunk += 10

        support_image_3d = np.concat(support_image_chunks, axis=2)
        support_data_seg_3d =np.concat(support_mask_chunks, axis=2) 

        support_image_3d = normalization(support_image_3d)
        support_image_3d = torch.rot90(torch.tensor(support_image_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        support_data_seg_3d = torch.rot90(torch.tensor(support_data_seg_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

        support_image_3d = F.interpolate(support_image_3d, size=(support_image_3d.shape[2], self.img_size, self.img_size), mode='trilinear', align_corners=False)
        support_data_seg_3d = F.interpolate(support_data_seg_3d, size=(support_data_seg_3d.shape[2], self.img_size, self.img_size), mode='nearest')
        support_image_3d = support_image_3d.squeeze(0).repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
        support_data_seg_3d = support_data_seg_3d.squeeze(0).squeeze(0)

        output_dict ={"image": image_3d, "label": data_seg_3d,
                "support_image": support_image_3d, "support_label": support_data_seg_3d,
                "name": "name"}
        
        return output_dict
        