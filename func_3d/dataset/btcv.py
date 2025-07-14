import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from func_3d.utils import random_click, generate_bbox
from collections import defaultdict

class BTCV(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None, variation=0):
        # Set the data list for training
        self.support_instance = args.support_instance
        self.selected_scans = ['img0003', 'img0022', 'img0034']  # List of selected scans for multiple CT scan support
        self.name_list = sorted([image for image in os.listdir(os.path.join(data_path, mode, 'image')) if image not in self.selected_scans])
        
        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        self.video_length = args.video_length
        self.newsize = (self.img_size, self.img_size)
        self.num_support = args.num_support

    def __len__(self):
        return len(self.name_list)

    def _load_volume(self, img_path, mask_path, scan_type="QUERY"):
        """Load a single CT scan volume with detailed debug information on slice distribution."""
        num_frame = len(os.listdir(mask_path))
        data_seg_3d = torch.zeros(num_frame, *self.newsize)
        image_3d = torch.zeros((num_frame, 3, *self.newsize))

        # Track slice distributions before and after cropping
        slice_distribution_before_merge = {4: 0, 12: 0, 13: 0}
        slice_distribution_after_merge = {4: 0, 12: 0}

        for frame_index in range(num_frame):
            # Load and preprocess mask
            mask = Image.fromarray(np.load(os.path.join(mask_path, f'{frame_index}.npy')))
            mask = mask.resize(self.newsize, Image.NEAREST)
            data_seg_3d[frame_index, ...] = torch.tensor(np.array(mask))

            # Load and preprocess image
            img = Image.open(os.path.join(img_path, f'{frame_index}.jpg')).convert('RGB')
            img = img.resize(self.newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1) 
            image_3d[frame_index, :, :, :] = img

            # Update slice distribution before cropping
            unique_classes = torch.unique(data_seg_3d[frame_index])
            for cls in unique_classes:
                if cls.item() in [4, 12, 13]:
                    slice_distribution_before_merge[int(cls.item())] += 1

        print(f"[{scan_type} - BEFORE MERGE] Scan: {os.path.basename(img_path)}")
        print(f"  Class 4: {slice_distribution_before_merge[4]} slices")
        print(f"  Class 12: {slice_distribution_before_merge[12]} slices")
        print(f"  Class 13: {slice_distribution_before_merge[13]} slices")

        # Merge Class 13 into Class 12
        data_seg_3d[data_seg_3d == 13] = 12
        # Crop to slices with valid masks (keep Class 0 as it is the background)
        for i in range(data_seg_3d.shape[0]):
            if torch.sum(data_seg_3d[i, ...]) > 0:  # Check for any non-zero mask
                data_seg_3d = data_seg_3d[i:, ...]
                image_3d = image_3d[i:, ...]
                break
        starting_frame = i
        for j in reversed(range(data_seg_3d.shape[0])):
            if torch.sum(data_seg_3d[j, ...]) > 0:  # Check for any non-zero mask
                data_seg_3d = data_seg_3d[:j+1, ...]
                image_3d = image_3d[:j+1, ...]
                break
        end_frame = j + 1
        # Update slice distribution after cropping
        for slice_idx in range(data_seg_3d.shape[0]):
            unique_classes = torch.unique(data_seg_3d[slice_idx])
            for cls in unique_classes:
                if cls.item() in [4, 12]:
                    slice_distribution_after_merge[int(cls.item())] += 1

        print(f"[{scan_type} - AFTER MERGE & CROPPING] Scan: {os.path.basename(img_path)}")
        print(f"  Class 4: {slice_distribution_after_merge[4]} slices")
        print(f"  Class 12 (after merge): {slice_distribution_after_merge[12]} slices")


        return data_seg_3d, image_3d


    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
        # Initialize variables to track slice distribution
        slice_distribution_before_merge = {4: 0, 12: 0, 13: 0}
        slice_distribution_after_merge = {4: 0, 12: 0}
        # Load query data
        # data_seg_3d, image_3d = self._load_volume(img_path, mask_path, scan_type="QUERY")
        data_seg_3d_shape = np.load(mask_path + '/0.npy').shape
        num_frame = len(os.listdir(mask_path))
        data_seg_3d = torch.zeros(num_frame, *self.newsize)
        image_3d = torch.zeros((num_frame, 3, *self.newsize))
        for frame_index in range(num_frame):
            mask = Image.fromarray(np.load(os.path.join(mask_path, f'{frame_index}.npy')))
            mask = mask.resize(self.newsize, Image.NEAREST)
            data_seg_3d[frame_index, ...] = torch.tensor(np.array(mask))
            img = Image.open(os.path.join(img_path, f'{frame_index}.jpg')).convert('RGB')
            img = img.resize(self.newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)
            image_3d[frame_index, :, :, :] = img
        # Merge class 13 into 12
        data_seg_3d[data_seg_3d == 13] = 12
        # Crop to slices with valid masks
        for i in range(data_seg_3d.shape[0]):
            if torch.sum(data_seg_3d[i, ...]) > 0:
                data_seg_3d = data_seg_3d[i:, ...]
                image_3d = image_3d[i:, ...]
                break
        starting_frame = i
        for j in reversed(range(data_seg_3d.shape[0])):
            if torch.sum(data_seg_3d[j, ...]) > 0:
                data_seg_3d = data_seg_3d[:j+1, ...]
                image_3d = image_3d[:j+1, ...]
                break
        end_frame = j + 1
        # Support set (from multiple selected CT scans)
        support_images = []
        support_masks = []
        slice_distribution_support_before_merge = defaultdict(int)
        slice_distribution_support_after_merge = defaultdict(int)
        for scan_name in self.selected_scans:  # Limit to num_scans
            support_img_path = os.path.join(self.data_path, "Training", 'image', scan_name)
            support_mask_path = os.path.join(self.data_path, "Training", 'mask', scan_name)
            # support_data_seg_3d, support_image_3d = self._load_volume(support_img_path, support_mask_path, scan_type="SUPPORT")
            support_num_frame = len(os.listdir(support_mask_path))
            support_data_seg_3d = torch.zeros(support_num_frame, *self.newsize)
            support_image_3d = torch.zeros((support_num_frame, 3, *self.newsize))
            for frame_index in range(support_num_frame):
                support_mask = Image.fromarray(np.load(os.path.join(support_mask_path, f'{frame_index}.npy')))
                support_mask = support_mask.resize(self.newsize, Image.NEAREST)
                support_data_seg_3d[frame_index, ...] = torch.tensor(np.array(support_mask))

                support_img = Image.open(os.path.join(support_img_path, f'{frame_index}.jpg')).convert('RGB')
                support_img = support_img.resize(self.newsize)
                support_img = torch.tensor(np.array(support_img)).permute(2, 0, 1)
                support_image_3d[frame_index, :, :, :] = support_img

                # Update slice distribution before merging
                unique_classes = torch.unique(support_data_seg_3d[frame_index])
                for cls in unique_classes:
                    if cls.item() in [4, 12, 13]:
                        slice_distribution_support_before_merge[int(cls.item())] += 1
            # Merge class 13 into 12
            support_data_seg_3d[support_data_seg_3d == 13] = 12
            # Crop to slices with valid masks
            for i in range(support_data_seg_3d.shape[0]):
                if torch.sum(support_data_seg_3d[i, ...]) > 0:
                    support_data_seg_3d = support_data_seg_3d[i:, ...]
                    support_image_3d = support_image_3d[i:, ...]
                    break
            support_starting_frame = i
            for j in reversed(range(support_data_seg_3d.shape[0])):
                if torch.sum(support_data_seg_3d[j, ...]) > 0:
                    support_data_seg_3d = support_data_seg_3d[:j+1, ...]
                    support_image_3d = support_image_3d[:j+1, ...]
                    break
            support_end_frame = j + 1
            # support_num_frame = support_data_seg_3d.shape[0]
            # Update slice distribution after merging and cropping
            for frame_index in range(support_data_seg_3d.shape[0]):
                unique_classes = torch.unique(support_data_seg_3d[frame_index])
                for cls in unique_classes:
                    if cls.item() in [4, 12]:
                        slice_distribution_support_after_merge[int(cls.item())] += 1

            support_images.append(support_image_3d)
            support_masks.append(support_data_seg_3d)
        # Concatenate support data
        support_image_3d = torch.cat(support_images, dim=0)
        support_data_seg_3d = torch.cat(support_masks, dim=0)
        
        output_dict = {
            "image": image_3d, 
            "label": data_seg_3d,
            "support_image": support_image_3d, 
            "support_label": support_data_seg_3d,
            "name": name
    }
        return output_dict
