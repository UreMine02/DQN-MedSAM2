# import os
# import numpy as np
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# from func_3d.utils import random_click, generate_bbox
# from collections import defaultdict

# class BTCV(Dataset):
#     def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training', prompt='click', seed=None, variation=0):
#         # Set the data list for training
#         self.support_instance = args.support_instance
#         self.selected_scans = ['img0003', 'img0022', 'img0034']  # List of selected scans for multiple CT scan support
#         self.name_list = sorted([image for image in os.listdir(os.path.join(data_path, mode, 'image')) if image not in self.selected_scans])
        
#         # Set the basic information of the dataset
#         self.data_path = data_path
#         self.mode = mode
#         self.prompt = prompt
#         self.img_size = args.image_size
#         self.transform = transform
#         self.transform_msk = transform_msk
#         self.seed = seed
#         self.variation = variation
#         self.video_length = args.video_length
#         self.newsize = (self.img_size, self.img_size)
#         self.num_support = args.num_support

#     def __len__(self):
#         return len(self.name_list)

#     def _load_volume(self, img_path, mask_path, scan_type="QUERY"):
#         """Load a single CT scan volume with detailed debug information on slice distribution."""
#         num_frame = len(os.listdir(mask_path))
#         data_seg_3d = torch.zeros(num_frame, *self.newsize)
#         image_3d = torch.zeros((num_frame, 3, *self.newsize))

#         # Track slice distributions before and after cropping
#         slice_distribution_before_merge = {4: 0, 12: 0, 13: 0}
#         slice_distribution_after_merge = {4: 0, 12: 0}

#         for frame_index in range(num_frame):
#             # Load and preprocess mask
#             mask = Image.fromarray(np.load(os.path.join(mask_path, f'{frame_index}.npy')))
#             mask = mask.resize(self.newsize, Image.NEAREST)
#             data_seg_3d[frame_index, ...] = torch.tensor(np.array(mask))

#             # Load and preprocess image
#             img = Image.open(os.path.join(img_path, f'{frame_index}.jpg')).convert('RGB')
#             img = img.resize(self.newsize)
#             img = torch.tensor(np.array(img)).permute(2, 0, 1) 
#             image_3d[frame_index, :, :, :] = img

#             # Update slice distribution before cropping
#             unique_classes = torch.unique(data_seg_3d[frame_index])
#             for cls in unique_classes:
#                 if cls.item() in [4, 12, 13]:
#                     slice_distribution_before_merge[int(cls.item())] += 1

#         print(f"[{scan_type} - BEFORE MERGE] Scan: {os.path.basename(img_path)}")
#         print(f"  Class 4: {slice_distribution_before_merge[4]} slices")
#         print(f"  Class 12: {slice_distribution_before_merge[12]} slices")
#         print(f"  Class 13: {slice_distribution_before_merge[13]} slices")

#         # Merge Class 13 into Class 12
#         data_seg_3d[data_seg_3d == 13] = 12
#         # Crop to slices with valid masks (keep Class 0 as it is the background)
#         for i in range(data_seg_3d.shape[0]):
#             if torch.sum(data_seg_3d[i, ...]) > 0:  # Check for any non-zero mask
#                 data_seg_3d = data_seg_3d[i:, ...]
#                 image_3d = image_3d[i:, ...]
#                 break
#         starting_frame = i
#         for j in reversed(range(data_seg_3d.shape[0])):
#             if torch.sum(data_seg_3d[j, ...]) > 0:  # Check for any non-zero mask
#                 data_seg_3d = data_seg_3d[:j+1, ...]
#                 image_3d = image_3d[:j+1, ...]
#                 break
#         end_frame = j + 1
#         # Update slice distribution after cropping
#         for slice_idx in range(data_seg_3d.shape[0]):
#             unique_classes = torch.unique(data_seg_3d[slice_idx])
#             for cls in unique_classes:
#                 if cls.item() in [4, 12]:
#                     slice_distribution_after_merge[int(cls.item())] += 1

#         print(f"[{scan_type} - AFTER MERGE & CROPPING] Scan: {os.path.basename(img_path)}")
#         print(f"  Class 4: {slice_distribution_after_merge[4]} slices")
#         print(f"  Class 12 (after merge): {slice_distribution_after_merge[12]} slices")


#         return data_seg_3d, image_3d


#     def __getitem__(self, index):
#         """Get the images"""
#         name = self.name_list[index]
#         img_path = os.path.join(self.data_path, self.mode, 'image', name)
#         mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
#         # Initialize variables to track slice distribution
#         slice_distribution_before_merge = {4: 0, 12: 0, 13: 0}
#         slice_distribution_after_merge = {4: 0, 12: 0}
#         # Load query data
#         # data_seg_3d, image_3d = self._load_volume(img_path, mask_path, scan_type="QUERY")
#         data_seg_3d_shape = np.load(mask_path + '/0.npy').shape
#         num_frame = len(os.listdir(mask_path))
#         data_seg_3d = torch.zeros(num_frame, *self.newsize)
#         image_3d = torch.zeros((num_frame, 3, *self.newsize))
#         for frame_index in range(num_frame):
#             mask = Image.fromarray(np.load(os.path.join(mask_path, f'{frame_index}.npy')))
#             mask = mask.resize(self.newsize, Image.NEAREST)
#             data_seg_3d[frame_index, ...] = torch.tensor(np.array(mask))
#             img = Image.open(os.path.join(img_path, f'{frame_index}.jpg')).convert('RGB')
#             img = img.resize(self.newsize)
#             img = torch.tensor(np.array(img)).permute(2, 0, 1)
#             image_3d[frame_index, :, :, :] = img
#         # Merge class 13 into 12
#         data_seg_3d[data_seg_3d == 13] = 12
#         # Crop to slices with valid masks
#         for i in range(data_seg_3d.shape[0]):
#             if torch.sum(data_seg_3d[i, ...]) > 0:
#                 data_seg_3d = data_seg_3d[i:, ...]
#                 image_3d = image_3d[i:, ...]
#                 break
#         starting_frame = i
#         for j in reversed(range(data_seg_3d.shape[0])):
#             if torch.sum(data_seg_3d[j, ...]) > 0:
#                 data_seg_3d = data_seg_3d[:j+1, ...]
#                 image_3d = image_3d[:j+1, ...]
#                 break
#         end_frame = j + 1
#         # Support set (from multiple selected CT scans)
#         support_images = []
#         support_masks = []
#         slice_distribution_support_before_merge = defaultdict(int)
#         slice_distribution_support_after_merge = defaultdict(int)
#         for scan_name in self.selected_scans:  # Limit to num_scans
#             support_img_path = os.path.join(self.data_path, "Training", 'image', scan_name)
#             support_mask_path = os.path.join(self.data_path, "Training", 'mask', scan_name)
#             # support_data_seg_3d, support_image_3d = self._load_volume(support_img_path, support_mask_path, scan_type="SUPPORT")
#             support_num_frame = len(os.listdir(support_mask_path))
#             support_data_seg_3d = torch.zeros(support_num_frame, *self.newsize)
#             support_image_3d = torch.zeros((support_num_frame, 3, *self.newsize))
#             for frame_index in range(support_num_frame):
#                 support_mask = Image.fromarray(np.load(os.path.join(support_mask_path, f'{frame_index}.npy')))
#                 support_mask = support_mask.resize(self.newsize, Image.NEAREST)
#                 support_data_seg_3d[frame_index, ...] = torch.tensor(np.array(support_mask))

#                 support_img = Image.open(os.path.join(support_img_path, f'{frame_index}.jpg')).convert('RGB')
#                 support_img = support_img.resize(self.newsize)
#                 support_img = torch.tensor(np.array(support_img)).permute(2, 0, 1)
#                 support_image_3d[frame_index, :, :, :] = support_img

#                 # Update slice distribution before merging
#                 unique_classes = torch.unique(support_data_seg_3d[frame_index])
#                 for cls in unique_classes:
#                     if cls.item() in [4, 12, 13]:
#                         slice_distribution_support_before_merge[int(cls.item())] += 1
#             # Merge class 13 into 12
#             support_data_seg_3d[support_data_seg_3d == 13] = 12
#             # Crop to slices with valid masks
#             for i in range(support_data_seg_3d.shape[0]):
#                 if torch.sum(support_data_seg_3d[i, ...]) > 0:
#                     support_data_seg_3d = support_data_seg_3d[i:, ...]
#                     support_image_3d = support_image_3d[i:, ...]
#                     break
#             support_starting_frame = i
#             for j in reversed(range(support_data_seg_3d.shape[0])):
#                 if torch.sum(support_data_seg_3d[j, ...]) > 0:
#                     support_data_seg_3d = support_data_seg_3d[:j+1, ...]
#                     support_image_3d = support_image_3d[:j+1, ...]
#                     break
#             support_end_frame = j + 1
#             # support_num_frame = support_data_seg_3d.shape[0]
#             # Update slice distribution after merging and cropping
#             for frame_index in range(support_data_seg_3d.shape[0]):
#                 unique_classes = torch.unique(support_data_seg_3d[frame_index])
#                 for cls in unique_classes:
#                     if cls.item() in [4, 12]:
#                         slice_distribution_support_after_merge[int(cls.item())] += 1

#             support_images.append(support_image_3d)
#             support_masks.append(support_data_seg_3d)
#         # Concatenate support data
#         support_image_3d = torch.cat(support_images, dim=0)
#         support_data_seg_3d = torch.cat(support_masks, dim=0)
        
#         output_dict = {
#             "image": image_3d, 
#             "label": data_seg_3d,
#             "support_image": support_image_3d, 
#             "support_label": support_data_seg_3d,
#             "name": name
#     }
#         return output_dict

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

# class Combined(Dataset):
#     def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', seed=None, variation=0):

#         # Set the data list for training
#         self.support_instance = args.support_instance
#         if mode == "Training":
#             self.dir = "imagesTr"
#         else:
#             self.dir = "imagesTs"

#         self.dataset = []
#         volume_id = 0
#         self.dataset_list = os.listdir(data_path)
        
#         for dataset in self.dataset_list:
#             if dataset.startswith("."):
#                 continue
#             data = Data(dataset)
#             for task in os.listdir(os.path.join(data_path, dataset)):
#                 if task.startswith("."):
#                     continue
#                 new_task = Task(task)
#                 data.add_new_task(new_task)
#                 for volume in os.listdir(os.path.join(data_path, dataset, task, self.dir)):
#                     if volume.startswith("."):
#                         continue                    
#                     new_volume_name = volume
#                     new_volume_path = os.path.join(data_path, dataset, task, self.dir, volume)
#                     new_volume = Volume(new_volume_name, volume_id, new_volume_path)
#                     volume_id += 1
#                     new_task.add_new_volume(new_volume)
#                 new_task.get_support_stance()
            
#             self.dataset.append(data)

#         self.name_list = [volume for dataset in self.dataset for task in dataset.task for volume in task.volume]
        
#         # Set the basic information of the dataset
#         self.data_path = data_path
#         self.mode = mode
#         self.prompt = prompt
#         self.img_size = args.image_size
#         self.transform = transform
#         self.transform_msk = transform_msk
#         self.seed = seed
#         self.variation = variation
#         if mode == 'Training':
#             self.video_length = args.video_length
#         else:
#             self.video_length = None
        
#         self.newsize = (self.img_size, self.img_size)

#         self.num_support = args.num_support
        
#     def __len__(self):
#         return len(self.name_list)

#     def __getitem__(self, index):

#         """Get the images"""
#         name = self.name_list[index]

#         img_path = name.train_path
#         mask_path = name.label_path

#         image = nib.load(img_path, mmap=True)
#         data_seg_3d = nib.load(mask_path, mmap=True)

#         image_header = image.header
#         data_seg_3d_header = data_seg_3d.header

#         if len(image_header.get_data_shape()) == 4:
#             num_frame = image_header.get_data_shape()[-2]
#         else:
#             num_frame = image_header.get_data_shape()[-1]

#         image_chunks, mask_chunks = [], []

#         for current_chunk in range(0, num_frame, 10):
#             if len(image_header.get_data_shape()) == 4:
#                 if image.shape[-1] > 2:
#                     image_chunk = image.slicer[:, :, current_chunk:current_chunk+10, 2].get_fdata()
#                 else:
#                     image_chunk = image.slicer[:, :, current_chunk:current_chunk+10, 0].get_fdata()
#             else:
#                 image_chunk = image.slicer[:, :, current_chunk:current_chunk+10].get_fdata()
#             mask_chunk = data_seg_3d.slicer[:, :, current_chunk:current_chunk+10].get_fdata()

#             image_chunk, mask_chunk = remove_negative_samples(image_chunk, mask_chunk)

#             image_chunks.append(image_chunk)
#             mask_chunks.append(mask_chunk)

#             current_chunk += 10
        
#         image_3d = np.concat(image_chunks, axis=2)
#         data_seg_3d =np.concat(mask_chunks, axis=2)

#         image_3d = normalization(image_3d)
#         image_3d = torch.rot90(torch.tensor(image_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
#         data_seg_3d = torch.rot90(torch.tensor(data_seg_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

#         image_3d = F.interpolate(image_3d, size=(image_3d.shape[2], self.img_size, self.img_size), mode='trilinear', align_corners=False)
#         data_seg_3d = F.interpolate(data_seg_3d, size=(data_seg_3d.shape[2], self.img_size, self.img_size), mode='nearest')
#         image_3d = image_3d.squeeze(0).repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
#         data_seg_3d = data_seg_3d.squeeze(0).squeeze(0)

#         support_img_path = name.support_volume.train_path
#         support_mask_path = name.support_volume.label_path

#         support_image = nib.load(support_img_path, mmap=True)
#         support_data_seg_3d = nib.load(support_mask_path, mmap=True)

#         support_image_header = support_image.header
#         support_data_seg_3d_header = support_data_seg_3d.header

#         if len(support_image_header.get_data_shape()) == 4:
#             support_num_frame = support_image_header.get_data_shape()[-2]
#         else:
#             support_num_frame = support_image_header.get_data_shape()[-1]
        
#         support_image_chunks, support_mask_chunks = [], []

#         for current_chunk in range(0, support_num_frame, 10):
#             if len(support_image_header.get_data_shape()) == 4:
#                 if support_image.shape[-1] > 2:
#                     support_image_chunk = support_image.slicer[:, :, current_chunk:current_chunk+10, 2].get_fdata()
#                 else:
#                     support_image_chunk = support_image.slicer[:, :, current_chunk:current_chunk+10, 0].get_fdata()
#             else:
#                 support_image_chunk = support_image.slicer[:, :, current_chunk:current_chunk+10].get_fdata()
#             support_mask_chunk = support_data_seg_3d.slicer[:, :, current_chunk:current_chunk+10].get_fdata()

#             support_image_chunk, support_mask_chunk = remove_negative_samples(support_image_chunk, support_mask_chunk)

#             support_image_chunks.append(support_image_chunk)
#             support_mask_chunks.append(support_mask_chunk)

#             current_chunk += 10

#         support_image_3d = np.concat(support_image_chunks, axis=2)
#         support_data_seg_3d =np.concat(support_mask_chunks, axis=2) 

#         support_image_3d = normalization(support_image_3d)
#         support_image_3d = torch.rot90(torch.tensor(support_image_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
#         support_data_seg_3d = torch.rot90(torch.tensor(support_data_seg_3d)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

#         support_image_3d = F.interpolate(support_image_3d, size=(support_image_3d.shape[2], self.img_size, self.img_size), mode='trilinear', align_corners=False)
#         support_data_seg_3d = F.interpolate(support_data_seg_3d, size=(support_data_seg_3d.shape[2], self.img_size, self.img_size), mode='nearest')
#         support_image_3d = support_image_3d.squeeze(0).repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
#         support_data_seg_3d = support_data_seg_3d.squeeze(0).squeeze(0)

#         output_dict ={"image": image_3d, "label": data_seg_3d,
#                 "support_image": support_image_3d, "support_label": support_data_seg_3d,
#                 "name": "name"}
        
#         return output_dict
        
class BTCV(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='Training',prompt = 'click', seed=None, variation=0):
        self.mode = mode
        self.data_path = data_path
        self.img_size = args.image_size
        self.transform = transform or T.ToTensor()
        self.transform_msk = transform_msk or T.ToTensor()
        self.name_list = []

        img_folder = "imagesTr" if mode == "Training" else "imagesTs"
        msk_folder = "labelsTr" if mode == "Training" else "labelsTs"

        for dataset_name in os.listdir(data_path):
            if dataset_name.startswith("."):
                continue
            dataset_path = os.path.join(data_path, dataset_name)
            for task_name in os.listdir(dataset_path):
                task_path = os.path.join(dataset_path, task_name)
                img_dir = os.path.join(task_path, img_folder)
                msk_dir = os.path.join(task_path, msk_folder)

                if not os.path.exists(img_dir) or not os.path.exists(msk_dir):
                    continue

                for fname in os.listdir(img_dir):
                    if not fname.endswith(".png"):
                        continue
                    img_path = os.path.join(img_dir, fname)
                    msk_path = os.path.join(msk_dir, fname)
                    if os.path.exists(msk_path):
                        self.name_list.append({
                            "img_path": img_path,
                            "mask_path": msk_path
                        })

        self.resize = T.Resize((self.img_size, self.img_size))
        print(f"[INFO] Found {len(self.name_list)} samples in {self.mode} set.")

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        item = self.name_list[idx]

        img = Image.open(item["img_path"]).convert("L")
        msk = Image.open(item["mask_path"]).convert("L")

        img = self.resize(img)
        msk = self.resize(msk)

        img_tensor = self.transform(img)
        msk_tensor = self.transform_msk(msk)
        # print('img_tensor.shape[0]',img_tensor.shape)  #[1,1024,1024]
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1) #[3,1024,1024]

        # support = chính ảnh (bạn có thể thay bằng random khác nếu cần)
        output = {
            "image": img_tensor[None], #[1,3,1024,1024]
            "label": msk_tensor, #[1,1024,1024]
            "support_image": img_tensor.clone()[None], #[1,3,1024,1024]
            "support_label": msk_tensor.clone(), #[1,1024,1024]
            "name": os.path.basename(item["img_path"])
        }
        # print("output: ")
        #image torch.Size([1, 3, 1024, 1024]) label torch.Size([1, 1, 1024, 1024]) support_image torch.Size([1, 3, 1024, 1024])support_label torch.Size([1, 1, 1024, 1024])
        # for k,v in output.items():
        #     print(k, v.shape)
        return output
