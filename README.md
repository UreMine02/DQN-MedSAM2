## Introduction

This project aims to build a foundation medical segmentation model with in-context-learning based on the newly released SAM2 (Segment Anything Model 2). In short, in-context-learning is a paradigm where the model learns from examples (support set) provided and demonstrates the same function in a new task (query set). For example, we provide 8 medical images of liver segmentations (both the images and their gt masks) and we would like the model to segment a liver in an unseen medical image.

Since we are doing an ICL-related research, we need to have some support images. We have set one of the training volumes to be the support volume for the rest of the training volume and the test volumes. Slices from that volume will become the support images. Suppose we want to segment a liver in a query image, we will draw images from that support volume to be the support images. This approach has its own limitation and it will be discussed in the Branch3 section.

## Important paths

Git repo: https://github.com/fuchuenli/medsam2-icl.git

Docker image: `fredaiml/ao7j2o:latest`

Best BTCV checkpoint: /mnt/12T/03_khoa/medsam2-icl/checkpoint/2025-01-09-12-20-08/epoch_54.pth **(Dice score: 0.879)**

BTCV Dataset: /mnt/12T/fred/medical_image/btcv

KiTS23 Dataset: `/mnt/12T/fred/medical_image/KiTS23/KiTS23`

Combined Dataset (Sarcoma + MSD + BTCV): `/mnt/12T/fred/medical_image/Combined_Dataset`

Sarcoma Dataset: `/mnt/12T/NIFTI_Soft-tissue-Sarcoma`

## Building a working environment

1. Download and install the docker image in the above link with:
   ```bash
   docker pull fredaiml/ao7j2o:latest
   ```
2. Download the git repo in the above link with:
   ```bash
   git clone https://github.com/fuchuenli/medsam2-icl.git
   ```
3. Activate the conda environment with:
   ```bash
   conda activate medsam2
   ```
4. There are 2 working branches which are `single-class-kits` and `single-class-combined`. Please switch to the desired branch using:
   ```bash
   git checkout desired_branch
   ```
5. Download the current best checkpoints with:
   ```bash
   cd checkpoint/
   bash download_ckpts.sh
   ```

## Training

To train a model, use the following command:

```bash
python train_3d.py -exp_name=single-class -dataset=kits -pretrain=./checkpoint/MedSAM2_pretrain.pth -val_freq=5 -lr=5e-5 -gpu_device=0 -image_size=512 -out_size=512 -video_length=12 -num_support=8 -support_instance=case_00028.nii.gz -data_path=/workspace/fred/medical_image/KiTS23/KiTS23/ -checkpoint_path=./checkpoint
```

There are a lot of arguments could be adjusted. The above are the majors and please check [`cfg.py`](http://cfg.py) for all of the arguments. 

```bash
python train_3d.py \
-exp_name=branch3-btcv\
-dataset=btcv\
-data_path=/workspace/fred/medical_image/btcv \
-distributed=False\
-gpu_device=0\
-wandb_enabled=True
```


In the checkpoint folder, you should see `MedSAM2_pretrain.pth` and `best_kits_checkpoint.pth` . MedSAM2_pretrain.pth is a checkpoint from MedSAM2 and best_kits_checkpoint.pth is our latest trained checkpoint with KiTS23.

## Evaluation

To evaluate a model,  it is very much similar to training.

```bash
python eval_3d.py \
			    -exp_name=single-class \
			    -dataset=kits \
			    -pretrain=./checkpoint/MedSAM2_pretrain.pth \
			    -gpu_device=0 \
			    -image_size=512 \
			    -out_size=512 \
			    -num_support=8 \
			    -support_instance=case_00028.nii.gz \
			    -data_path=/mnt/fred/medical_image/KiTS23/KiTS23/ 
```

## General Workflow

The workflow could be divided into 3 sessions.

1. Input support images and their gt masks
2. Predict the segmentation masks of the query images
3. Evaluate and backpropagate the losses

The general workflow of the code are as follows.

### File Level

`train_3d.py` → [`function.py`](http://function.py) → `sam2_video_predictor.py` → `sam2_base.py` → model_components

- `train_3d.py`: Preparation of the training such as creating the model, prepare the dataloader and prepare the optimizer etc.
- [`function.py`](http://function.py): Main training and evaluation function.
- `sam2_video_predictor.py`: A high level class to connect the components of the model such as  the image encoder, prompt encoder and mask decoder etc
- `sam2_base.py`: A Low level class to connect to the components

### Function Level

`main()` → `train()` → `train_sam()`

- The above functions are to prepare the required elements for training such as dataloaders and creating a model
    
     →  `net.train_add_new_mask()`
    
    - The support images and support masks are added in this step before forward propagating the query images
    
    → `net.train_propagate_in_video()` 
    
    - Forward propagation of the query images starts from here
        
        → `self._run_single_frame_inference()` 
        
        - The above method are used to connect to the model components
            
            → `self.get_image_feature()`
            
            - Connect to the image encoder to get the encoded image
            
            → `self.track_step()`
            
            - The above method are used to connect to the model components
                
                → `self._prepare_memory_conditioned_features()`
                
                - Connect to the memory attention to mix the query images with the memories
                
                → `self._forward_sam_heads()`
                
                - Connect to the prompt encoder to get the sparse and the dense embedding
                - Connect to the mask decoder to get the predicted masks, predicted IoU and predicted occulsion score
                
                → `self._encode_new_memory()` 
                
                - Connect to the memory encoder to create new memory for the current query images and the predicted masks and save the memories into the memory bank. A point worth noting is the memory bank is a dictionary named `inference_state[”output_dict”]`

##High-level Architecture (image dummy for put)

### BTCV KITS to some important function

- `net.train_add_new_mask()`
    - This function is to get the support sets into the memory bank before forward passing the query images.
    - When we predict the segmentation masks of the query images in the forward pass(`net.train_propagate_in_video()` /`self._run_single_frame_inference()`), the query images can use the support sets as examples to do segmentation.
    - One important tricks of the function is the gt masks are being used instead of the predicted masks in creating the memories of the support images(Please check `self._encode_new_memory()`for more details)
- `self._prepare_memory_conditioned_features()`
    - This function is to fuse the encoded query images with the memories to target the objects we want to segment.
    - For the conventional approach, we choose all of the support images and the most recent 6 query images as the memories and we do attention between the current encoded query images with the memories. Ideally, the objects we want to segment will become more prominent since the objects in the memories are also prominent.
    - maskmem_features, their positional embeddings and the object pointers are the memories mentioned above. Please see below `self._forward_sam_heads()` and `self._encode_new_memory()` for the details.
- `self._forward_sam_heads()`
    - This function first produces sparse embeddings and dense embeddings from the prompt encoder. Then, it predicts segmentation masks from the mask decoder.
    - Prompt encoder: The prompt encoder will convert prompts(point, bbox and masks) into an embedding. For points and bboxs, they will be converted into sparse embeddings and it will be converted to dense embeddings for masks. In our case, the gt masks from the support set will be inputed into the prompt encoder during `net.train_add_new_mask()`.
    - Mask decoder: The mask decoder takes in memory conditioned query image, sparse embeddings and dense embeddings to produce the outputs(mask token, object score token and IoU token) with a transformer. The below image is a diagram showing how the transformer works.
    - The memory conditioned query images will do cross-attention with the tokens and the tokens will do self-attention to produce 3 outputs which are mask token, object score token and IoU token which corresponding to the input tokens.
        - Maks token: produce object pointers and hence segmentation masks
            - Object pointer: a protoytpe of the object which is a embedding carries sementic meaning of the object
            - Predicted masks: the object pointer does dot product with the memory conditioned query images(upscaled) to get the predicted masks
        - Object score token: determine whether the object is in the frame(occulsion score). A point worthing noting, if occulsion score < 0, it predict the object is not present in the image and it will set predicted masks to all black regardless of the predicted masks. Please check
        - IoU token: predict the IoU score of the segmentation

- `self._encode_new_memory()`
    - This function is to create memories to store in the memory back and the memories could be used during `self._prepare_memory_conditioned_features()` .
    - The memory is created by inputing the encoded query images and the predicted masks into a convolutional-based module. The outputs are the maskmem_features and its positional embedding. They will be used in `self._prepare_memory_conditioned_features()`.
    - A point to note is the memory back is an item in a dictionary inference_state[”output_dict”]. For `inference_state[”output_dict”][”cond_frame_output”]`, it stores support set memories from `net.train_add_new_mask()` while `inference_state[”output_dict”][”non_cond_frame_output”]` is for query set memories.

## Branch 1: single-class-kits

1. This branch is developing for the KiTS23 dataset which is a dataset with segmentations of kidneys, kidney tumors and renal cysts
2. The main features of this branch is the dataloader in [kits.py](http://kits.py).  You may follow `get_dataloader()` in `tran_3d.py` to get to the dataloader.
3. By following ICL-SAM approach, we combined the kidney and kidney tumor into one class and set renal cysts as the background.
4. The latest class dice score is 0.904 which is superior when it compares to ICL-SAM results, 0.844.
5. The memory selector has not been added back in this branch.

A general workflow of the dataloader is described as below.

### Dataloader

**Query volume**

1. Load the entire volume of the path
2. Combine the labels of kidneys and kidney tumors and set cysts as the background
3. Remove all the slices that have backgrounds only
4. Normalization and interpolation

**Support volume**

1. Load the entire volume of the path
2. Combine the labels of kidneys and kidney tumors and set cysts as the background
3. Remove all the slices that have backgrounds only
4. Normalization and interpolation

**Output**

A dictionary of 

{”image”: tensor of the query volume, 

“label”: the gt mask of the query volume, 

“support_image”: the support volume, 

“support_label”: the gt mask of the support volume,

“name”: the name of the query volume

}

### Data Preprocessing after the dataloader

The outputs of the dataloader are with multi-class. To make the output work in a single class fashion, we use extract_object() in function.py. The step of extract_object are shown as follows.

Query volume

1. Find frames that contain the class in the volume
2. Randomly draw 10 frames(since we may not have enough memory to process the whole volume)

Support volume

1. Find frames that contain the class in the volume
2. Randomly draw 10 frames(since we may not have enough memory to process the whole volume)

**Output**

A dictionary of 

{”image”: tensor of the query volume of a particular class, 

“label”: the gt mask of the query volume of a particular class, 

“support_image”: the support volume of a particular class, 

“support_label”: the gt mask of the support volume of a particular class

}

### Combined:

1. This branch is developing for the combined dataset which consisted of BTCV, MSD and the public sarcoma dataset.
2. The main features of this branch is the dataloader in [combined.py](http://combined.py).  You may follow `get_dataloader()` in `tran_3d.py` to get to the dataloader.
3. The dataloader of this branch works similar to the dataloader in branch1. However, we have changed a couple things to load huge multiple datasets.
    1. Load a volume in different chunks to prevent insufficent memory.
    2. Set some special conditions to solve cater some dataset. For example, some volumes are have an extra dimension for multiple modality which result in 4 dimensions.
4. An attempt has been made to train with this huge dataset in the A6k server with a singel GPU for 9 hours. The training went through well and no errors are shown. However, it shows that 215 hours are still needed after training it overnight.
5. `single-class-combined` is not as updated as the `single class-kits` branch so some codes between them may be different.
6. The memory selector has not been added back in this branch.

## Branch3: Mix-selector-mask-confidence-focal

The main feature of this branch is an addition of a memory selector. The workflow of this branch is very much similar to the general workflow mentioned above except for the `self._prepare_memory_conditioned_features()`.  We modified the way of selecting the memory to mix with the current query image. The method of our new approach lives in `memory_score()` in `self._prepare_memory_conditioned_features()`.

**This branch is developed to work with BTCV.**

### Memory Scorer

For the `self._prepare_memory_conditioned_features()` memtioned above, we select the most recent frames as the memories to mix with the current query images. However, we select the most relevant, reliable and confident frames as the memories in this approch.

The original approach only select frames based on recenty which means they only select the most recent frames. In our approach, we select frames based on 3 metrics which are mask confidence, IoU and recenty. For the purpose of testing the upper bound of the memory selector, gt masks are involved to evaluate the score of those metrics.
