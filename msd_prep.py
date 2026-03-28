import os
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm

root = "/data/datasets/MSD"
for task in os.listdir(root):
    if task.startswith("."):
        continue
    
    for subset in ["labelsTr", "labelsTs"]:
        sample = []
        for gt_name in tqdm(os.listdir(os.path.join(root, task, subset)), desc=f"{task}_{subset}", ncols=80):
            if gt_name.startswith("."):
                continue
            
            gt_path = os.path.join(task, subset, gt_name)
            gt_raw = nib.load(os.path.join(root, gt_path))
            gt = np.asanyarray(gt_raw.dataobj)
            
            for label in np.unique(gt):
                if label == 0:
                    continue
                
                pos = (gt == label).sum(axis=(0,1))
                n_pos = np.argwhere(pos).shape[0]
                sample.append((task, gt_path, label, n_pos))
                
        df = pd.DataFrame(sample, columns=["task", "gt_path", "obj_id", "n_pos"])
        df.to_csv(f"./data/MSD/{task}_{subset}.csv")