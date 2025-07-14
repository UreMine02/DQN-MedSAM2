from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="/mnt/fred/MedSeg/medsam2-fred/checkpoint/best_mix_selector_checkpoint.pth",  
    path_in_repo="best_mix_selector_checkpoint.pth",  # Where you want to store the file in the repo
    repo_id="fred1865844/a3o954",
    repo_type="model"  # Can also be 'dataset' or 'space' depending on the repo
)