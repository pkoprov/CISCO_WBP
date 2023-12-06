from analysis.read_merge_align_write import merge, get_folder_path
import sys
import os
from datetime import datetime


# Get the path of the parent directory of the current working directory
sys.path.append(os.getcwd())


folders = os.listdir(r"data\Kernels")
init_idx = folders.index(os.path.basename(get_folder_path()))
fin_idx = folders.index(datetime.now().strftime("%Y_%m_%d"))
folders = folders[init_idx:fin_idx+1]

asset_types = ["Bambu", "UR", "VF"]

for folder in folders:
    print(os.path.join(r"data\Kernels", folder))
    asset_fold = [asset for asset in os.listdir(os.path.join(
        r"data\Kernels", folder)) if os.path.isdir(os.path.join(r"data\Kernels", folder, asset))]
    for asset in asset_types:
        if asset in asset_types:
            path = os.path.join(r"data\Kernels", folder, asset)
            merge(os.path.join(r"data\Kernels", folder, asset))
    
    
