import sys
import os

# Get the path of the parent directory of the current working directory
sys.path.append(os.getcwd())
from data.merge_X_all import folders_to_process
from analysis.read_merge_align_write import merge


folder_list = folders_to_process("VF-2")
assets = {"Bambu_S": "2023_11_25", "Bambu_M": "2023_11_25", "UR5e_N": "2023_11_21",
          "UR10e_A": "2023_11_21", "VF-2_1": "2023_11_18", "VF-2_2": "2023_11_18"}

for folder in folder_list:
    asset_types = {"Bambu":[], "UR":[], "VF":[]}
    path = os.path.join(r"data\Kernels", folder)
    folder_assets = [asset for asset in os.listdir(path) if os.path.isdir(os.path.join(path, asset))]
    for typ in asset_types:
        asset_types[typ] += [asset for asset in folder_assets if typ in asset]
        if asset_types[typ] == []:
            continue
        files = [os.path.join(path, asset, "merged_X.csv") for asset in asset_types[typ] if os.path.exists(os.path.join(path, asset, "merged_X.csv"))]
        merge(files,path)