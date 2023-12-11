
import sys
import os

# Get the path of the parent directory of the current working directory
sys.path.append(os.getcwd())
from data.read_merge_align_write import merge
from data.merge_X_all import folders_to_process


def main(force=False):
    folder_list = folders_to_process("VF-2")

    for folder in folder_list:
        print(folder)
        asset_types = {"Bambu": [], "UR": [], "VF": []}
        path = os.path.join(r"data\Kernels", folder)
        folder_assets = [asset for asset in os.listdir(
            path) if os.path.isdir(os.path.join(path, asset))]
        for typ in asset_types:
            asset_types[typ] += [asset for asset in folder_assets if typ in asset]
            if any([typ in f for f in [csv for csv in os.listdir(path) if csv.endswith("_merged.csv")]]) and not force:
                print(typ, "_merged.csv already exists")
                continue
            if asset_types[typ] == []:
                continue
            files = [os.path.join(path, asset, "merged_X.csv") for asset in asset_types[typ]
                     if os.path.exists(os.path.join(path, asset, "merged_X.csv"))]
            merge(files, path)


if __name__ == "__main__":
    main()
