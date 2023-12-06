try:
    from read_merge_align_write import select_files
    from FDA import is_convertible_to_float
except ModuleNotFoundError:
    from analysis.read_merge_align_write import select_files
    from analysis.FDA import is_convertible_to_float
from datetime import datetime
import os
import pandas as pd


def update(old=None, new=None):
    if old is None:
        print("Select the file to update")
        old = select_files()[0]

    old_df = pd.read_csv(old)
    old_df = old_df.iloc[:30, :]
    old_df.columns = [float(col) if is_convertible_to_float(
        col) else col for col in old_df.columns]

    if new is None:
        print("Select the file to update with")
        new = select_files()[0]

    new_df = pd.read_csv(new)
    new_df.columns = [float(col) if is_convertible_to_float(
        col) else col for col in new_df.columns]

    old_df.drop(range(new_df.shape[0]), inplace=True)
    df = pd.concat([old_df, new_df], axis=0, ignore_index=True)

    asset_type = old_df.iloc[0, 0]

    new_dir = os.path.dirname(new)

    filename = new_dir+f"/{asset_type}_merged_new.csv"
    df.to_csv(filename, index=False)
    print("saved to ", filename)


if __name__ == "__main__":
    folders = os.listdir("data\Kernels")
    fin_idx = folders.index(datetime.now().strftime("%Y_%m_%d"))

    assets = {"Bambu_S": "2023_11_25", "Bambu_M": "2023_11_25", "UR5e_N": "2023_11_21",
              "UR10e_A": "2023_11_21", "VF-2_1": "2023_11_18", "VF-2_2": "2023_11_18"}
    for asset in assets:
        print("Processing", asset, "...")
        init_idx = folders.index(assets[asset])
        folder_list = folders[init_idx:fin_idx+1]
        for i in range(1, len(folder_list)):
            if folder_list[i] == folder_list[-1]:
                print(f"Last folder for {asset}")
                continue
            print(folder_list[i])
            if os.path.exists(f"data\Kernels\{folder_list[i+1]}\{asset}\{asset}_merged_new.csv"):
                print("File already existss")
                continue
            try:
                update(old=f"data\Kernels\{folder_list[i]}\{asset}\{asset}_merged_new.csv",
                       new=f"data\Kernels\{folder_list[i+1]}\{asset}\merged_X.csv")
            except FileNotFoundError as e:
                print(e)
                print("Select the file to update")
                old = select_files()[0]
                print("Select the file to add")
                new = select_files()[0]
                update(old=old, new=new)
