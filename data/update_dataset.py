
import pandas as pd
import sys
import os

# Get the path of the parent directory of the current working directory
sys.path.append(os.getcwd())
from data.merge_X_all import folders_to_process, START
from analysis.FDA import is_convertible_to_float
from data.read_merge_align_write import select_files


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

    drop_rows = new_df.shape[0] - 30 + old_df.shape[0]

    old_df.drop(range(drop_rows), inplace=True)
    df = pd.concat([old_df, new_df], axis=0, ignore_index=True)

    asset_type = old_df.iloc[0, 0]

    new_dir = os.path.dirname(new)

    filename = new_dir+f"/{asset_type}_merged_new.csv"
    df.to_csv(filename, index=False)
    print("saved to ", filename)


def main(force=False):
    folder_list = folders_to_process("VF-2")
    for folder in folder_list:
        assets = [i for i in os.listdir(
            f"data/Kernels/{folder}") if os.path.isdir(os.path.join(f"data/Kernels/{folder}", i))]
        for asset in assets:
            if "VF" in asset:
                typ = "VF-2"
            elif "Bambu" in asset:
                typ = "Bambu"
            elif "UR" in asset:
                typ = "UR"
            else:
                print("Unknown type")
                typ = input("Type the asset type:\n>>")
            if os.path.exists(f"data\Kernels\{folder}\{asset}\{asset}_merged_new.csv") and not force:
                print(f"File {asset}_merged_new.csv already exists")
                continue
            elif folder == START[typ]:
                print(f"This is the initial folder for {asset}... skipping")
                continue
            try:
                prev_fold = folder_list[folder_list.index(folder)-1]
                old = f"data\Kernels\{prev_fold}\{asset}\merged_X.csv" if prev_fold == START[
                    typ] else f"data\Kernels\{folder_list[folder_list.index(folder)-1]}\{asset}\{asset}_merged_new.csv"
                update(old=old,
                       new=f"data\Kernels\{folder}\{asset}\merged_X.csv")
            except FileNotFoundError as e:
                print(e)
                print("Select the file to update")
                old = select_files()[0]
                print("Select the file to add")
                new = select_files()[0]
                update(old=old, new=new)


if __name__ == "__main__":
    main()
