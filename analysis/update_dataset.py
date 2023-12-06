try:
    from read_merge_align_write import select_files
    from FDA import is_convertible_to_float
except ModuleNotFoundError:
    from analysis.read_merge_align_write import select_files
    from analysis.FDA import is_convertible_to_float
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
    folder_list = os.listdir("data\Kernels")
    start_idx = folder_list.index("2023_11_26")
    stop_idx =  folder_list.index("2023_12_04")+1
    folder_list = folder_list[start_idx:stop_idx]
    asset = "Bambu_S"
    for i in range(1,len(folder_list)):
        print(folder_list[i])
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
            

