import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog


plt.ion()

# Function to check if a string can be converted to a float
def is_convertible_to_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
#  select folder
def get_folder_path(folder=r".\data\Kernels"):
    if folder is None:
        folder = r".\data\Kernels"
    # Initialize Tkinter
    root = tk.Tk()

    # Show an "Open" dialog box and return the path to the selected folder
    folder_path = filedialog.askdirectory(
        initialdir=folder)
    root.destroy()  # Close the Tkinter root window to prevent freezing
    return folder_path


def select_files(init_dir=None):
    # This ensures that we're creating a new Tk instance every time
    root = tk.Tk()

    # This should bring up the file dialog
    file_paths = filedialog.askopenfilenames(initialdir=init_dir)

    # Close the root window after selection
    root.destroy()
    return file_paths


def save_file(file=None,folder = r".\data\Kernels"):
    root = tk.Tk()

    # Show the save file dialog
    file_path = filedialog.asksaveasfilename(initialfile=file,
                                             initialdir=folder,
                                             defaultextension=".csv",  # You can set a default extension
                                             # Define file types
                                             filetypes=[
                                                 ("CSV files", "*.csv"), ("All files", "*.*")],
                                             )

    root.destroy()  # Destroy the root window

    return file_path


def create_dataset(folder=None):
    # read data from each sample and merge into one dataframe
    df_all, asset, axis, root = united_frame(folder)
    print(root)

    df_all = df_all.transpose()
    # set Time to be columns
    df_all.columns = df_all.iloc[0]
    df_all.drop("Time", inplace=True)
    # rename indices from Samples to UR-5e_...
    indices = [asset for i in range(df_all.shape[0])]
    df_all.index = indices
    df_all.index.name = "asset"
    # save to csv
    filename = save_file(f"merged_{axis}.csv", folder)

    df_all.to_csv(filename)
    print("saved to ", filename)


def united_frame(folder):
    print("Select folder with data")
    if folder is None:
        root = get_folder_path(folder)
    else:
        root = folder
    if not root:
        exit(0)

    # check if folder contains subfolders
    if not any([os.path.isdir(os.path.join(root, f)) for f in os.listdir(root)]):
        asset = os.path.basename(root)
        root = os.path.dirname(root)
    else:
        subfolders = os.listdir(root)
        print("Available assets:")
        for i, subfolder in enumerate(subfolders):
            print(f"{i}: {subfolder}")
        asset = input("Select asset:\n>>> ")
        asset = subfolders[int(asset)]

    columns = ["Time", "X", "Y", "Z"]

    # axis = input("Select axis:\nx, y, z\n>>> ").upper()
    axis = "X"

    # sort by time and take first 4000 samples for UR-5 and 8500 for VF-2
    if "UR" in asset:
        length = 4000
    elif "VF" in asset:
        length = 8500
    elif "Bambu" in asset:
        length = 4000
    else:
        length = int(input("How many seconds of data to take?\n>>> "))*1000

    df_all = pd.DataFrame(
        np.round(np.arange(0, 8.5, 0.001), 3), columns=['Time'])
    i = 0
    for file in os.listdir(os.path.join(root, asset)):
        if ".csv" in file and asset in file and "merged" not in file:
            i += 1
            df = pd.read_csv(f"{root}/{asset}/{file}")
            df.columns = columns
            df = df[["Time", axis]]
            df_all = prepare_sample(df, i, df_all)

    df_all = df_all.sort_values(by="Time").reset_index(
        drop=True).iloc[:length, :]

    if df_all.isna().any().any():
        df_all.fillna(df_all.median(), inplace=True)

    # deal with missing values
    for row in df_all.iterrows():
        if row[1].isna().any():
            row[1].fillna(row[1].median(), inplace=True)

    # remove outliers
    alldata = df_all.drop("Time", axis=1)
    # Calculate the overall mean and standard deviation
    mean = alldata.mean()
    std = alldata.std()
    # Find values that are outside of mean ± 3*std
    outliers = df_all[(np.abs(alldata - mean) > 3 * std)]
    if outliers.any().any():  # Check if any True exists in the DataFrame
        warnings.warn("Outliers found", UserWarning)
    for column in alldata.columns:
        alldata[column] = np.where(
            outliers[column].notna(), mean[column], alldata[column])

    # If you want to update the original df_all DataFrame with corrected values:
    df_all.update(alldata)
    return df_all, asset, axis, root


def prepare_sample(df, i=0, df_all=pd.DataFrame(columns=['Time'])):
    df.columns = ["Time", f"Sample_{i}"]
    df["Time"] = df["Time"].round(3)
    df.drop_duplicates(["Time"], inplace=True)
    df.iloc[:, 1] -= df.iloc[:, 1].mean()
    df_all = df_all.merge(df, on="Time", how="outer")
    return df_all


def merge(files=None, folder=None):
    if files is None:
        cmd = "y"
        files = []
        while cmd.lower() == 'y':
            print("Select files to merge")
            path = select_files(r".\data\Kernels")
            if not path:
                break
            print(path)
            files.extend(path)

        files = np.unique(files)
    print(files)

    asset_type = files[0].split('\\')[-2][:-2] if "UR" not in files[0] else "UR"
    df_total = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, index_col=0)
        df.columns = [float(col) if is_convertible_to_float(
            col) else col for col in df.columns]
        df_total = pd.concat([df_total, df], axis=0)
    
    if not folder:
        filename = save_file(f"{asset_type}_merged.csv")
    else:
        filename = os.path.join(folder, f"{asset_type}_merged.csv")

    df_total.to_csv(filename, index_label="asset")
    print("saved to ", filename)


if __name__ == "__main__":
    cmd = None
    while cmd != 'q':
        cmd = input(
            "Create datset for asset or merge datasets (or quit)\n(c\m\q)\n>>> ")
        if cmd == 'c':
            create_dataset()
        elif cmd == 'm':
            merge()
