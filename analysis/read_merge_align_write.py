from math import e
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
# from analysis.FDA import Sample
try:
    from plotting import shift_for_maximum_correlation
except ModuleNotFoundError:
    from analysis.plotting import shift_for_maximum_correlation
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# from skfda.misc.metrics import l2_distance


plt.ion()


#  select folder
def get_folder_path():
    # Initialize Tkinter
    root = tk.Tk()

    # Show an "Open" dialog box and return the path to the selected folder
    folder_path = filedialog.askdirectory(initialdir = r"D:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels")
    root.destroy()  # Close the Tkinter root window to prevent freezing
    return folder_path

# select file


def select_files():
    # This ensures that we're creating a new Tk instance every time
    root = tk.Tk()

    # This should bring up the file dialog
    file_paths = filedialog.askopenfilenames()

    # Close the root window after selection
    root.destroy()
    return file_paths


def main():
    # read data from each sample and merge into one dataframe
    df_all, asset, axis, root = united_frame()
    print(root)
    if os.path.exists(f"{root}/{asset}/benchmark_{axis}.csv"):
        benchmark = pd.read_csv(
            f"{root}/{asset}/benchmark_{axis}.csv", index_col=0)
        benchmark = benchmark.iloc[:, 0]
    else:
        benchmark = df_all.iloc[:, 1:].mean(axis=1)

    # # shift all samples so that they have maximum correlation with benchmark
    # for col in df_all.columns:
    #     if col == "Time":
    #         continue
    #     # shift df so that it has maximum correlation with df_benchmark

    #     df_all[col] = shift_for_maximum_correlation(benchmark, df_all[col])[0]

    # else:
    #     benchmark.index = df_all["Time"]
    #     benchmark.to_csv(f"{root}/{asset}/benchmark_{axis}.csv")


    df_all = df_all.transpose()
    # set Time to be columns
    df_all.columns = df_all.iloc[0]
    df_all.drop("Time", inplace=True)
    # rename indices from Samples to UR-5e_...
    indices = [asset for i in range(df_all.shape[0])]
    df_all.index = indices
    # save to csv
    df_all.to_csv(f"{root}/{asset}/merged_{axis}.csv")


def united_frame():
    print("Select folder with data")
    root = get_folder_path()
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

    axis = input("Select axis:\nx, y, z\n>>> ").upper()

    df_all = pd.DataFrame(columns=['Time'])
    i = 0
    for file in os.listdir(os.path.join(root, asset)):
        if ".csv" in file and asset in file:
            i += 1
            df = pd.read_csv(f"{root}/{asset}/{file}",
                             names=columns)[["Time", axis]]
            df_all = prepare_sample(df, i, df_all )

    # sort by time and take first 4000 samples for UR-5 and 8500 for VF-2
    if "UR" in asset:
        len = 4000
    elif "VF" in asset:
        len = 8500
    else:
        len = int(input("How many seconds of data to take?\n>>> "))*1000

    df_all = df_all.sort_values(by="Time").reset_index(drop=True).iloc[:len, :]

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

def prepare_sample( df,i=0, df_all=pd.DataFrame(columns=['Time'])):
    df.columns = ["Time", f"Sample_{i}"]
    df["Time"] = df["Time"].round(3)
    df.drop_duplicates(["Time"], inplace=True)
    df.iloc[:, 1] -= df.iloc[:, 1].mean()
    df_all = df_all.merge(df, on="Time", how="outer")
    return df_all


def merge():
    cmd = input("Merge files? (y/n)\n>>> ")
    files = []
    while cmd.lower() == 'y':
        print("Select files to merge")
        path = select_files()
        if not path:
            break
        print(path)
        files.extend(path)

    files = np.unique(files)
    print(files)

    type = input("What type of asset is this?\n>>>")
    df_total = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, index_col=0)
        df_total = pd.concat([df_total, df], axis=0)

    folder_to_save = get_folder_path()

    df_total.to_csv(f"{folder_to_save}/{type}_merged.csv", index_label="asset")

# df_all.reset_index(inplace=True)
# df_all.rename(columns={"index": "asset"}, inplace=True)

# sample = Sample(df_all)
# fdata = sample.FData()


# exclude = np.array([], type(int))
# for lim in ['top', 'bottom']:
#     dist = l2_distance(fdata[lim][1:], fdata[lim][0])
#     exclude = np.append(exclude, np.where(dist > dist.mean() + dist.std()))

# exclude = np.unique(exclude)+1
# sample = sample.drop(exclude).reset_index(drop=True)
# fdata = Sample(sample).FData()

# x = df_all.iloc[0, 1:].astype(float)
# y = df_all.iloc[10, 1:].astype(float)
# x.plot()
# y.plot()
# grid = df_all.columns[1:].values.astype(float)


# x_fd = skfda.FDataGrid(x, grid)
# y_fd = skfda.FDataGrid(y, grid)

# x_fd.plot(axes=plt.gca())
# y_fd.plot(axes=plt.gca())


# l2_distance(x_fd, y_fd)

# fdata["top"].plot()
# fdata["top"].mean().plot(axes=plt.gca(), linewidth=4)
if __name__ == "__main__":
    cmd = input("Create datset for asset or merge datasets (c or m)\n>>> ")
    if cmd == 'c':
        main()
    elif cmd == 'm':
        merge()
