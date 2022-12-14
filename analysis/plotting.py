import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# find max points within each asset
def find_max_points(asset_type, magnitude):
    home = rf"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\{asset_type}"
    max_points = {}
    for i, folder in enumerate(os.listdir(home)):
        if os.path.isdir(f"{home}/{folder}"):
            print(folder)
            max_points[folder] = []
            for file in os.listdir(home + "\\" + folder):
                if ".csv" in file:
                    df = np.array(pd.read_csv(f"{home}/{folder}/{file}").iloc[:, 1])
                    df = df - np.mean(df) + magnitude * i
                    max_points[folder].append(np.argmax(df))
    return max_points


def plot_several_assets(asset_type, magnitude, sig_len, max_points):
    plt.figure()
    home = rf"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\{asset_type}"
    sig_len = sig_len
    i = 0  # counter for each asset
    for folder in os.listdir(home):
        if os.path.isdir(f"{home}/{folder}"):
            n = 0  # counter for each run
            print(folder)
            center = round(np.median(max_points[folder]))
            interval = [center - 500, center + 500]
            for j, file in enumerate(os.listdir(home + "\\" + folder)):
                if ".csv" in file:
                    if j < 20:  # skip first n dfs
                        continue
                    df = np.array(pd.read_csv(f"{home}/{folder}/{file}").iloc[:, 1])
                    df = df - np.mean(df)
                    shift = center - np.argmax(df[interval[0]:interval[1]]) - interval[0]
                    if shift < 0:
                        df = df[-shift:]
                    else:
                        df = np.concatenate((np.zeros(shift), df))
                    df = df + magnitude * i
                    plt.plot(df[:sig_len], alpha=0.3, color="blue")  # plot all files in main plot
                    # plt.pause(0.2)
                    # plt.show()
                    n += 1
            #     if n == 3:
            #         break
            # i += 1
    plt.title(f"Vibration patterns of {asset_type} in X axis", fontsize=24)
    plt.xticks(np.arange(0, sig_len + 500, 500), np.arange(0, sig_len / 100 + 5, 5) / 10, fontsize=24)
    plt.yticks(np.arange(-magnitude / 4, magnitude * (len(max_points) - 0.25), magnitude / 4),
               np.array([[-magnitude / 4, 0, -magnitude / 4, i] for i in max_points.keys()]).flatten(), fontsize=20)
    plt.xlabel("Time, sec", fontsize=24)
    plt.ylabel("Acceleration, g", fontsize=24)
    plt.grid()
    plt.xlim(-50, sig_len + 50)
    plt.ylim(-magnitude / 2, magnitude * (len(max_points) - 0.5))
    [plt.hlines(magnitude * (0.5 + i), -50, sig_len + 50, color="red", alpha=0.5) for i in range(len(max_points))]


if __name__ == "__main__":

    # look at all 3 axes without last 500 rows
    folder = r"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\2022_12_12\UR-5e Cary"

    dim = []  # dimensions of samples
    for file in os.listdir(folder):
        if ".csv" in file:
            df = np.array(pd.read_csv(f"{folder}/{file}").iloc[:8500, 1:4])
            # print(len(df))
            # df = df[:3200]
            plt.figure(file)
            plt.plot(df)
            print(file)
            plt.pause(0.2)
            plt.show()
            input()
            dim.append(df.shape[0])

    ###########################################################################
    ############## plot all assets with several runs on one plot ##############

    ############ For FV-2s ############

    magnitude = 0.2  # for VF-2
    asset_type = "VF"

    sig_len = 9000
    max_points = find_max_points(asset_type, magnitude)
    plot_several_assets(asset_type, magnitude, sig_len, max_points)

    plt.subplots_adjust(bottom=0.075, top=0.95, left=0.055, right=0.99)

    ############ For Prusas ############

    magnitude = 1  # for PRUSA
    asset_type = "PRUSA"
    sig_len = 8500
    max_points = find_max_points(asset_type, magnitude)
    plot_several_assets(asset_type, magnitude, sig_len, max_points)

    plt.subplots_adjust(bottom=0.075, top=0.95, left=0.055, right=0.99)

    ############ For UR-5e ############
    magnitude = 0.5  # for UR
    asset_type = "UR"
    sig_len = 4500
    max_points = find_max_points(asset_type, magnitude)
    plot_several_assets(asset_type, magnitude, sig_len, max_points)

    plt.subplots_adjust(bottom=0.075, top=0.95, left=0.055, right=0.99)
