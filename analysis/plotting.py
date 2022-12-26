import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.signal import correlation_lags


def shift_for_maximum_correlation(x, y):
    correlation = correlate(x, y, mode="full")
    lags = correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation)]
    print(f"Best lag: {lag}")
    if lag <= 0:
        y = y[abs(lag):]
    else:
        y = np.insert(y, 0, np.full(lag, y.mean()))
    return y


def plot_several_assets(asset_type, magnitude, sig_len):
    samples = {}
    home = rf"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\{asset_type}"
    i = 0  # counter for each asset
    for folder in os.listdir(home):
        if os.path.isdir(f"{home}/{folder}"):
            # count the number of samples of each asset -> create a dictionary and then increment the counter
            samples[folder] = 0
            n = 0  # counter for each run
            j = 0  # counter for each file
            for file in os.listdir(home + "\\" + folder):
                if ".csv" in file:
                    samples[folder] += 1
                    j += 1

                    df = np.array(pd.read_csv(f"{home}/{folder}/{file}").iloc[:, 1])
                    df = df - np.mean(df)
                    df = df + magnitude * i
                    if j == 1:
                        df_bench = df
                    else:
                        df = shift_for_maximum_correlation(df_bench, df)

                    plt.plot(df[:sig_len], alpha=0.3, color="blue")  # plot all files in main plot
                    # plt.pause(0.1)
                    # plt.show()
                n += 1
                if n == 3:
                    break
            i += 1

    plt.title(f"Vibration patterns of {asset_type} in X axis", fontsize=24)
    plt.xticks(np.arange(0, sig_len + 500, 500), np.arange(0, sig_len / 100 + 5, 5) / 10, fontsize=24)
    plt.yticks(np.arange(-magnitude / 4, magnitude * (len(samples) - 0.25), magnitude / 4),
               np.array([[-magnitude / 4, 0, magnitude / 4, i] for i in samples.keys()]).flatten(), fontsize=20)
    plt.xlabel("Time, sec", fontsize=24)
    plt.ylabel("Acceleration, g", fontsize=24)
    plt.grid()
    plt.xlim(-50, sig_len + 50)
    plt.ylim(-magnitude / 2, magnitude * (len(samples) - 0.5))
    [plt.hlines(magnitude * (0.5 + i), -50, sig_len + 50, color="red", alpha=0.5) for i in range(len(samples))]


if __name__ == "__main__":
    #
    # # look at all 3 axes without last 500 rows
    # folder = r"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\2022_12_12\UR-5e Cary"
    #
    # dim = []  # dimensions of samples
    # for file in os.listdir(folder):
    #     if ".csv" in file:
    #         df = np.array(pd.read_csv(f"{folder}/{file}").iloc[:8500, 1:4])
    #         # print(len(df))
    #         # df = df[:3200]
    #         plt.figure(file)
    #         plt.plot(df)
    #         print(file)
    #         plt.pause(0.2)
    #         plt.show()
    #         input()
    #         dim.append(df.shape[0])

    ###########################################################################
    ############## plot all assets with several runs on one plot ##############

    ############ For FV-2s ############

    magnitude = 0.2  # for VF-2
    asset_type = "VF"
    sig_len = 9000
    plot_several_assets(asset_type, magnitude, sig_len)

    plt.subplots_adjust(bottom=0.075, top=0.95, left=0.055, right=0.99)
    plt.show()

    ############ For Prusas ############

    magnitude = 1  # for PRUSAEE
    asset_type = "PRUSA"
    sig_len = 8500
    plot_several_assets(asset_type, magnitude, sig_len)

    plt.subplots_adjust(bottom=0.075, top=0.95, left=0.055, right=0.99)
    plt.show()

    ############ For UR-5e ############
    magnitude = 0.5  # for UR
    asset_type = "UR"
    sig_len = 4500
    plot_several_assets(asset_type, magnitude, sig_len)

    plt.subplots_adjust(bottom=0.075, top=0.95, left=0.055, right=0.99)
    plt.show()
