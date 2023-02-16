import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.signal import correlation_lags


def shift_for_maximum_correlation(x, y, timed = False):
    x_full = x
    y_full = y
    if x.ndim == 2:
        x = x_full.iloc[:, 1]
    if y.ndim == 2:
        y = y_full.iloc[:, 1]
    correlation = correlate(x, y, mode="full")
    lags = correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation)]
    print(f"Best lag: {lag}")
    if lag < 0:
        if timed:
            y_full = np.column_stack([y_full[abs(lag):, 0] - y_full[abs(lag), 0], y_full[abs(lag):, 1:]])
        elif y_full.ndim == 1:
            y_full = np.hstack([y[abs(lag):], x[lag:]])
    elif lag > 0:
        if timed:
            #
            start = x_full.iloc[:lag, 0]
            time = np.hstack([start,y_full.iloc[:-lag,0]+start.iloc[-1]+0.001])
            y_full = np.column_stack([time, np.vstack([x_full.iloc[:lag,1:], y_full.iloc[:-lag,1:]])])
        elif y_full.ndim == 1:
            y_full = np.hstack([x[:lag], y[:-lag]])

    return y_full, lag


def plot_several_assets(asset_type, magnitude, sig_len, n_samples):
    samples = {}
    home = rf"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\{asset_type}"
    i = 0  # counter for each asset
    for folder in os.listdir(home):
        if os.path.isdir(f"{home}/{folder}"):
            # count the number of samples of each asset -> create a dictionary and then increment the counter
            samples[folder] = 0

            df = pd.read_csv(f"{home}/{folder}/merged.csv")
            df = df.T[1:]
            df -= df.mean()
            df = df +i * magnitude
            for n in range(n_samples):
                plt.plot(df.index.values.astype(float), df.iloc[:sig_len, n], alpha=1/n_samples, color="blue") # plot all files in main plot
            i += 1

    plt.title(f"Vibration patterns of {asset_type} in X axis", fontsize=24)
    plt.yticks(np.arange(-magnitude / 4, magnitude * (len(samples) - 0.25), magnitude / 4),
               np.array([[-magnitude / 4, 0, magnitude / 4, i] for i in samples.keys()]).flatten(), fontsize=20)
    plt.xlabel("Time, sec", fontsize=24)
    plt.ylabel("Acceleration, g", fontsize=24)
    plt.grid()
    plt.margins(0.005)
    plt.ylim(-magnitude / 2, magnitude * (len(samples) - 0.5))
    [plt.hlines(magnitude * (0.5 + i), 0, sig_len/1000, color="red", alpha=0.5) for i in range(len(samples))]


if __name__ == "__main__":

    ############## plot all assets with several runs on one plot ##############

    ############ For FV-2s ############

    magnitude = 0.2  # for VF-2
    asset_type = "VF"
    sig_len = 8500
    plot_several_assets(asset_type, magnitude, sig_len, n_samples=10)
    plt.gcf().set_size_inches(20, 4)
    plt.tight_layout()
    plt.savefig("Vibration patterns for VF.png")
    plt.show()

    ############ For Prusas ############
    magnitude = 1  # for PRUSA
    asset_type = "PRUSA"
    sig_len = 8000
    plot_several_assets(asset_type, magnitude, sig_len, n_samples=10)
    plt.gcf().set_size_inches(20, 10)
    plt.tight_layout()
    plt.savefig("Vibration patterns for Prusa.png")
    plt.show()

    ############ For UR-5e ############
    magnitude = 0.5  # for UR
    asset_type = "UR"
    sig_len = 4000
    plot_several_assets(asset_type, magnitude, sig_len, n_samples=10)
    plt.gcf().set_size_inches(20, 6.5)
    plt.tight_layout()
    plt.savefig("Vibration patterns for UR.png")
    plt.show()
