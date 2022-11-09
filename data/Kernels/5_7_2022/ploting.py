import sys

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


folder = r"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\5_7_2022\UR-5e-1"
plt.figure()
for file in os.listdir(folder):
    if ".csv" in file:
        # input()
        df = np.array(pd.read_csv(f"{folder}/{file}").iloc[:,1])
        print(len(df))
        df = df[:3200]
        plt.plot(df)
        print(file)
        # dim.append(df.shape[0])
        # plt.pause(0.1)


for i in range(1,5):
    folder = fr"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\5_7_2022\UR-5e-{i}"
    # plt.subplot(4, 1, i)
    for j in range(1,11):
        file = os.listdir(folder)[j]

        df = np.array(pd.read_csv(f"{folder}/{file}").iloc[:, 1])
        df = df - np.mean(df) + 10 * i
        print(len(df))
        df = df[:3200]
        plt.plot(df, alpha=0.2, color="blue")
plt.title("Vibration patterns of UR-5e in X axis", fontsize=24)
plt.xlabel("Time, sec",fontsize=24)
plt.ylabel("Acceleration, g",fontsize=24)
plt.grid()
plt.xticks(np.arange(0, 3600, 400), np.arange(0, 45, 5)/10, fontsize=24)
plt.yticks(np.arange(7.5, 46, 2.5), np.array([[-2.5,0,2.5,f"UR-5e-{i}"] for i in range(1,5)]).flatten(), fontsize=24)

plt.xlim(0, 3200)
plt.ylim(6.5, 45)
[plt.hlines(15 + 10 * i, 0, 3200, color="red", alpha=0.5) for i in range(3)]


plt.figure("Vibration patterns of Prusa I Indigo")
len_list = []
folder = fr"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\2022_11_07\Prusa I Indigo"
for file in os.listdir(folder):
    if ".csv" in file:
        df = np.array(pd.read_csv(f"{folder}/{file}").iloc[:, 1])
        df = df - np.mean(df)
        len_list.append(len(df))
        print(len(df))
        df = df[:8800]
        print(file)
        plt.plot(df, alpha=0.2, color="blue")
        plt.pause(0.1)
        input()

plt.title("Vibration patterns of UR-5e in X axis", fontsize=24)
plt.xlabel("Time, sec",fontsize=24)
plt.ylabel("Acceleration, g",fontsize=24)
plt.grid()
plt.xticks(np.arange(0, 3600, 400), np.arange(0, 45, 5)/10, fontsize=24)
plt.yticks(np.arange(7.5, 46, 2.5), np.array([[-2.5,0,2.5,f"UR-5e-{i}"] for i in range(1,5)]).flatten(), fontsize=24)

plt.xlim(0, 3200)
plt.ylim(6.5, 45)
[plt.hlines(15 + 10 * i, 0, 3200, color="red", alpha=0.5) for i in range(3)]