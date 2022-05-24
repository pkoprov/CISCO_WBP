import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

folder = "./Kernels/5_7_2022/UR-5e-4"
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


