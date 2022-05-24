import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

axis = 1
kernel = np.array(pd.read_csv("../data/Kernels/5_7_2022/UR-5e-1/UR-5e-1_270935_1651511658.csv").iloc[:3200, axis])

# plt.subplots(4,1)
# plt.subplot(4,1,1)
# plt.plot(kernel)
# plt.title('VF-2_1')

standard_score = max(np.convolve(np.flip(kernel), kernel))

n = 1

folders = "./Kernels/5_7_2022/"
for folder in os.listdir(folders):
    if os.path.isdir(f"{folders}/{folder}"):

        print(f"____________________{folder}____________________")
        normalized_score = []
        plt.subplot(4,1,n)

        for m, file in enumerate(os.listdir(f"{folders}/{folder}")):
            if ".csv" in file:
                # print(file)
                unknown = np.array(pd.read_csv(f"{folders}/{folder}/{file}").iloc[:3200,axis])
                # if (m+1)%5 == 0:                plt.plot(unknown)
                score = max(np.convolve(np.flip(unknown), kernel))
                normalized_score.append(score/standard_score)
                # print(score/standard_score)
        n+=1
        normalized_score = np.array(normalized_score)
        print(stats.describe(normalized_score))

