import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

axis = 1
kernel = np.array(pd.read_csv("VF-2_1 wTool/12_28/2021_12_28 16-43-22_VF-2-1_with tool.csv").iloc[:,axis])

# plt.plot(kernel)
# plt.title('VF-2_1')

standard_score = max(np.convolve(np.flip(kernel), kernel))


folders = ["VF-2_1 wTool/22_01_08","VF-2_1 wTool/12_23","VF-2_1 wTool/12_24", "VF-2_1 wTool/12_25",
           "VF-2_1 wTool/12_26 (plate was changed)", "VF-2_1 wTool/12_27", "VF-2_1 wTool/12_28","VF-2_1 wTool/2022_02_16",
           "VF-2_2 wTool/2021_12_23", "VF-2_2 wTool/2022_01_11","VF-2_2 wTool/2022_02_16"]
for folder in folders:
    print(folder)
    for file in os.listdir(folder):
        if ".csv" in file:
            # print(file)
            unknown = np.array(pd.read_csv(f"{folder}/{file}").iloc[:,axis])
            score = max(np.convolve(np.flip(unknown), kernel))
            print(score/standard_score)