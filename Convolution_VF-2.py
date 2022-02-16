import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

folders = ["VF-2_1 wTool/12_28", "VF-2_2 wTool/2021_12_23"]
for i, folder in enumerate(folders):
    k=0
    for j, file in enumerate(os.listdir(folder)):
        if k==2:
            break
        if ".csv" in file:
            df = pd.read_csv(f"{folder}/{file}")
            globals()[f"kernel{i}{j}"] = np.array(df.iloc[:,1])
            k+=1
            # plt.figure()
            # plt.plot(globals()[f"train{i}{j}"])

fig, ax = plt.subplots(2,2)
ax[0][0].set_title("VF-2_1")
ax[0][1].set_title("VF-2_2")

k=0
for i in range(2):
    for j in range(2):
        ax[i][j].plot(globals()[f"kernel{j}{i}"], label="Kernel")
        k+=1


s = np.zeros([2,2])
s1 = np.zeros([2,2])


for folder in ["VF-2_1 wTool/12_27", "VF-2_2 wTool/2021_12_23"]:#, , "VF-2_1 wTool/12_27"]:
    for num, file in enumerate(os.listdir(folder)):
        if num in range(2):
            continue
        if ".csv" in file:
            unknown = np.array(pd.read_csv(f"{folder}/{file}").iloc[:, 1])

            for i in range(2):
                for j in range(2):
                    a=np.convolve(np.flip(unknown), globals()[f"kernel{j}{i}"])
                    s[i,j]=max(a)
            S = np.max(s)
            s1 = s/S
            print("Looks like plot #", s1.argmax()+1)
            ax[s1.argmax()//2][s1.argmax()%2].plot(unknown, label=f"{file}")
            ax[s1.argmax()//2][s1.argmax()%2].legend()
plt.show()