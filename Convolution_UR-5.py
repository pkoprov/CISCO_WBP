import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


k=0
folders = ["UR-5e_1/wTool", "UR-5e_2"]
for i, folder in enumerate(folders):
    for j, file in enumerate(os.listdir(folder)):
        if ".csv" in file:
            df = pd.read_csv(f"{folder}/{file}")
            globals()[f"train{i}{j}"] = np.array(df.iloc[:,1])
            # plt.figure()
            # plt.plot(globals()[f"train{i}{j}"])


s = np.zeros([2,2])
s1 = np.zeros([2,2])

fig, ax = plt.subplots(2,2)
ax[0][1].set_title("UR-5e_2")
ax[0][0].set_title("UR-5e_1")
k=0
for i in range(2):
    for j in range(2):
        ax[i][j].plot(globals()[f"train{j}{i}"], label="Kernel")
        # plt.plot(globals()[f"train{i}{j}"], label=f"{label}")
        k+=1


for folder in folders:
    for num, file in enumerate(os.listdir(folder)):
        if num in range(2):
            continue
        if ".csv" in file:
            train_x = np.array(pd.read_csv(f"{folder}/{file}").iloc[:,1])

            for i in range(2):
                for j in range(2):
                    a=np.convolve(np.flip(train_x),globals()[f"train{j}{i}"])
                    s[i,j]=max(a)
            S = np.max(s)
            s1 = s/S
            print("Looks like plot #", s1.argmax()+1)
            ax[s1.argmax()//2][s1.argmax()%2].plot(train_x, label=f"{file}")
            ax[s1.argmax()//2][s1.argmax()%2].legend()
