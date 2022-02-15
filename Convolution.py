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
            # plt.figure(k)
            # plt.plot(globals()[f"train{i}{j}"])
            # k+=1


s = np.zeros([2,3])
s1 = np.zeros([2,3])
dat_range = [[4500,6500],[5500,7500]]

k=0
for i in range(2):
    m,n = dat_range[i]
    for j in range(6):
        if i==0 and j==0:
            train00=train00[5000:7000]
        elif i==0 and j==5:
            train05 = train05[5500:7500]
        else:
            globals()[f"train{i}{j}"] = globals()[f"train{i}{j}"][m:n]
        plt.figure(k)
        plt.plot(range(m,n), globals()[f"train{i}{j}"])
        k+=1

k=0
for i in range(2):
    m,n = dat_range[i]
    for j in range(3):
        plt.figure(k)
        plt.plot(globals()[f"train{i}{j}"])
        k+=1



folder = "UR-5e_2/"
for file in os.listdir(folder):
    if ".csv" in file:
        train_x = np.array(pd.read_csv(f"{folder}{file}").iloc[:,1])

        for i in range(2):
            m, n = dat_range[i]
            for j in range(3):
                a=np.convolve(np.flip(train_x),globals()[f"train{i}{j}"])
                s[i,j]=max(a)
        S = np.max(s)
        s1 = s/S
        print(s1.argmax())
        plt.figure(s1.argmax())
        plt.plot(range(len(train_x)), train_x)