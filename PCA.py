import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def fig():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.grid()


length = 3474
plt.ion()

coord_list = ['all','x','y','z']
# create global variables to store x,y,z and xyz data
for i in range(4):
    globals()[f'df_UR5_{coord_list[i]}'] = pd.DataFrame()

colors = ['r', 'g', 'b', 'k']

for folder in os.listdir(f"2022_02_28"):
    if "_ex" in folder:
        for file in os.listdir(f"2022_02_28/{folder}"):
            df = pd.read_csv(f"2022_02_28/{folder}/{file}")
            all_coord_df = pd.concat([df["X"],df["Y"],df["Z"],pd.Series(file[-11:-4])],ignore_index=True)
            x_coord_df = pd.concat([df["X"], pd.Series(file[-11:-4])], ignore_index=True)
            y_coord_df = pd.concat([df["Y"], pd.Series(file[-11:-4])], ignore_index=True)
            z_coord_df = pd.concat([df["Z"], pd.Series(file[-11:-4])], ignore_index=True)
            df_UR5_all = pd.concat([df_UR5_all,all_coord_df],axis=1, ignore_index=True)
            df_UR5_x = pd.concat([df_UR5_x, x_coord_df], axis=1, ignore_index=True)
            df_UR5_y = pd.concat([df_UR5_y, y_coord_df], axis=1, ignore_index=True)
            df_UR5_z = pd.concat([df_UR5_z, z_coord_df], axis=1, ignore_index=True)
#
# df_keys = []
# for coord in globals().keys():
#     if "df_UR5_" in coord:
#         df_keys.append(coord)
#
# for i, n in zip(range(4), [0,6,12,18]):
#     globals()[f"PCA{i}"], globals()[f"Vh{i}"], globals()[f"df{i}"] = getPCA(n,2)


#############################################################
df4 = df_UR5_all.iloc[:-1,0:6].astype('float').transpose().reset_index(drop=True)
df3 = df_UR5_all.iloc[:-1,6:12].astype('float').transpose().reset_index(drop=True)
df2 = df_UR5_all.iloc[:-1,12:18].astype('float').transpose().reset_index(drop=True)
df1 = df_UR5_all.iloc[:-1,18:].astype('float').transpose().reset_index(drop=True)
#############################################################
# projecting on UR5_4 PC
#############################################################


def drawPCA(df):
    num = int(df[-1])
    U, S, V = np.linalg.svd(globals()[df])

    for i in range(4):
        PCS = np.matmul(globals()[f"df{i+1}"], V.transpose())
        c = 0 if i == num-1 else 1
        plt.scatter(PCS[0], PCS[1], c=colors[c])
        plt.title(f'Projection on 2 PC of UR5_{num}_all',fontsize = 20)

plt.figure()
for n in range(4):
    plt.subplot(2, 2, n + 1)
    drawPCA(f"df{n+1}")
