import os

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Patch
from sklearn import metrics, svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


length = 3100

coord_list = ['all', 'x', 'y', 'z']
# create global variables to store x,y,z and xyz data
for i in range(4):
    globals()[f'df_UR5_{coord_list[i]}'] = pd.DataFrame()

home = "data/Kernels/5_7_2022"
for folder in os.listdir(home):
    # if "_ex" in folder:
    if os.path.isdir(f"{home}/{folder}"):
        for file in os.listdir(f"{home}/{folder}"):
            if '.csv' in file:
                df = pd.read_csv(f"{home}/{folder}/{file}")
                type = pd.Series(file[:7])
                X = df.iloc[:length, 1]
                Y = df.iloc[:length, 2]
                Z = df.iloc[:length, 3]
                all_coord_df = pd.concat([X, Y, Z, type], ignore_index=True)
                x_coord_df = pd.concat([X, type], ignore_index=True)
                y_coord_df = pd.concat([Y, type], ignore_index=True)
                z_coord_df = pd.concat([Z, type], ignore_index=True)
                df_UR5_all = pd.concat([df_UR5_all, all_coord_df], axis=1, ignore_index=True)
                df_UR5_x = pd.concat([df_UR5_x, x_coord_df], axis=1, ignore_index=True)
                df_UR5_y = pd.concat([df_UR5_y, y_coord_df], axis=1, ignore_index=True)
                df_UR5_z = pd.concat([df_UR5_z, z_coord_df], axis=1, ignore_index=True)

############################################################
# create dataframes for each robot
coord = "x"
for i in range(1, 5):
    globals()[f"df{i}"] = globals()[f"df_UR5_{coord}"].loc[:globals()[f"df_UR5_{coord}"].shape[0] - 2,
                          globals()[f"df_UR5_{coord}"].iloc[-1, :] == f"UR-5e-{i}"].iloc[:, :] \
        .astype('float').transpose().reset_index(drop=True)

############################################################
# projecting on UR5_4 PC
############################################################


def drawPCA(df, num):
    type = int(df[-1])
    U, S, V = np.linalg.svd(globals()[df])

    explained_variance_ = (S ** 2) / (globals()[df].shape[0] - 1)
    total_var = explained_variance_.sum()
    explained_variance_ratio_ = sum(explained_variance_[:num] / total_var)
    print(f"Explained variance of robot {type} by {num} of PC = {explained_variance_ratio_}")

    for i in range(4):
        PCS = np.matmul(globals()[f"df{i + 1}"], V.transpose())
        c = "g" if i == type - 1 else "r"
        # ax.scatter3D(PCS[0], PCS[1], PCS[2], c=colors[c])
        match num:
            case 1:
                plt.scatter(PCS[0], np.zeros(len(PCS[0])), c=c)
            case 2:
                plt.scatter(PCS[0], PCS[1], c=c)

        plt.title(f'Projection on {num} PC of UR5_{type}_{coord}', fontsize=20)


for n in range(4):
    plt.subplot(2, 2, n + 1)
    # plt.figure()
    # ax = plt.axes(projection='3d')
    drawPCA(f"df{n + 1}", 2)  # , ax)


#############################################################
# ____________ PCA from scikitlearn ____________
#############################################################
# Instantiate PCA
pca = PCA(n_components=2)
# pca = PCA(n_components=3)

data = df_UR5_all.transpose()

train = pd.concat([data.iloc[:, :]])  # , data.iloc[15:44, :], data.iloc[45:74, :], data.iloc[75:116, :]])
test = pd.concat([data.iloc[29:, :]])

model = pd.DataFrame(data=pca.fit_transform(StandardScaler().fit_transform(train.iloc[:, :-1])),
                     columns=['PC1', 'PC2'])
pred = pd.DataFrame(data=pca.transform(StandardScaler().fit_transform(test.iloc[:, :-1], train.iloc[:, :-1])),
                    columns=['PC1', 'PC2'])

# df = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3'])
target = pd.Series(train.iloc[:, -1], name='target')

result_df = pd.concat([model, target], axis=1)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1)
# ax = plt.axes(projection='3d')
ax.set_xlabel('First Principal Component ', fontsize=15)
ax.set_ylabel('Second Principal Component ', fontsize=15)
# ax.set_zlabel('Third Principal Component ', fontsize=15)
# ax.set_title('Principal Component Analysis (3PCs)', fontsize = 20)

targets = ["UR-5e-1", "UR-5e-2", "UR-5e-3", "UR-5e-4"]
colors = ['r', 'g', 'b', 'k']
for target, color in zip(targets, colors):
    indicesToKeep = df_UR5_all.transpose()[9300] == target
    ax.scatter(result_df.loc[indicesToKeep, 'PC1'],
               result_df.loc[indicesToKeep, 'PC2'],
               # result_df.loc[indicesToKeep, 'PC3'],
               c=color,
               s=50)
ax.legend(targets)
ax.grid()



