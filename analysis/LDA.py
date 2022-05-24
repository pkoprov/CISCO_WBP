import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler


# read data and create dataframes
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

##################################################################################################
# ____________________________________________ LDA ____________________________________________
##################################################################################################
plt.figure(figsize=(20, 12))

plt.suptitle('LDA for 2 class data')
target = Patch(color='r', label='target')
outlier = Patch(color='b', label='outlier')
for rob in range(1, 5):

    target_data = df_UR5_x.loc[:df_UR5_x.tail(1).index[0],
                  (df_UR5_x.tail(1) == f"UR-5e-{rob}").to_numpy()[0]].transpose()
    target_data.iloc[:, -1] = "target"
    outlier_data = df_UR5_x.loc[:df_UR5_x.tail(1).index[0],
                   (df_UR5_x.tail(1) != f"UR-5e-{rob}").to_numpy()[0]].transpose()
    outlier_data.iloc[:, -1] = "outlier"

    data = pd.concat([target_data, outlier_data])
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    sc = StandardScaler()
    X = sc.fit_transform(X)

    lda = LDA(n_components=1)
    X = lda.fit_transform(X, Y)

    plt.subplot(4, 1, rob)
    for name, data in zip(Y, X):
        c = 'b' if name == 'outlier' else 'r'
        plt.scatter(data, 0, c=c, label=name)
    plt.axvline(np.max(X[np.where(Y == 'outlier')]))
    plt.title(f"UR-5e_{rob}")
    plt.legend(handles=[target, outlier])
