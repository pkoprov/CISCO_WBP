import os

import numpy as np
import pandas as pd

from analysis.plotting import shift_for_maximum_correlation

sig_len_dic = {"UR": 4500, "VF": 9000, "PRUSA": 8500}
# asset = "VF-2"
# asset = "UR"
asset = "Prusa"

coord_list = ['all', 'x', 'y', 'z']
# create global variables to store x,y,z and xyz data
# asset_type = "VF"
# asset_type = "UR"
asset_type = "PRUSA"
home = rf"C:\Users\pkoprov\PycharmProjects\Vibration_Patterns\data\Kernels\{asset}\{asset_type}"
sig_len = sig_len_dic[asset_type]

# create a dictionary to store all data
for coor in coord_list:
    if coor == "all":
        globals()[f'df_{asset}_{coor}'] = np.empty((0, sig_len*3 + 1))
    else:
        globals()[f'df_{asset}_{coor}'] = np.empty((0, sig_len + 1))

for folder in os.listdir(home):
    if os.path.isdir(f"{home}/{folder}"):  # check if it is a folder
        print(folder)
        j = 0
        if asset in folder:
            for file in os.listdir(f"{home}/{folder}"):
                if '.csv' in file:
                    j += 1
                    df = np.array(pd.read_csv(f"{home}/{folder}/{file}"))[:, 1:]

                    # set the first df as a benchmark, and shift all following dfs to match it
                    if j == 1:
                        df_bench = df
                    else:
                        df, _ = shift_for_maximum_correlation(df_bench, df)

                    if df.shape[0] < sig_len:
                        df = np.vstack([df, np.full((sig_len - df.shape[0] ,3), df.mean(axis=0))])
                    else:
                        df = df[:sig_len]

                    X, Y, Z = [df[:, cor] for cor in range(df.shape[1])]  # split the data into x,y,z
                    ALL = np.hstack([X, Y, Z])  # concatenate x,y,z into one array

                    for coord, COORD in zip(coord_list, [ALL, X, Y, Z]):
                        # globals()[f'df_{asset}_{coord}'] = pd.concat(
                        #     [globals()[f'df_{asset}_{coord}'], globals()[f'{coord}_coord_df']], axis=1,
                        #     ignore_index=True)
                        globals()[f'df_{asset}_{coord}'] = np.vstack(
                            [globals()[f'df_{asset}_{coord}'], np.hstack([folder, COORD])])



import matplotlib.pyplot as plt

data = globals()[f'df_{asset}_{coord}'][:, 1:].astype(float)
for i in range(0,data.shape[0]):
    plt.plot(data[i], color="blue", alpha=0.3)

plt.figure(figsize=[1600, 900])
names = np.unique(globals()[f'df_{asset}_{coord}'][:,0])
n = -1
prev_name = str()
for name in names:
    if name != prev_name:
        n += 1
        prev_name = name
    named_data = data[globals()[f'df_{asset}_{coord}'][:,0] == name]
    for dat in named_data:
        plt.plot(dat + n*1, color="blue", alpha=0.3)


############################################################
# create files
############################################################
for coord in coord_list:
    try:
        pd.DataFrame(globals()[f'df_{asset}_{coord}']).to_csv(f"{home}/{asset}_{coord}.csv", index=False)  # save to csv
        print(f"Succesfully reated file: {asset}_{coord}.csv")
    except:
        print(f"Failed to create file: {asset}_{coord}.csv")
