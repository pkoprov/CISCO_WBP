import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = "data/Kernels/2023_02_07/UR-5e_Wilmington"
#
# class Data(pd.DataFrame):
#
#
#     def __init__(self, root_folder , length, *args, **kw):
#         super().__init__(columns=["Time"],*args, **kw)
#         self.root = root_folder
#         self.length = length
#
#     @property
#     def _constructor(self):
#         return Data
#
#     def __repr__(self):
#         return super().__repr__()
#
#     def load(self):
#         df_all = pd.DataFrame(columns=["Time"])
#         i = 0
#         for file in os.listdir(self.root):
#             if ".csv" in file:
#                 i+=1
#                 df = pd.read_csv(f"{self.root}/{file}").iloc[:self.length, :2]
#                 df.columns=["Time",f"Sample_{i}"]
#                 df["Time"] = df["Time"].round(3)
#                 df_all = df_all.merge(df, on = "Time", how="outer")
#         df_all = df_all.sort_values(by="Time").reset_index(drop=True).iloc[:self.length, :]
#         # deal with missing values
#         for row in df_all.iterrows():
#             if row[1].isna().any():
#                 row[1].fillna(np.median(row[1]), inplace=True)
#         # rename columns so they don't appear in legend
#         cols = ["_" + col for col in df_all.columns]
#         cols[0] = 'Time'
#         cols[1] = 'Samples'
#         df_all.columns = cols
#         return df_all
#
#
#
# df = Data(root, 4000)
# df_all = df.load()
#
#

df_all = pd.DataFrame(columns=["Time"])
i = 0
for file in os.listdir(root):
    if ".csv" in file:
        i += 1
        df = pd.read_csv(f"{root}/{file}").iloc[:, :2]
        df.columns = ["Time", f"Sample_{i}"]
        df["Time"] = df["Time"].round(3)
        df.drop_duplicates(["Time"], inplace=True)
        df_all = df_all.merge(df, on="Time", how="outer")

df_all = df_all.sort_values(by="Time").reset_index(drop=True).iloc[:4000, :]

# deal with missing values
for row in df_all.iterrows():
    if row[1].isna().any():
        row[1].fillna(row[1].median(), inplace=True)

# rename columns so they don't appear in legend
cols = ["_" + col for col in df_all.columns]
cols[0] = 'Time'
cols[1] = 'Samples'
df_all.columns = cols

# plot all samples
df_all.plot(x="Time", alpha=0.3, color="red")

median = df_all.iloc[:, 1:].median(axis=1)
std = df_all.iloc[:, 1:].std(axis=1)
plt.plot(df_all["Time"], median, color="black", label="Median")
plt.fill_between(df_all["Time"], median - 1.5 * std, median + 1.5 * std, color="grey", label="Median\u00b11.5std")
plt.legend()
plt.margins(1e-2)
plt.tight_layout()

df = pd.read_csv("data/Kernels/2023_01_25/tets/tets_1674659254.csv").iloc[:4000, 1:]
df = df - df.mean()
df.plot()
plt.hist(abs(df))
plt.boxplot(abs(df))
np.quantile(abs(df), 0.95)
df_abs_sorted = abs(df.to_numpy()).flatten()
df_abs_sorted.sort()
(np.where(df_abs_sorted > 0.0024 * 3)[0][0] - 1) / len(df_abs_sorted)
