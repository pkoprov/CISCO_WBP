import os

import matplotlib.pyplot as plt
from analysis.plotting import shift_for_maximum_correlation
import pandas as pd


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

# read data from each sample and merge into one dataframe
root = "data/Kernels/2023_02_07/UR-5e_Cary"

df_all = pd.DataFrame(columns=["Time"])
i = 0
for file in os.listdir(root):
    if ".csv" in file and "merged" not in file:
        i += 1
        df = pd.read_csv(f"{root}/{file}").iloc[:, :2]
        df.columns = ["Time", f"Sample_{i}"]
        df["Time"] = df["Time"].round(3)
        df.drop_duplicates(["Time"], inplace=True)
        df_all = df_all.merge(df, on="Time", how="outer")

# sort by time and take first 4000 samples for UR-5 and 8500 for VF-2
df_all = df_all.sort_values(by="Time").reset_index(drop=True).iloc[:8500, :]

# deal with missing values
for row in df_all.iterrows():
    if row[1].isna().any():
        row[1].fillna(row[1].median(), inplace=True)

# shift all samples so that they have maximum correlation with benchmark
for col in df_all.columns:
    if col == "Time":
        continue
    # shift df so that it has maximum correlation with df_benchmark
    try:
        df_all[col] = shift_for_maximum_correlation(benchmark, df_all[col])[0]
    except:
        benchmark = df_all[col]
else:
    del benchmark


# rename columns so they don't appear in legend
cols = ["_" + col for col in df_all.columns]
cols[0] = 'Time'
cols[1] = 'Samples'
df_all.columns = cols

# plot all samples
df_all.plot(x="Time", alpha=0.3, color="blue")

# plot median and 1.5std
median = df_all.iloc[:, 1:].median(axis=1)
std = df_all.iloc[:, 1:].std(axis=1)
plt.plot(df_all["Time"], median, color="black", label="Median")
plt.fill_between(df_all["Time"], median - 1.5 * std, median + 1.5 * std, color="grey", label="Median\u00b11.5std")
plt.legend()
plt.margins(1e-2)
plt.tight_layout()


df_all = df_all.transpose()
# set Time to be columns
df_all.columns = df_all.iloc[0]
df_all.drop("Time", inplace=True)
# rename indices from Samples to UR-5e_...
indices = [root.split("/")[-1][:7] for i in range(df_all.shape[0])]
df_all.index = indices
# save to csv
df_all.to_csv(f"{root}/merged.csv")

# merge all merged.csv files into one
## for VF-2
# df_total = df_total.append(pd.read_csv(f"{root}/{file}",index_col=0))
# for UR-5e
df_total = pd.DataFrame()
for folder in os.listdir("data/Kernels/2023_02_07"):
    if os.path.isdir(f"data/Kernels/2023_02_07/{folder}"):
        for file in os.listdir(f"data/Kernels/2023_02_07/{folder}"):
            if "merged.csv" in file:
                df_total = df_total.append(pd.read_csv(f"data/Kernels/2023_02_07/{folder}/{file}",index_col=0))

df_total.to_csv("data/Kernels/2023_02_07/merged.csv")
