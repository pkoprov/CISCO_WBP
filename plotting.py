import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def reduced(df, med):
    lim = 3474/2
    bound = [int(med-lim),int(med+lim)]
    return df[bound[0]: bound[1]]

length = 3474

folder = "270936"
for i, file in enumerate(os.listdir(f"2022_02_28/{folder}")):
    df = pd.read_csv(f"2022_02_28/{folder}/{file}").iloc[1:,1:4]
    globals()[f"df{i}"] = df
    plt.figure()
    plt.plot(df.iloc[:,1])


#########################################
# for 270904
# mid = 2850
# df0_ = np.array(reduced(df0,mid))
# mid = 1980
# df1_ = np.array(reduced(df1,mid))
# mid = 2650
# df2_ = np.array(reduced(df2,mid))
# mid = 2820
# df3_ = np.array(reduced(df3,mid))
# mid = 2960
# df4_ = np.array(reduced(df4,mid))
# mid = 2240
# df5_ = np.array(reduced(df5,mid))
#########################################
#########################################
# for 270936
# mid = 2680
# df0_ = np.array(reduced(df0,mid))
# mid = 2970
# df1_ = np.array(reduced(df1,mid))
# mid = 2270
# df2_ = np.array(reduced(df2,mid))
# mid = 1780
# df3_ = np.array(reduced(df3,mid))
# mid = 2790
# df4_ = np.array(reduced(df4,mid))
# mid = 2570
# df5_ = np.array(reduced(df5,mid))
#########################################
#########################################
# for 270904
# mid = 3050
# df0_ = np.array(reduced(df0,mid))
# mid = 2210
# df1_ = np.array(reduced(df1,mid))
# mid = 2820
# df2_ = np.array(reduced(df2,mid))
# mid = 1850
# df3_ = np.array(reduced(df3,mid))
# mid = 2760
# df4_ = np.array(reduced(df4,mid))
# mid = 2850
# df5_ = np.array(reduced(df5,mid))
#########################################
#########################################
# for 270328
# df0_ = np.array(df0[1637:])
# df1_ = np.array(df1[810:810+length])
# df2_ = np.array(df2[70:70+length])
# df3_ = np.array(df3[1325:1325+length])
# df4_ = np.array(df4[555:555+length])
# df5_ = np.array(df5[870:870+length])
#########################################

plt.figure()
for i in range(6):
    plt.plot(globals()[f"df{i}_"])

if not os.path.exists(f"2022_02_28/{folder}_ex"):
    os.mkdir(f"2022_02_28/{folder}_ex")

for file in os.listdir(f"2022_02_28/{folder}"):
    pd.DataFrame(globals()[f"df{i}_"]).to_csv(f"2022_02_28/{folder}_ex/{file}.csv", header=["X", "Y", "Z"], index=False)

