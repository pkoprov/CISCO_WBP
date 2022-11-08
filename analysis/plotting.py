import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


# Function to extract the data of only moving part
def reduced(df, med):
    lim = 3474 / 2
    if med < lim:
        zeros_quantity = int(lim - med)
        zero_df = pd.DataFrame([[0, 9.8, 1]] * zeros_quantity, columns=df.columns)
        df = df.iloc[0:int(med + lim), :]
        df = pd.concat([zero_df, df])
        return np.array(df)
    else:
        bound = [int(med - lim), int(med + lim)]
        return np.array(df[bound[0]: bound[1]])


ur5_dict = {"270936":"UR5_1","270935":"UR5_2","270904":"UR5_3","270328":"UR5_4"}

length = 3474  # length of moving time

plt.figure()
n = 1
for folder in os.listdir(f"../2022_02_28"):
    if "_ex" not in folder:
        for i, file in enumerate(os.listdir(f"2022_02_28/{folder}")):
            df = pd.read_csv(f"2022_02_28/{folder}/{file}").iloc[1:, 1:4]
            globals()[f"df{i}"] = df
            # plt.figure()
            # plt.plot(df.iloc[:,1])

        match folder:
            case "270935":
                mid = 2815
                df0_ = reduced(df0, mid)
                mid = 1945
                df1_ = reduced(df1, mid)
                mid = 2615
                df2_ = reduced(df2, mid)
                mid = 2785
                df3_ = reduced(df3, mid)
                mid = 2925
                df4_ = reduced(df4, mid)
                mid = 2205
                df5_ = reduced(df5, mid)
            #########################################
            case "270936":
                mid = 2645
                df0_ = reduced(df0, mid)
                mid = 2935
                df1_ = reduced(df1, mid)
                mid = 2235
                df2_ = reduced(df2, mid)
                mid = 1745
                df3_ = reduced(df3, mid)
                mid = 2755
                df4_ = reduced(df4, mid)
                mid = 2535
                df5_ = reduced(df5, mid)
            #########################################
            case "270904":
                mid = 2980
                df0_ = reduced(df0, mid)
                mid = 2140
                df1_ = reduced(df1, mid)
                mid = 2750
                df2_ = reduced(df2, mid)
                mid = 1780
                df3_ = reduced(df3, mid)
                mid = 2690
                df4_ = reduced(df4, mid)
                mid = 2780
                df5_ = reduced(df5, mid)
            #########################################
            case "270328":
                mid = 3300
                df0_ = reduced(df0, mid)
                mid = 2470
                df1_ = reduced(df1, mid)
                mid = 1730
                df2_ = reduced(df2, mid)
                mid = 2990
                df3_ = reduced(df3, mid)
                mid = 2215
                df4_ = reduced(df4, mid)
                mid = 2530
                df5_ = reduced(df5, mid)
        #########################################
        for i in range(6):
            plt.subplot(2,2,n)
            plt.plot(globals()[f"df{i}_"])
        plt.title(ur5_dict.get(folder))
        n += 1

        if not os.path.exists(f"2022_02_28/{folder}_ex"):
            os.mkdir(f"2022_02_28/{folder}_ex")

        for i, file in enumerate(os.listdir(f"2022_02_28/{folder}")):
            pd.DataFrame(globals()[f"df{i}_"]).to_csv(f"2022_02_28/{folder}_ex/{file}", header=["X", "Y", "Z"],
                                                      index=False)

plt.figure()
for folder in os.listdir(f"../2022_02_28"):
    if "_ex" in folder:
        for file in os.listdir(f"2022_02_28/{folder}"):
            df = pd.read_csv(f"2022_02_28/{folder}/{file}")
            plt.plot(df)
