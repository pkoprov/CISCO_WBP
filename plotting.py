import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def reduced(df, med):
    lim = 3474/2
    if med < lim:
        zeros_quantity = int(lim - med)
        bound = [0,int(med+lim)]
        zero_df = pd.DataFrame([[0,9.8,1]]*zeros_quantity,columns=df.columns)
        df = pd.concat([zero_df, df])
    else:
        bound = [int(med-lim),int(med+lim)]
    answer = np.array(df[bound[0]: bound[1]])
    return answer


length = 3474
for folder in os.listdir(f"2022_02_28"):
    if "_ex" not in folder:
        for i, file in enumerate(os.listdir(f"2022_02_28/{folder}")):
            df = pd.read_csv(f"2022_02_28/{folder}/{file}").iloc[1:,1:4]
            globals()[f"df{i}"] = df
            # plt.figure()
            # plt.plot(df.iloc[:,1])

        #########################################
        match folder:
            case "270935":
                mid = 2815
                df0_ = reduced(df0,mid)
                mid = 1945
                df1_ = reduced(df1,mid)
                mid = 2615
                df2_ = reduced(df2,mid)
                mid = 2785
                df3_ = reduced(df3,mid)
                mid = 2925
                df4_ = reduced(df4,mid)
                mid = 2205
                df5_ = reduced(df5,mid)
        #########################################
        #########################################
            case "270936":
                mid = 2645
                df0_ = reduced(df0,mid)
                mid = 2935
                df1_ = reduced(df1,mid)
                mid = 2235
                df2_ = reduced(df2,mid)
                mid = 1745
                df3_ = reduced(df3,mid)
                mid = 2755
                df4_ = reduced(df4,mid)
                mid = 2535
                df5_ = reduced(df5,mid)
        #########################################
        #########################################
            case "270904":
                mid = 2980
                df0_ = reduced(df0,mid)
                mid = 2140
                df1_ = reduced(df1,mid)
                mid = 2750
                df2_ = reduced(df2,mid)
                mid = 1780
                df3_ = reduced(df3,mid)
                mid = 2690
                df4_ = reduced(df4,mid)
                mid = 2780
                df5_ = reduced(df5,mid)
        #########################################
        #########################################
            case "270328":
                mid = 3300
                df0_ = reduced(df0,mid)
                mid = 2470
                df1_ = reduced(df1,mid)
                mid = 1730
                df2_ = reduced(df2,mid)
                mid = 2990
                df3_ = reduced(df3,mid)
                mid = 2215
                df4_ = reduced(df4,mid)
                mid = 2530
                df5_ = reduced(df5,mid)
        #########################################


        # plt.figure()
        for i in range(6):
            # plt.plot(globals()[f"df{i}_"])
            plt.plot(globals()[f"df{i}_"][:, 1])

        if not os.path.exists(f"2022_02_28/{folder}_ex"):
            os.mkdir(f"2022_02_28/{folder}_ex")

        for i, file in enumerate(os.listdir(f"2022_02_28/{folder}")):
            pd.DataFrame(globals()[f"df{i}_"]).to_csv(f"2022_02_28/{folder}_ex/{file}", header=["X", "Y", "Z"], index=False)

plt.figure()
folder_ex = f"{folder}_ex"
for i, file in enumerate(os.listdir(f"2022_02_28/{folder_ex}")):
    df = pd.read_csv(f"2022_02_28/{folder_ex}/{file}")["X"]
    plt.plot(df)