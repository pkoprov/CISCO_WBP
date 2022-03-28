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


def getPCA(pos, comp_num):
    i = int(input(f"Pick df (0..3): {[str(n)+'). ' + key for n, key in enumerate(df_keys)]}"))
    if i > 3:
        print("Incorrect index")
        pass
    try:
        df = globals()[df_keys[i]].iloc[:,pos:pos+6].transpose()
    except:
        print("Incorrect position")
        pass
    df = df.iloc[:,:-1].astype('float')
    mu = np.mean(df, axis = 0)
    df = df-mu
    df = np.array(df)
    U,S,Vh = np.linalg.svd(df)
    manPCA = pd.DataFrame(np.matmul(df,Vh)).iloc[:,:comp_num]
    return manPCA, Vh, df


length = 3474
plt.ion()

coord_list = ['all','x','y','z']
# create global variables to store x,y,z and xyz data
for i in range(4):
    globals()[f'df_UR5_{coord_list[i]}'] = pd.DataFrame()

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

df_keys = []
for coord in globals().keys():
    if "df_UR5_" in coord:
        df_keys.append(coord)

for i, n in zip(range(4), [0,6,12,18]):
    globals()[f"PCA{i}"], globals()[f"Vh{i}"], globals()[f"df{i}"] = getPCA(n,2)
colors = ['r', 'g', 'b', 'k']
#############################################################
# projecting on UR5_4 PC
#############################################################
fig()
plt.title(f'Projection on 2 PC of UR5_4_X',fontsize = 20)

plt.scatter(PCA0[0],PCA0[1], c = colors[0])

for i in [1,2,3]:
    PCS = pd.DataFrame(np.matmul(globals()[f"df{i}"],Vh0)).iloc[:,:2]
    plt.scatter(PCS[0],PCS[1], c = colors[i])


plt.legend(["UR5_4","UR5_3","UR5_2","UR5_1"])
#############################################################
# projecting on UR5_1 PC
#############################################################
fig()
plt.title(f'Projection on 2 PC of UR5_1_X',fontsize = 20)

plt.scatter(PCA3[0],PCA3[1], c = colors[3])

for i in [0,1,2]:
    PCS = pd.DataFrame(np.matmul(globals()[f"df{i}"],Vh3)).iloc[:,:2]
    plt.scatter(PCS[0],PCS[1], c = colors[i])


plt.legend(["UR5_1","UR5_4","UR5_3","UR5_2"])

#############################################################
# projecting on UR5_3 PC
#############################################################
fig()
plt.title(f'Projection on 2 PC of UR5_3_X',fontsize = 20)

plt.scatter(PCA2[0],PCA2[1], c = colors[2])

for i in [0,1,3]:
    PCS = pd.DataFrame(np.matmul(globals()[f"df{i}"],Vh2)).iloc[:,:2]
    plt.scatter(PCS[0],PCS[1], c = colors[i])


plt.legend(["UR5_3","UR5_4","UR5_2","UR5_1"])
#############################################################
# projecting on UR5_2 PC
#############################################################
fig()
plt.title(f'Projection on 2 PC of UR5_2_X',fontsize = 20)

plt.scatter(PCA1[0],PCA1[1], c = colors[1])

for i,c in zip([0,2,3], "bkg"):
    PCS = pd.DataFrame(np.matmul(globals()[f"df{i}"],Vh1)).iloc[:,:2]
    plt.scatter(PCS[0],PCS[1], c = colors[i])


plt.legend(["UR5_2","UR5_4","UR5_2","UR5_1"])


# df = df_UR5_x.iloc[:-1,:6].transpose()
# x = StandardScaler().fit_transform(df.iloc[:,:-1])


#############################################################
# 2 Principal components
#############################################################
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)
# principalComponents = pca.fit_transform(UR5_4)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA with X from sklearn', fontsize = 20)
colors = ['r', 'g', 'b', 'k']
ax.scatter(principalDf['PC1'], principalDf['PC2'], c = 'r', s = 50)


for target, color in zip(targets,colors):
    indicesToKeep = df_UR5.iloc[:,-1] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'PC1'], principalDf.loc[indicesToKeep, 'PC2'], c = color, s = 50)
ax.legend(targets)
ax.grid()

#############################################################
# 3 Principal components
#############################################################
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3'])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA with XYZ', fontsize = 20)
colors = ['r', 'g', 'b', 'k']

for target, color in zip(targets,colors):
    indicesToKeep = df_UR5.iloc[:,-1] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'PC1'], principalDf.loc[indicesToKeep, 'PC2'],
               principalDf.loc[indicesToKeep, 'PC3'], c = color, s = 50)
ax.legend(targets)

