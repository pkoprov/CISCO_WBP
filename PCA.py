import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

length = 3474
plt.ion()
df_UR5 = pd.DataFrame()
for folder in os.listdir(f"2022_02_28"):
    if "_ex" in folder:
        for file in os.listdir(f"2022_02_28/{folder}"):
            df = pd.read_csv(f"2022_02_28/{folder}/{file}")
            all_coord_df = pd.concat([df["X"],df["Y"],df["Z"],pd.Series(file[-11:-4])],ignore_index=True)
            df_UR5= pd.concat([df_UR5,all_coord_df],axis=1, ignore_index=True)

df_UR5 = df_UR5.transpose()
targets = df_UR5.iloc[:,-1].unique()
x = StandardScaler().fit_transform(df_UR5.iloc[:,:-1])
#############################################################
# 2 Principal components
#############################################################
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA with XYZ', fontsize = 20)
colors = ['r', 'g', 'b', 'k']

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