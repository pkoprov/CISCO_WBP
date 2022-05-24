import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.model_selection import train_test_split


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


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
# ________________________________________ OneClass SVM _________________________________________
##################################################################################################

figure, axes = plt.subplots(2,2)
plt.suptitle('PCA for 2 class data')


for rob in range(1,5):
    n = rob - 1
    target_data = df_UR5_x.loc[:df_UR5_x.tail(2).index[0], (df_UR5_x.tail(1) == f"UR-5e-{rob}").to_numpy()[0]].transpose().astype("float")
    outlier_data = df_UR5_x.loc[:df_UR5_x.tail(2).index[0], (df_UR5_x.tail(1) != f"UR-5e-{rob}").to_numpy()[0]].transpose().astype("float")

    U, S, V = np.linalg.svd(target_data)

    for name, data in zip(['target', 'outlier'], [target_data, outlier_data]):
        globals()[f"PCS_{name}"] = np.matmul(data, V.transpose())
        c = 'b' if name == 'target' else 'r'
        axes[n//2][n%2].scatter(globals()[f"PCS_{name}"].iloc[:,-2], globals()[f"PCS_{name}"].iloc[:,-1], c = c, label=name)
    #
    # center = [PCS_target.iloc[:,-1].mean(), PCS_target.iloc[:,-2].mean()]
    # axes[n//2][n%2].scatter(center[0], center[1], c="k", s=100, label="center weight")
    # confidence_ellipse(PCS_target.iloc[:,-1], PCS_target.iloc[:,-2], axes[n//2][n%2], n_std=1.96, edgecolor='b')
    axes[n//2][n%2].title.set_text(f"Projection of PCS of UR-5e_{rob}")
    axes[n//2][n%2].legend()
    # axes[n // 2][n % 2].aspect_ratio(1)
#
# explained_variance_ = (S ** 2) / (target_data.shape[0] - 1)
# total_var = explained_variance_.sum()
# explained_variance_ratio_ = sum(explained_variance_[:2] / total_var)


X_train, X_test, y_train, y_test = train_test_split(df_UR5_x.head(-1).transpose(), df_UR5_x.tail(1).transpose(), test_size=0.3)  # 70% training and 30% test


target_train_data = X_train.iloc[(y_train == f"UR-5e-{rob}").to_numpy()].astype("float")
outlier_train_data = X_train.iloc[(y_train != f"UR-5e-{rob}").to_numpy()].astype("float")
target_test_data = X_test.iloc[(y_test == f"UR-5e-{rob}").to_numpy()].astype("float")
outlier_test_data = X_test.iloc[(y_test != f"UR-5e-{rob}").to_numpy()].astype("float")

U, S, V = np.linalg.svd(outlier_train_data)
names = ['target_train_data', 'outlier_train_data','target_test', 'outlier_test']
for name, data in zip(names, [target_train_data, outlier_train_data,target_test_data,outlier_test_data]):
    globals()[f"PCS_{name}"] = np.matmul(data, V.transpose())
    c = 'b' if 'target' in name else 'r'
    m = "*" if "test" in name else "."
    plt.scatter(globals()[f"PCS_{name}"].iloc[:,0], globals()[f"PCS_{name}"].iloc[:,1], c = c, marker =m, label=name)

#
# center = [PCS_target.iloc[:,-1].mean(), PCS_target.iloc[:,-2].mean()]
# axes[n//2][n%2].scatter(center[0], center[1], c="k", s=100, label="center weight")
# confidence_ellipse(PCS_target.iloc[:,-1], PCS_target.iloc[:,-2], axes[n//2][n%2], n_std=1.96, edgecolor='b')
axes[n//2][n%2].title.set_text(f"Projection of PCS of UR-5e_{rob}")
axes[n//2][n%2].legend()