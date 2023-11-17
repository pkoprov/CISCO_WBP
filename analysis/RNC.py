from analysis.FDA import Sample
from plot_errors_from_FDA import plot_errors
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from skfda.misc.metrics import PairwiseMetric, linf_distance


if __name__ == '__main__':
    plt.ion()
    asset = input("""
    Which asset do you want to plot?
    Options:
    1. VF
    2. UR
    3. Prusa
    > """)

    match asset.upper():
        case '1' | 'VF':
            asset = 'VF'
        case '2' | 'UR':
            asset = 'UR'
        case '3' | 'PRUSA':
            asset = 'Prusa'

    data = pd.read_csv(rf'data\Kernels\2023_02_07\{asset}_merged.csv')
    sample = Sample(data)
    labels = sample.labels
    unique_labels = labels.unique()
    pickled_data = f'data/{asset}_data.pkl'

    if not os.path.exists(pickled_data):
        fd_dict = sample.FData()
        with open(pickled_data, 'wb') as f:
            pickle.dump(fd_dict, f)
    else:
        with open(pickled_data, 'rb') as f:
            fd_dict = pickle.load(f)

    key = 'top'
    X = fd_dict[key]

    # X = sample.numeric
    y_l = labels.values
    for label in unique_labels:

        y = (y_l == label).astype(int)
        
        X_train, X_test, y_train_l, y_test_l = train_test_split(X,y_l,test_size=0.25,shuffle=True,
                                                            stratify=y,random_state=0)
        
        y_train = (y_train_l == label).astype(int)
        y_test = (y_test_l == label).astype(int)


        
        c1_train_ind = np.where(y_train_l == label)[0]
        c1_test_ind = np.where(y_test_l == label)[0]
        radius = np.std(X_train[c1_train_ind].data_matrix, axis=0).reshape(-1,1)
        radius = np.std(X_train.loc[y_train_l==label], axis=0).values
        sample = X_train[y_train_l==label].mean()  # Center of the ball

        # fig = X_train.plot(group=y_train, group_colors=['C0', 'C1'])

        # Plot ball
        # sample.plot(fig=fig, color='red', linewidth=3)
        # lower = sample.data_matrix - radius
        # upper = sample.data_matrix + radius

        # fig.axes[0].fill_between(sample.grid_points[0], lower.flatten(), upper.flatten(), alpha=0.25, color='C1')
       

        # Creation of pairwise distance
        l_inf = PairwiseMetric(linf_distance)
        distances = l_inf(sample, X_train)[0]  # L_inf distances to 'sample'

        margin = distances[y_train_l == label].mean()+distances[y_train_l == label].std()
        # Plot samples in the ball
        # X_train[distances <= margin].plot(axes= plt.gca(),color='C0', alpha = 0.2)
        # sample.plot(axes = plt.gca(), color='red', linewidth=3)
        # plt.plot(sample.grid_points[0], lower.flatten(), color='red', linestyle=':', alpha = 0.5)
        # plt.plot(sample.grid_points[0], upper.flatten(), color='red', linestyle=':', alpha = 0.5)

        # y_train_l[distances <= margin]
        test_distances = l_inf(sample, X_test)[0]

        act_pos = np.where(y_test_l!=label)[0]
        act_neg = np.where(y_test_l==label)[0]
        pred_pos = np.where(test_distances > margin)[0]  
        pred_neg = np.where(test_distances <= margin)[0]

        TP = len(np.intersect1d(pred_pos, act_pos))
        TN = len(np.intersect1d(pred_neg, act_neg))
        FP = len(np.intersect1d(pred_pos, act_neg))
        FN = len(np.intersect1d(pred_neg, act_pos))

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        F1 = 2 * (precision * sensitivity) / (sensitivity + precision)
        confusion_matrix = np.array([[TP, FN], [FP, TN]])
        print(confusion_matrix)
        print(f"Sensitivity: {sensitivity}, Specificity: {specificity}, Precision: {precision}, F1 score: {F1}")

        input("Press enter to continue")
